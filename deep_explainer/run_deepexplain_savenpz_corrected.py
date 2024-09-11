import sys
import os
import argparse
import h5py
import numpy as np
import tensorflow as tf
from deeplift.dinuc_shuffle import dinuc_shuffle
from deeplift.visualization import viz_sequence
import matplotlib.backends.backend_pdf

nn_demo_path = '/camp/home/weie/lab_space_weie/training/run_deepexplainer/bin/Neural_Network_DNA_Demo'
sys.path.append(nn_demo_path)
from helper import IOHelper

# Disable TensorFlow v2 behavior (if needed)
tf.compat.v1.disable_v2_behavior()
dinuc_shuffle_n = 100
ns = 10

# Define functions for loading model, preparing input, and DeepExplain
def load_model(model_ID):
    # Define custom objects dictionary
    custom_objects = {'pearson_correlation': pearson_correlation}
    try:
        model = tf.keras.models.load_model(model_ID, custom_objects=custom_objects)
        return model
    except IOError:
        raise IOError(f"Error loading model from {model_ID}")

def one_hot_encode(sequence):
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    seq_len = len(sequence)
    one_hot_matrix = np.zeros((seq_len, 4), dtype=np.int8)
    
    for i, nucleotide in enumerate(sequence):
        if nucleotide.upper() in mapping:
            one_hot_matrix[i, mapping[nucleotide.upper()]] = 1
    
    return one_hot_matrix

def prepare_input(sequence_set):
    # Convert sequences to one-hot encoding matrix
    input_fasta_data_A = IOHelper.get_fastas_from_file(sequence_set, uppercase=True)

    # Number of sequences and the length of each sequence
    num_sequences = len(input_fasta_data_A.sequence)
    sequence_length = len(input_fasta_data_A.sequence.iloc[0])

    # Create an empty array to hold the one-hot encoded sequences
    seq_matrix_A = np.zeros((num_sequences, sequence_length, 4), dtype=np.int8)

    # Fill the array with one-hot encoded values
    for i, sequence in enumerate(input_fasta_data_A.sequence):
        seq_matrix_A[i] = one_hot_encode(sequence)

    # Debugging: Print the first few sequences to check one-hot encoding
    print(seq_matrix_A[0, 0:20, :])  # Print first 20 nucleotides of the first sequence
    
    X = np.nan_to_num(seq_matrix_A)  # Replace NaN with zero and infinity with large finite numbers

    return X

def pearson_correlation(y_true, y_pred):
    x = y_true - tf.reduce_mean(y_true)
    y = y_pred - tf.reduce_mean(y_pred)
    correlation = tf.reduce_sum(x * y) / (tf.sqrt(tf.reduce_sum(tf.square(x))) * tf.sqrt(tf.reduce_sum(tf.square(y))))
    return correlation

### deepExplainer functions
def dinuc_shuffle_several_times(list_containing_input_modes_for_an_example,
                                seed=1234):
    assert len(list_containing_input_modes_for_an_example) == 1
    onehot_seq = list_containing_input_modes_for_an_example[0]
    rng = np.random.RandomState(seed)
    to_return = np.array([dinuc_shuffle(onehot_seq, rng=rng) for i in range(dinuc_shuffle_n)])
    return [to_return]  # wrap in list for compatibility with multiple modes

# get hypothetical scores also
def combine_mult_and_diffref(mult, orig_inp, bg_data):
    assert len(orig_inp) == 1
    projected_hypothetical_contribs = np.zeros_like(bg_data[0]).astype("float")
    assert len(orig_inp[0].shape) == 2
    # At each position in the input sequence, we iterate over the one-hot encoding
    #  possibilities (eg: for genomic sequence, this is ACGT i.e.
    #  1000, 0100, 0010 and 0001) and compute the hypothetical 
    #  difference-from-reference in each case. We then multiply the hypothetical 
    #  differences-from-reference with the multipliers to get the hypothetical contributions. 
    #  For each of the one-hot encoding possibilities, 
    #  the hypothetical contributions are then summed across the ACGT axis to estimate 
    #  the total hypothetical contribution of each position. This per-position hypothetical 
    #  contribution is then assigned ("projected") onto whichever base was present in the 
    #  hypothetical sequence. 
    #  The reason this is a fast estimate of what the importance scores *would* look 
    #  like if different bases were present in the underlying sequence is that 
    #  the multipliers are computed once using the original sequence, and are not 
    #  computed again for each hypothetical sequence.
    for i in range(orig_inp[0].shape[-1]):
        hypothetical_input = np.zeros_like(orig_inp[0]).astype("float")
        hypothetical_input[:, i] = 1.0
        hypothetical_difference_from_reference = (hypothetical_input[None, :, :] - bg_data[0])
        hypothetical_contribs = hypothetical_difference_from_reference * mult[0]
        projected_hypothetical_contribs[:, :, i] = np.sum(hypothetical_contribs, axis=-1)
    return [np.mean(projected_hypothetical_contribs, axis=0)]

def my_deepExplainer(model, one_hot, bg):
    sys.path.insert(0, '/camp/home/weie/lab_space_weie/training/run_deepexplainer/bin/shap')
    import shap
    print("SHAP module path:", shap.__file__)

    # Output layer
    out_layer = -1
    if bg == "random":
        np.random.seed(seed=1111)
        background = one_hot[np.random.choice(one_hot.shape[0], ns, replace=True)]
        explainer = shap.DeepExplainer((model.layers[0].input, model.layers[out_layer].output),
                                       data=background,
                                       combine_mult_and_diffref=combine_mult_and_diffref)
    elif bg == "dinuc_shuffle":
        explainer = shap.DeepExplainer((model.layers[0].input, model.layers[out_layer].output),
                                       data=dinuc_shuffle_several_times,
                                       combine_mult_and_diffref=combine_mult_and_diffref)
    else:
        raise ValueError("Unsupported background type. Choose either 'random' or 'dinuc_shuffle'.")

    # Running on all sequences
    shap_values_hypothetical = explainer.shap_values(one_hot)
    print(np.array(shap_values_hypothetical))

    # Adjust shapes for element-wise multiplication
    shap_values_hypothetical = np.array(shap_values_hypothetical)
    if shap_values_hypothetical.shape[-2] == 1:
        shap_values_hypothetical = np.squeeze(shap_values_hypothetical, axis=-2)

    print("Adjusted shape of shap_values_hypothetical:", shap_values_hypothetical.shape)

    # Perform element-wise multiplication
    shap_values_contribution = shap_values_hypothetical * one_hot

    return shap_values_hypothetical, shap_values_contribution

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run DeepSHAP DeepExplainer script with custom output filename")
    parser.add_argument('-o', '--output', type=str, required=True,
                        help='Desired output .h5 filename')
    parser.add_argument('-m', '--model', type=str, required=True,
                        help='CNN model file')
    parser.add_argument('-s', '--sequence_set', type=str, required=True,
                        help='Sequence set')
    parser.add_argument('-b', '--bg', type=str, required=True,
                        choices=['random', 'dinuc_shuffle'],
                        help='Background type: random or dinuc_shuffle')
    return parser.parse_args()

if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_arguments()

    # Assign argparse arguments to variables
    output_filename = args.output
    model_ID = args.model
    sequence_set = args.sequence_set
    bg = args.bg

    # Load sequences and model
    print("\nLoading sequences and model ...\n")
    X_all = prepare_input(sequence_set)
    keras_model = load_model(model_ID)

    # Run DeepExplain
    print("\nRunning DeepExplain ...\n")
    scores = my_deepExplainer(keras_model, X_all, bg=bg)

    # Save importance scores
    print("\nSaving ...\n")
    if os.path.isfile(output_filename):
        os.remove(output_filename)

    with h5py.File(output_filename, "w") as f:
        g = f.create_group("contrib_scores")
        g.create_dataset("class", data=scores[1])
        print("Done for contrib_scores")

        g = f.create_group("hyp_contrib_scores")
        g.create_dataset("class", data=scores[0])
        print("Done for hyp_contrib_scores")

    # Define filename based on sequence_set
    filename = os.path.splitext(os.path.basename(sequence_set))[0]

    # Transpose and save reshaped contrib_scores
    reshaped_contrib_scores = scores[1].transpose(0, 2, 1, 3)  # Transpose to (1, 1001, 4004, 4)
    np.savez(f'/camp/home/weie/lab_space_weie/training/TF-modisco/newpos_only/contrib_scores{filename}.npz',reshaped_contrib_scores) 

    # Save onehot encoded
    # np.savez(f'/camp/home/weie/lab_space_weie/training/TF-modisco/interval_onehot_{filename}.npz', X_all.transpose(0, 2, 1))
