### Load arguments

import sys, getopt
import tensorflow as tf
tf.compat.v1.disable_v2_behavior() # <-- HERE !
### other parameters
# number of background sequences to take an expectation over
ns=10
# number of dinucleotide shuffled sequences per sequence as background
dinuc_shuffle_n=100

def main(argv):
   model_ID = ''
   try:
      opts, args = getopt.getopt(argv,"hm:s:c:b:",["model=", "sequence_set=", "bg="])
   except getopt.GetoptError:
      print('run_DeepSHAP_DeepExplainer.py -m <CNN model file> -s <sequence set> -b <random/dinuc_shuffle>')
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         print('run_DeepSHAP_DeepExplainer.py -m <CNN model file> -s <sequence set> -b <random/dinuc_shuffle>')
         sys.exit()
      elif opt in ("-m", "--model"):
         model_ID = arg
      elif opt in ("-s", "--sequence_set"):
         sequence_set = arg
      elif opt in ("-b", "--bg"):
         bg = arg
   if model_ID=='': sys.exit("CNN model file not found")
   if sequence_set=='': sys.exit("sequence_set not found")
   if bg=='': sys.exit("background not found")
   print('CNN model file is ', model_ID)
   print('sequence_set is ', sequence_set)
   print('background is ', bg)
   return model_ID, sequence_set, bg

if __name__ == "__main__":
   model_ID, sequence_set, bg = main(sys.argv[1:])

# output files to ...
import os
model_out_path=model_ID

### Load libraries

import pandas as pd


# sys.path.append('bin/')
# from helper import IOHelper, SequenceHelper # from https://github.com/bernardo-de-almeida/Neural_Network_DNA_Demo.git

nn_demo_path = '/camp/home/weie/lab_space_weie/training/run_deepexplainer/bin/Neural_Network_DNA_Demo'
sys.path.append(nn_demo_path)
from helper import IOHelper, SequenceHelper

import random
random.seed(1234)


### Functions
def one_hot_encode_along_channel_axis(sequence):
    to_return = np.zeros((len(sequence),4), dtype=np.int8)
    seq_to_one_hot_fill_in_array(zeros_array=to_return,
                                 sequence=sequence, one_hot_axis=1)
    return to_return
def seq_to_one_hot_fill_in_array(zeros_array, sequence, one_hot_axis):
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}  # Explicit mapping

    assert one_hot_axis == 0 or one_hot_axis == 1
    if one_hot_axis == 0:
        assert zeros_array.shape[1] == len(sequence)
    elif one_hot_axis == 1:
        assert zeros_array.shape[0] == len(sequence)

    # Fill the array based on the mapping
    for i, char in enumerate(sequence):
        char = char.upper()  # Convert to uppercase
        if char in mapping:
            char_idx = mapping[char]
            if one_hot_axis == 0:
                zeros_array[char_idx, i] = 1
            elif one_hot_axis == 1:
                zeros_array[i, char_idx] = 1
        elif char == 'N':  # Handle 'N' as all zeros
            continue
        else:
            raise RuntimeError(f"Unsupported character: {char}")


def prepare_input(fasta):
    # Convert sequences to one-hot encoding matrix
    file_seq = fasta
    input_fasta_data_A = IOHelper.get_fastas_from_file(file_seq, uppercase=True)

    # Number of sequences and the length of each sequence
    num_sequences = len(input_fasta_data_A.sequence)
    sequence_length = len(input_fasta_data_A.sequence.iloc[0])

    # Create an empty array to hold the one-hot encoded sequences
    seq_matrix_A = np.zeros((num_sequences, sequence_length, 4), dtype=np.int8)

    # Fill the array with one-hot encoded values
    for i, sequence in enumerate(input_fasta_data_A.sequence):
        seq_to_one_hot_fill_in_array(seq_matrix_A[i], sequence, one_hot_axis=1)
    
    # Print a sample of the one-hot encoded sequence
    print(seq_matrix_A[0, 0:20, :])

    # Replace NaN with zero (not usually needed, as there should be no NaNs)
    X = np.nan_to_num(seq_matrix_A)

    # Reshape if needed (this step seems redundant in this case, but kept for consistency)
    X_reshaped = X.reshape((X.shape[0], X.shape[1], X.shape[2]))

    return X_reshaped

# def load_model(model):
#     model = tf.keras.models.load_model(model)
#     #keras_model.summary()
#     return model
def pearson_correlation(y_true, y_pred):
    x = y_true - tf.reduce_mean(y_true)
    y = y_pred - tf.reduce_mean(y_pred)
    correlation = tf.reduce_sum(x * y) / (tf.sqrt(tf.reduce_sum(tf.square(x))) * tf.sqrt(tf.reduce_sum(tf.square(y))))
    return correlation

def load_model(model_ID):
    # Define custom objects dictionary
    custom_objects = {'pearson_correlation': pearson_correlation}
    try:
        model = tf.keras.models.load_model(model_ID, custom_objects=custom_objects)
        return model
    except IOError:
        raise IOError(f"Error loading model from {model_ID}")
        
from deeplift.dinuc_shuffle import dinuc_shuffle
import numpy as np


### deepExplainer functions
def dinuc_shuffle_several_times(list_containing_input_modes_for_an_example,
                                seed=1234):
  assert len(list_containing_input_modes_for_an_example)==1
  onehot_seq = list_containing_input_modes_for_an_example[0]
  rng = np.random.RandomState(seed)
  to_return = np.array([dinuc_shuffle(onehot_seq, rng=rng) for i in range(dinuc_shuffle_n)])
  return [to_return] #wrap in list for compatibility with multiple modes

# get hypothetical scores also
def combine_mult_and_diffref(mult, orig_inp, bg_data):
    assert len(orig_inp)==1
    projected_hypothetical_contribs = np.zeros_like(bg_data[0]).astype("float")
    assert len(orig_inp[0].shape)==2
    #At each position in the input sequence, we iterate over the one-hot encoding
    # possibilities (eg: for genomic sequence, this is ACGT i.e.
    # 1000, 0100, 0010 and 0001) and compute the hypothetical 
    # difference-from-reference in each case. We then multiply the hypothetical
    # differences-from-reference with the multipliers to get the hypothetical contributions.
    #For each of the one-hot encoding possibilities,
    # the hypothetical contributions are then summed across the ACGT axis to estimate
    # the total hypothetical contribution of each position. This per-position hypothetical
    # contribution is then assigned ("projected") onto whichever base was present in the
    # hypothetical sequence.
    #The reason this is a fast estimate of what the importance scores *would* look
    # like if different bases were present in the underlying sequence is that
    # the multipliers are computed once using the original sequence, and are not
    # computed again for each hypothetical sequence.
    for i in range(orig_inp[0].shape[-1]):
        hypothetical_input = np.zeros_like(orig_inp[0]).astype("float")
        hypothetical_input[:,i] = 1.0
        hypothetical_difference_from_reference = (hypothetical_input[None,:,:]-bg_data[0])
        hypothetical_contribs = hypothetical_difference_from_reference*mult[0]
        projected_hypothetical_contribs[:,:,i] = np.sum(hypothetical_contribs,axis=-1) 
    return [np.mean(projected_hypothetical_contribs,axis=0)]

def my_deepExplainer(model, one_hot, bg):
    sys.path.insert(0, '/camp/home/weie/lab_space_weie/training/run_deepexplainer/bin/shap')
    import shap
    print("SHAP module path:", shap.__file__)
    import numpy as np
        # output layer
    out_layer=-1
    if bg=="random":
        np.random.seed(seed=1111)
        background = one_hot[np.random.choice(one_hot.shape[0], ns, replace=False)]
        explainer = shap.DeepExplainer((model.layers[0].input, model.layers[out_layer].output),
          data=background,
          combine_mult_and_diffref=combine_mult_and_diffref)
    if bg=="dinuc_shuffle":
        explainer = shap.DeepExplainer((model.layers[0].input, model.layers[out_layer].output),
          data=dinuc_shuffle_several_times,
          combine_mult_and_diffref=combine_mult_and_diffref)
        
    # running on all sequences
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



print("\nLoading sequences and model ...\n")

X_all = prepare_input(sequence_set)
keras_model = load_model(model_ID)



print("\nRunning DeepExplain ...\n")

scores=my_deepExplainer(keras_model, X_all, bg=bg)
# scores=my_deepExplainer(keras_model, X_all[0:7392], bg=bg) # for testing

print("\nSaving ...\n")

import h5py
import os

sequence_set_out=os.path.basename(os.path.normpath(sequence_set))


if (os.path.isfile(model_out_path+"_"+sequence_set_out+"_"+bg+"_deepSHAP_DeepExplainer_importance_scores.h5")):
    os.remove(str(model_out_path+"_"+sequence_set_out+"_"+bg+"_deepSHAP_DeepExplainer_importance_scores.h5"))
f = h5py.File(model_out_path+"_"+sequence_set_out+"_"+bg+"_deepSHAP_DeepExplainer_importance_scores.h5", "w")

g = f.create_group("contrib_scores")
# save the actual contribution scores
g.create_dataset("class", data=scores[1])
print("Done for contrib_scores")

g = f.create_group("hyp_contrib_scores")
# save the hypothetical contribution scores
g.create_dataset("class", data=scores[0])
print("Done for hyp_contrib_scores")

f.close()



### print scores of same enhancers
print("\nPrint DeepExplain scores for some examples ...\n")

from deeplift.visualization import viz_sequence


### print scores of specific enhancers
print("\nPrint DeepExplain scores for some examples ...\n")

from deeplift.visualization import viz_sequence
import matplotlib.backends.backend_pdf
example_indices = [0, 199, 399, 599, 799]

contrib_scores = scores[1]
print("Dimension of contrib_scores: ", contrib_scores.shape)
print(contrib_scores[0, 0, 0:20, :])
# Define the length of the window to visualize

# Create the PDF output file


import os
from deeplift.visualization import viz_sequence
import matplotlib.backends.backend_pdf

# Ensure the directory exists
# output_dir = "adora"
# os.makedirs(output_dir, exist_ok=True)

example_indices = [0, 99, 199, 299, 399, 499, 599, 699, 799, 899]

contrib_scores = scores[1]
print("Dimension of contrib_scores: ", contrib_scores.shape)

# Define the length of the window to visualize
window_size = 100

# Loop over the example indices and save each as a separate PDF
for i in example_indices:
    selected_data = contrib_scores[0, 0, i:i+window_size, :]  # Adjusted indexing
    selected_data = np.squeeze(selected_data)
    pdf_output_file = os.path.join(f"deepSHAP_DeepExplainer_scores_examples_D9_pM_{i}.pdf")
    pdf = matplotlib.backends.backend_pdf.PdfPages(pdf_output_file)
    # Plot and save the selected window
    pdf.savefig(viz_sequence.plot_weights(selected_data, subticks_frequency=20))
    pdf.close()

print("\nEnded\n")