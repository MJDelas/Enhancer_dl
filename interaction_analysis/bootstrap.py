import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K

def generate_random_sequence(seq_length):
    bases = ['A', 'T', 'G', 'C']
    prob = [0.2588, 0.2588, 0.2412, 0.2412]
    sequence = ''.join(np.random.choice(bases, seq_length, p=prob))
    return sequence

# Custom Pearson Correlation function
def pearson_correlation(y_true, y_pred):
    x = y_true - tf.reduce_mean(y_true)
    y = y_pred - tf.reduce_mean(y_pred)
    correlation = tf.reduce_sum(x * y) / (tf.sqrt(tf.reduce_sum(tf.square(x))) * tf.sqrt(tf.reduce_sum(tf.square(y))))
    return correlation

# Load model function
def load_model_with_custom_objects(model_ID):
    custom_objects = {'pearson_correlation': pearson_correlation}
    try:
        model = load_model(model_ID, custom_objects=custom_objects)
        return model
    except IOError:
        raise IOError(f"Error loading model from {model_ID}")

# Generate random sequences with central bases replaced
def generate_random_replacement_sequences(sequence, num_replacements, motif_length):
    replacements = []
    center_start = len(sequence) // 2 - motif_length // 2
    center_end = center_start + motif_length

    for _ in range(num_replacements):
        random_center = ''.join(np.random.choice(['A', 'T', 'G', 'C'], motif_length))
        new_sequence = sequence[:center_start] + random_center + sequence[center_end:]
        replacements.append(new_sequence)

    return replacements

# Insert motif into sequence
def insert_motif(sequence, motif, position):
    if 'x' in motif:
        parts = motif.split('x')
        prefix = ''.join(parts[:-1])
        suffix = parts[-1]
        num_x = len(parts) - 1
        replacement_seq = sequence[position + len(prefix):position + len(prefix) + num_x]
        new_sequence = sequence[:position] + prefix + replacement_seq + suffix + sequence[position + len(motif):]
    else:
        new_sequence = sequence[:position] + motif + sequence[position + len(motif):]
    
    if len(new_sequence) != len(sequence):
        raise ValueError(f"Final sequence length is {len(new_sequence)}, but expected {len(sequence)}")
    
    return new_sequence

# Function to convert sequence to one-hot encoded format
def one_hot_encode_sequence(sequence, seq_length):
    one_hot_encoded = np.zeros((seq_length, 4), dtype=int)
    base_to_index = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    for i, base in enumerate(sequence):
        if base in base_to_index:
            one_hot_encoded[i, base_to_index[base]] = 1
    return one_hot_encoded

# Predict accessibility function
def predict_accessibility(model, sequences):
    seq_array = np.array([one_hot_encode_sequence(seq, 1001) for seq in sequences])
    predictions = model.predict(seq_array)
    return np.array(predictions)

# Reverse complement function
def reverse_complement(sequence):
    return sequence[::-1].translate(str.maketrans('ATGC', 'TACG'))

# Main function
def main(model_path, motifA, num_sequences, output_csv):
    model = load_model_with_custom_objects(model_path)

    # Directly generate random sequences
    sequences = [generate_random_sequence(1001) for _ in range(num_sequences)]

    results = []

    for idx, sequence in enumerate(sequences):
        original_accessibility = predict_accessibility(model, [sequence])[0][0]

        motifA_position = len(sequence) // 2 - len(motifA) // 2
        seq_with_motifA = insert_motif(sequence, motifA, motifA_position)
        motifA_accessibility = predict_accessibility(model, [seq_with_motifA])[0][0]

        rev_comp_sequence = reverse_complement(sequence)
        rev_comp_with_motifA = reverse_complement(seq_with_motifA)
        rev_comp_accessibility = predict_accessibility(model, [rev_comp_sequence])[0][0]
        rev_comp_motifA_accessibility = predict_accessibility(model, [rev_comp_with_motifA])[0][0]

        # Generate random replacements by modifying only the central motif-length bases
        random_replacements = generate_random_replacement_sequences(sequence, 1000, len(motifA))

        # Predict accessibility for all random replacements at once
        replacement_accessibilities = predict_accessibility(model, random_replacements)

        # Count how many replacements have accessibility exceeding the motifA-inserted sequence
        num_replacements_exceeding_motifA = np.sum(replacement_accessibilities > motifA_accessibility)

        # Store results
        results.append({
            'sequence_index': idx,
            'original_or_reversecomplement': 'original',
            'null_accessibility': original_accessibility,
            'motifA_accessibility': motifA_accessibility,
            'num_replacements_exceeding_motifA': num_replacements_exceeding_motifA
        })
        results.append({
            'sequence_index': idx,
            'original_or_reversecomplement': 'reverse_complement',
            'null_accessibility': rev_comp_accessibility,
            'motifA_accessibility': rev_comp_motifA_accessibility,
            'num_replacements_exceeding_motifA': num_replacements_exceeding_motifA
        })

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze DNA sequences with motif insertion and accessibility prediction")
    parser.add_argument("--model_path", type=str, default="/camp/home/weie/lab_space_weie/training/new_models/new_models/D5_p1_NFIAn.h5", help="Path to the Keras model file")
    parser.add_argument("--motifA", type=str, required=True, help="Motif A (can include 'x' placeholders)")
    parser.add_argument("--num_sequences", type=int, default=1000, help="Number of valid sequences to generate")
    parser.add_argument("--output_csv", type=str, required=True, help="Filename to save the results as CSV")

    args = parser.parse_args()
    main(args.model_path, args.motifA, args.num_sequences, args.output_csv)