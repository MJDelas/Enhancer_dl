import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import random

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

# Generate random sequences
def generate_random_sequences(num_sequences, seq_length):
    bases = np.array(['A', 'T', 'G', 'C'])
    prob = np.array([0.2588, 0.2588, 0.2412, 0.2412])
    sequences = [''.join(np.random.choice(bases, seq_length, p=prob)) for _ in range(num_sequences)]
    return sequences

def insert_motif(sequence, motif, position):
    if 'x' in motif:
        # Handle motif with placeholders
        parts = motif.split('x')
        prefix = ''.join(parts[:-1])
        suffix = parts[-1]
        
        # Number of x placeholders
        num_x = len(parts) - 1
        
        # Extract the segment of the sequence that will replace the x's
        replacement_seq = sequence[position + len(prefix):position + len(prefix)+num_x]

        # Construct the new sequence with the motif inserted
        new_sequence = sequence[:position] + prefix + replacement_seq + suffix + sequence[position+len(motif):]
    else:
        # Handle motif without placeholders
        new_sequence = sequence[:position] + motif + sequence[position+len(motif):]
    # Ensure the length of the final sequence is correct
    if len(new_sequence) != len(sequence):
        raise ValueError(f"Final sequence length is {len(new_sequence)}, but expected {len(sequence)}")

    return new_sequence

def one_hot_encode_sequence(sequence, seq_length):
    # Ensure that the sequence length matches the expected length
    if len(sequence) != seq_length:
        raise ValueError(f"Sequence length {len(sequence)} does not match expected length {seq_length}.")
    
    one_hot_encoded = np.zeros((seq_length, 4), dtype=np.uint8)
    base_to_index = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    for i, base in enumerate(sequence):
        if base in base_to_index:
            one_hot_encoded[i, base_to_index[base]] = 1
        else:
            raise ValueError(f"Unknown base '{base}' encountered in sequence.")
    
    return one_hot_encoded

def predict_accessibility(model, sequences):
    seq_array = np.array([one_hot_encode_sequence(seq, 1001) for seq in sequences], dtype=np.uint8)
    predictions = model.predict(seq_array, batch_size=32)
    return predictions

def reverse_complement(sequence):
    return sequence[::-1].translate(str.maketrans('ATGC', 'TACG'))

def generate_random_replacement_for_motifs(sequence, motifA_pos, motifB_pos, motifA_len, motifB_len, num_replacements):
    replacements = []
    seq_length = len(sequence)
    for _ in range(num_replacements):
        # Generate random sequences for the first and second motifs
        random_motif_first = ''.join(np.random.choice(['A', 'T', 'G', 'C'], motifA_len))
        random_motif_second = ''.join(np.random.choice(['A', 'T', 'G', 'C'], motifB_len))

        # Replace the original motifs with the random ones
        new_sequence = (sequence[:motifA_pos] + random_motif_first + 
                        sequence[motifA_pos + motifA_len: motifB_pos] + 
                        random_motif_second + sequence[motifB_pos + motifB_len:])
        
        # Ensure that the length is still 1001 bp
        if len(new_sequence) != seq_length:
            raise ValueError(f"Generated sequence length {len(new_sequence)} does not match expected length {seq_length}.")
        
        replacements.append(new_sequence)

    return replacements

def main(model_path, motifA, motifB, num_sequences, output_csv):
    model = load_model_with_custom_objects(model_path)

    sequences = generate_random_sequences(num_sequences=num_sequences, seq_length=1001)
    results = []

    for idx, sequence in enumerate(sequences):
        # Predict accessibility for the original sequence
        original_accessibility = predict_accessibility(model, [sequence])[0][0]
        # Insert motif A in the center
        motifA_position = len(sequence) // 2 - len(motifA) // 2
        seq_with_motifA = insert_motif(sequence, motifA, motifA_position)
        motifA_accessibility = predict_accessibility(model, [seq_with_motifA])[0][0]

        # Insert motif B 25 bp downstream from the center
        motifB_position = motifA_position + 25
        seq_with_motifB = insert_motif(sequence, motifB, motifB_position)
        motifB_accessibility = predict_accessibility(model, [seq_with_motifB])[0][0]

        # Insert both motif A and motif B
        seq_with_motifsAB = insert_motif(seq_with_motifA, motifB, motifB_position)
        motifAB_accessibility = predict_accessibility(model, [seq_with_motifsAB])[0][0]

        # Predict accessibility for reverse complements
        rev_comp_sequence = reverse_complement(sequence)
        rev_comp_with_motifsAB = reverse_complement(seq_with_motifsAB)
        rev_comp_accessibility = predict_accessibility(model, [rev_comp_sequence])[0][0]
        rev_comp_motifAB_accessibility = predict_accessibility(model, [rev_comp_with_motifsAB])[0][0]

        # Generate 1000 random replacements for motif A and B positions
        random_replacements = generate_random_replacement_for_motifs(
            sequence, motifA_position, motifB_position, len(motifA), len(motifB), 1000
        )
        replacement_accessibilities = predict_accessibility(model, random_replacements)
        num_exceeding_motifAB = np.sum(replacement_accessibilities >= motifAB_accessibility)

        # Store results
        results.append({
            'sequence_index': idx,
            'original_or_reversecomplement': 'original',
            'null_accessibility': original_accessibility,
            'motifA_accessibility': motifA_accessibility,
            'motifB_accessibility': motifB_accessibility,
            'motifAB_accessibility': motifAB_accessibility,
            'insert_position_A_start': motifA_position,
            'insert_position_B_start': motifB_position,
            'num_replacements_exceeding_motifAB': num_exceeding_motifAB
        })
        results.append({
            'sequence_index': idx,
            'original_or_reversecomplement': 'reverse_complement',
            'null_accessibility': rev_comp_accessibility,
            'motifA_accessibility': predict_accessibility(model, [reverse_complement(seq_with_motifA)])[0][0],
            'motifB_accessibility': predict_accessibility(model, [reverse_complement(seq_with_motifB)])[0][0],
            'motifAB_accessibility': rev_comp_motifAB_accessibility,
            'insert_position_A_start': motifA_position,
            'insert_position_B_start': motifB_position,
            'num_replacements_exceeding_motifAB': num_exceeding_motifAB
        })

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze DNA sequences with motif insertion and accessibility prediction")
    parser.add_argument("--model_path", type=str, default="/camp/home/weie/lab_space_weie/training/new_models/new_models/D5_p1_NFIAn.h5", help="Path to the Keras model file")
    parser.add_argument("--motifA", type=str, default="AA", help="Motif A (can include 'x' placeholders)")
    parser.add_argument("--motifB", type=str, default="TT", help="Motif B (can include 'x' placeholders)")
    parser.add_argument("--num_sequences", type=int, default=1000, help="Number of valid sequences to generate")
    parser.add_argument("--output_csv", type=str, default='toy.csv', help="Filename to save the results as CSV")

    args = parser.parse_args()
    main(args.model_path, args.motifA, args.motifB, args.num_sequences, args.output_csv)