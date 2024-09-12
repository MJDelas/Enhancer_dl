
## this should work? I changed some stuff from my original script


import numpy as np
from Bio import SeqIO
import tensorflow as tf
import os
import argparse
from tensorflow.keras.models import load_model

def parse_args():
    parser = argparse.ArgumentParser(description='Predict using trained model.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model.')
    parser.add_argument('--fasta', type=str, required=True, help='Path to test sequences FASTA file.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the output predictions.')
    return parser.parse_args()

def main():
    args = parse_args()

    # Extract base name from model path
    base_name = os.path.basename(args.model_path).replace('.h5', '')

    print(f"Running predictions for model: {base_name}")
    print(f"FASTA file: {args.fasta}")

    # Load the model
    model = load_model(args.model_path, custom_objects={'pearson_correlation': pearson_correlation})

    # Load sequences from FASTA and predict
    sequences = load_fasta_sequences(args.fasta)
    predictions = model.predict(sequences)
    
    # Save the predictions
    save_predictions(args.output_dir, f'{base_name}', predictions)
    print(f"Predictions saved in {args.output_dir}")

def load_fasta_sequences(fasta_path):
    sequences = list(SeqIO.parse(fasta_path, "fasta"))
    
    max_length = 1001  # Adjust based on your sequences' lengths
    encoded_sequences = np.array([one_hot_encode_sequence(str(seq.seq), max_length) for seq in sequences], dtype=np.float32)
    
    return encoded_sequences

def one_hot_encode_sequence(sequence, max_length):
    mapping = {'A': [1,0,0,0], 'C': [0,1,0,0], 'G': [0,0,1,0], 'T': [0,0,0,1], 'a': [1,0,0,0], 'c': [0,1,0,0], 'g': [0,0,1,0], 't': [0,0,0,1]}
    encoded = [mapping.get(base, [0, 0, 0, 0]) for base in sequence]
    padding = [[0, 0, 0, 0]] * (max_length - len(encoded))
    return np.array(encoded + padding, dtype=np.float32)

def save_predictions(output_dir, label_type, predictions):
    output_file = os.path.join(output_dir, f'{label_type}_predictions.csv')
    df = pd.DataFrame({'Prediction': predictions.flatten()})
    df.to_csv(output_file, index=False)

if __name__ == "__main__":
    main()