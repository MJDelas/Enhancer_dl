import argparse
import pandas as pd

def contains_n(seq):
    """Check if the sequence contains 'N'."""
    return 'N' in seq

def main():
    invalid_rows = []
    parser = argparse.ArgumentParser()
    parser.add_argument('--fasta', required=True, help='FASTA file to filter')
    parser.add_argument('--csv', required=True, help='CSV file to filter')
    args = parser.parse_args()

    # Read and filter the fasta file
    with open(args.fasta) as f:
        lines = f.readlines()
    
    filtered_fasta_lines = []
    sequence_index = -1  # To keep track of the sequence index in the fasta file
    for i in range(0, len(lines), 2):
        header = lines[i]
        seq = lines[i + 1].strip()
        sequence_index += 1
        if contains_n(seq):
            invalid_rows.append(sequence_index)
        else:
            filtered_fasta_lines.append(header)
            filtered_fasta_lines.append(seq + '\n')
    
    # Write the filtered fasta sequences to a new file
    with open('filtered_' + args.fasta, 'w') as f:
        f.writelines(filtered_fasta_lines)
    
    # Filter the CSV file
    df = pd.read_csv(args.csv)
    df_filtered = df.drop(index=invalid_rows)
    
    # Write the filtered CSV to a new file
    df_filtered.to_csv('filtered_' + args.csv, index=False)
    
    print(f'Number of rows excluded: {len(invalid_rows)}')

if __name__ == "__main__":
    main()
