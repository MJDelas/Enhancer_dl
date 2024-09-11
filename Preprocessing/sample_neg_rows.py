import pandas as pd
import argparse
import os

def sample_df(file, nrows, output_dir):
    nrows=int(nrows)
    df=pd.read_csv(file, delimiter="\t") #tab delimited file (bed file)
    df=df.sample(n=nrows, random_state=42)
    output_file = os.path.join(f'{output_dir}/random_neg_rows.bed')
    df.to_csv(output_file, index=False,  sep='\t')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sample random windows (negative samples)')
    parser.add_argument('--file', help='Input bed file', required=True)
    parser.add_argument('--nrows', help='number of rows', required=True)
    parser.add_argument('--output_dir', help='Output file path', required=True)
    args = parser.parse_args()
    sample_df(args.file, args.nrows, args.output_dir) 
