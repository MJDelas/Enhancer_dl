import argparse
import os
import pyBigWig
import numpy as np
import pandas as pd


def extract_depth_normalized_coverage(bw_file, bed_file, output_dir, window_size=1001, center_size=201):
    bw = pyBigWig.open(bw_file)
    bed = pd.read_csv(bed_file, delimiter='\t',header=None,names=["chr","start","end"])
    print(bed.head(n=5))
    chrom=bed["chr"]
    start=bed["start"]
    end=bed["end"]
    coverage = np.zeros(len(chrom))
    bw_name = os.path.basename(bw_file).replace('.mLb.clN.bigWig', '')
    print(chrom[:5])
    # Iterate over chromosomes
    center_start = ((start + end - center_size) // 2).astype(int)
    center_end = center_start + center_size
    print(center_start[:5],center_end[:5])
    # for each row in the bed file, find the big window and the center to extract pybw stats from
    for i in range(len(start)):
        stats = bw.stats(chrom[i], center_start[i], center_end[i], exact=True)
        if stats[0] is None or stats[0] == 0:
            coverage[i] = 1e-10
        else:
            coverage[i] = stats[0]

    bw.close()
        # Create DataFrame from results
        # Create DataFrame from results
    df = pd.DataFrame({
        'Avg_Coverage': np.log(coverage)
    })
    # Save DataFrame to file
    output_file = os.path.join(output_dir, f'{bw_name}_consensus_peaks_coverage_table.csv')
    df.to_csv(output_file, index=False)
    print(f"Saved coverage table for {bw_name} to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate log average depth-normalized ATAC coverage over windows.')
    parser.add_argument('--input_bigwig', help='Input bigWig file(s)', required=True)
    parser.add_argument('--input_bed', help='Input bed file', required=True)
    parser.add_argument('--output_dir', help='Output file path', required=True)
    parser.add_argument('--window_size', type=int, default=1001, help='Window size (default: 1001)')
    parser.add_argument('--center_size', type=int, default=201, help='Window size of the centre we are extracting the log accessibility from (default: 201)')
    args = parser.parse_args()
    extract_depth_normalized_coverage(args.input_bigwig, args.input_bed, args.output_dir, args.window_size, args.center_size)


