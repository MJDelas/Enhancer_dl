#!/bin/bash

# Array of input filenames
input_files=("intersected_columns.bed" "random_neg_rows.bed")
output_file="merged_samples_1001bp.bed"

# Empty the output file if it already exists
> "$output_file"

# Loop over each input file
for input_file in "${input_files[@]}"; do
    awk '
    {
        # Adjust coordinates
        start = $2 - 400;
        end = $3 + 400;

        # Ensure the start is not negative
        if (start < 0) {
            start = 0;
        }

        
        # Print duplicate line with + strandedness
        print $1"\t"start"\t"end"\t+";
    }
    ' "$input_file" >> "$output_file"
done
