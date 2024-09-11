#!/bin/bash

# Directory paths
input_dir="/camp/home/weie/lab_space_weie/training/TF-modisco/new_pos_only"
npz_dir="/camp/home/weie/lab_space_weie/training/TF-modisco/pos_fasta_npz"
output_dir="/camp/home/weie/lab_space_weie/training/TF-modisco/new_pos_only/modisco_output"

# Iterate over each contrib_scores_*.npz file
for hyp_file in ${input_dir}/hyp_contrib_score_*.npz; do
    # Extract the base name without directory
    hyp_file_name=$(basename ${hyp_file})
    
    # Extract the relevant part of the file name (e.g., WT_D11_p1_NFIAp_pos_test)
    file_identifier=$(echo ${hyp_file_name} | sed -E 's/^hyp_contrib_score_([^\.]+)\.npz$/\1/')

    # Define the corresponding interval_onehot file
    interval_onehot_file="${npz_dir}/${file_identifier}_pos_test.fa.npz"
    
    # Define the output file name
    modisco_output_file="${output_dir}/modisco_results_${file_identifier}.h5"

    echo "Running TF-MoDISco for file identifier: ${file_identifier}"

    # Run TF-MoDISco motifs
    modisco motifs -s ${interval_onehot_file} -a ${hyp_file} -n 2000 -o ${modisco_output_file}
done
