#!/bin/bash

# Directory paths
modisco_results_dir="/camp/home/weie/lab_space_weie/training/TF-modisco/new_pos_only/modisco_output"
jaspar_file="/camp/home/weie/lab_space_weie/training/TF-modisco/JASPAR/JASPAR2022_CORE_vertebrates_non-redundant_pfms_meme.txt"

# Iterate over each modisco_results_*.h5 file
for result_file in ${modisco_results_dir}/modisco_results_WT*.h5; do
    # Extract the base name without directory
    result_file_name=$(basename ${result_file})
    
    # Extract the relevant part of the file name (e.g., D9_pM_NFIAp_Early_1)
    file_identifier=$(echo ${result_file_name} | sed -e 's/^modisco_results_//' -e 's/.h5$//')
    
    # Define the output directory for the report
    report_output_dir="${modisco_results_dir}/report/report_${file_identifier}"
    
    # Define the job name
    job_name="report_${file_identifier}"
    script_name="job_${job_name}.sh"
    
    echo "Creating and submitting job: ${job_name}"
    
    # Create the SLURM job script
    cat <<EOT > ${script_name}
#!/bin/bash
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --job-name=${job_name}
#SBATCH --time=12:00:00
#SBATCH --partition=ncpu
#SBATCH --output=${modisco_results_dir}/output_report_${file_identifier}.txt
#SBATCH --error=${modisco_results_dir}/error_report_${file_identifier}.txt

# Load necessary modules and activate conda environment
ml Anaconda3
source /camp/apps/eb/software/Anaconda/conda.env.sh
conda activate meme_env

# Run TF-MoDISco report
modisco report -i ${result_file} -o ${report_output_dir} -m ${jaspar_file}
EOT

    # Submit the job script
    sbatch ${script_name}
done