#!/bin/bash
#SBATCH --job-name=coverage_calc
#SBATCH --array=0-13               # 13 bigwig files for wildtype
#SBATCH --time=24:00:00           
#SBATCH --cpus-per-task=1         
#SBATCH --mem=16G
#SBATCH --partition=ncpu                 



BIGWIG_DIR="/nemo/lab/briscoej/home/users/delasj/hpc_camp/Glial_ATAC/results/bwa/mergedReplicate/bigwig"
BED_FILE="/camp/home/weie/lab_space_weie/notlogged_coverage/intersected_columns.bed" #replace with bed file you desire
OUTPUT_DIR="~/lab_space_weie/trial/pos"


BIGWIG_FILES=($BIGWIG_DIR/WT*.bigWig)

# Get the BigWig file corresponding to the current array task ID
BIGWIG_FILE=${BIGWIG_FILES[$SLURM_ARRAY_TASK_ID]}

#conda stuff
ml Anaconda3
source /camp/apps/eb/software/Anaconda/conda.env.sh
conda activate accessibility_dl
which python

# Activate the specific conda environment
echo "Activating conda environment 'accessibility_dl':"
if conda activate accessibility_dl; then
    echo "Environment activated successfully."
else
    echo "Failed to activate environment!" >&2
    exit 1
fi

python exact_coverage_intersect_bed.py --input_bigwig "$BIGWIG_FILE" --input_bed "$BED_FILE" --output_dir "$OUTPUT_DIR"