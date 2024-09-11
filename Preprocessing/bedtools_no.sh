#!/bin/bash
#SBATCH --job-name=bedtools_notoverlapped
#SBATCH --ntasks=1 
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --partition=ncpu
#SBATCH --mail-type=END,FAIL

ml BEDTools/2.31.0-GCC-12.3.0    
bedtools intersect -a central_columns1.bed -b /nemo/lab/briscoej/home/users/delasj/hpc_camp/Glial_ATAC/results/bwa/mergedLibrary/macs/broadPeak/consensus/consensus_peaks.mLb.clN.bed -v > not_overlapped_columns.bed 