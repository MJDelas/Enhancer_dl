- The environments:
- All the prediction and deepshap with jupyter_tf (in environment.yml), and everything TF-modisco related with meme_env (meme.yml)<br/><br/>

- The models:
- I have the zip files of all the models (the name should be with day e.g. D5 and domain e.g. pMN)
- To use the model files, I have a script maybe_predict.py that saves the prediction in a csv file<br/><br/>
- The training example is in Training/example_usage.txt

- Deep explainer:
- deep_explainer/run_deepexplain_savenpz_corrected.py saves the contribution scores of the sequences
- Can be used to visualisation <br/><br/>

- tf-modisco:
- Requires npz file of the one hot encoded fasta
- Command line modisco motifs in script modisco_and_meme/run_modisco_int_new1.sh, then compared to jaspar database using modisco_and_meme/submit_meme2.sh
- To help produce understandable reports modisco_and_meme/Interpret_modisco.ipynb is used to extract motif names and find matching names from MA codes

- Interaction analysis:
- In the folder interaction_analysis
- Bash script to run in interaction_analysis/interactiveD11.txt
- (Implementation requires understanding the orientation of motif, impact of placement of motifs, distances between motifs)
