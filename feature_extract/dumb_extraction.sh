#!/bin/bash

#SBATCH -n 1                   #The number of cores should match the '--threads' parameter of subjunc
#SBATCH -N 1                            #Run on 1 node
#SBATCH --mem=25000                   

#SBATCH -t 08:00:00 #Indicate duration using HH:MM:SS
#SBATCH -p general #Based on your duration               

#SBATCH -o ./error/ngram_tfidf_out
#SBATCH -e ./error/ngram_tfidf_error
#SBATCH --mail-type=ALL
#SBATCH --mail-user=alley@college.harvard.edu

# --------------
source new-modules.sh
cd /n/regal/scrb152/Students/sandias42/mlware/feature_extract
module load python/2.7.11-fasrc01
source activate mlware_big
python dumb_extraction.py --vectorizer=hash_4gram_tfidf