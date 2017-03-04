#!/bin/bash
#
#SBATCH -p general                # partition (queue)
#SBATCH -N 1                      # number of nodes
#SBATCH -n 1                      # number of cores
#SBATCH --mem 80000              # memory pool for all cores
#SBATCH -t 0-8:00                 # time (D-HH:MM)
#SBATCH -o ./err/slurm.%N.%j.out        # STDOUT
#SBATCH -e ./err/slurm.%N.%j.err        # STDERR
#SBATCH --mail-type=END,FAIL      # notifications for job done & fail
#SBATCH --mail-user=khimulya@college.harvard.edu # send-to address

module load python/2.7.6-fasrc01
source activate HAMILTON

python ./neural_circuits_2gram_nocutoff.py
