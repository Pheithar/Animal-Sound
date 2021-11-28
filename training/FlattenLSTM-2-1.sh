#!/bin/sh
### General options
### –- specify queue --
#BSUB -q gpua100
### -- set the job Name --
#BSUB -J FlattenLSTM-2-1
### -- ask for number of cores (default: 1) --
#BSUB -n 1
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 10:00
# request 5GB of system-memory
#BSUB -R "rusage[mem=16GB]"
### -- set the email address --
#BSUB -u avalverdemahou@gmail.com
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o outs/gpu_%J.out
#BSUB -e outs/gpu_%J.err
# -- end of LSF options --



module load python3/3.9.6
module load numpy/1.21.1-python-3.9.6-openblas-0.3.17
module load pandas/1.3.1-python-3.9.6
module load cuda/11.3
python3 -m pip install --user torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html --no-warn-script-location
python3 -m pip install --user scikit-image tqdm tabulate h5py sklearn pyyaml seaborn --no-warn-script-location

echo Training
echo --------
python3 FlattenLSTM.py FlattenLSTM-2-1
echo --------
echo Done
