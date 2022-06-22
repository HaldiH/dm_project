#!/bin/sh

#SBATCH --partition=shared-cpu
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=128

# load Anaconda, this will provide Jupyter as well.
module load Anaconda3/2021.05

export XDG_RUNTIME_DIR=""

# specify here the directory containing your notebooks
JUPYTER_NOTEBOOK_DIR=./

# sepcify here the port to listen to
PORT=1234

# Choose one of the solutions below:

# if you want to access jupyternotebook through proxy socks or x2go, it should listen to the node's ip.
IP=$SLURMD_NODENAME

# if you want to access jupyternotebook through an ssh tunnel, it should listen to the localhost (safer)
#IP=localhost

srun jupyter notebook --no-browser --port=$PORT --ip=$IP --notebook-dir=$JUPYTER_NOTEBOOK_DIR
