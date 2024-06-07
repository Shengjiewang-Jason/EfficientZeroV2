#!/bin/bash
set -ex
#export OMP_NUM_THREADS=1
#export CUDA_DEVICE_ORDER='PCI_BUS_ID'
#export CUDA_VISIBLE_DEVICES=0,1
#export CUDA_VISIBLE_DEVICES=2,3
export CUDA_VISIBLE_DEVICES=6,7
#export CUDA_VISIBLE_DEVICES=6,7
export NCCL_IGNORE_DISABLED_P2P=1
export HYDRA_FULL_ERROR=1
export MASTER_PORT='12399'
export MUJOCO_GL=egl

# python ez/train.py exp_config=ez/config/exp/atari.yaml #> profile.txt
python ez/train.py exp_config=ez/config/exp/dmc_image.yaml #> profile.txt
# python ez/train.py exp_config=ez/config/exp/dmc_state.yaml #> profile.txt