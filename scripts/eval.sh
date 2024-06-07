#!/bin/bash

set -ex
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=0,2
export MASTER_PORT='12399'
export MUJOCO_GL=egl

#python ez/eval.py exp_config=ez/config/exp/atari.yaml
python ez/eval.py exp_config=ez/config/exp/dmc_image.yaml
#python ez/eval.py exp_config=ez/config/exp/dmc_state.yaml