# Copyright (c) EVAR Lab, IIIS, Tsinghua University.
#
# This source code is licensed under the GNU License, Version 3.0
# found in the LICENSE file in the root directory of this source tree.

import os
import time
os.environ["RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE"] = "1"
import ray
import wandb
import hydra
import torch
import multiprocessing

import sys
sys.path.append(os.getcwd())

import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp

from pathlib import Path
from omegaconf import OmegaConf

from ez import agents
from ez.utils.format import set_seed, init_logger
from ez.worker import start_workers, join_workers
from ez.eval import eval


@hydra.main(config_path='./config', config_name='config', version_base='1.1')
def main(config):
    if config.exp_config is not None:
        exp_config = OmegaConf.load(config.exp_config)
        config = OmegaConf.merge(config, exp_config)

    if config.ray.single_process:
        config.train.self_play_update_interval = 1
        config.train.reanalyze_update_interval = 1
        config.actors.data_worker = 1
        config.actors.batch_worker = 1
        config.data.num_envs = 1

    if config.ddp.world_size > 1:
        mp.spawn(start_ddp_trainer, args=(config,), nprocs=config.ddp.world_size)
    else:
        start_ddp_trainer(0, config)


def start_ddp_trainer(rank, config):
    assert rank >= 0
    print(f'start {rank} train worker...')
    agent = agents.names[config.agent_name](config)         # update config
    manager = None
    num_gpus = torch.cuda.device_count()
    num_cpus = multiprocessing.cpu_count()
    ray.init(num_gpus=num_gpus, num_cpus=num_cpus, object_store_memory=150 * 1024 * 1024 * 1024 if config.env.image_based else 100 * 1024 * 1024 * 1024)
    set_seed(config.env.base_seed + rank >= 0)              # set seed
    # set log

    if rank == 0:
        # wandb logger
        if config.ddp.training_size == 1:
            wandb_name = config.env.game + '-' + config.wandb.tag
            print(f'wandb_name={wandb_name}')
            logger = wandb.init(
                name=wandb_name,
                project=config.wandb.project,
                # config=config,
            )
        else:
            logger = None
        # file logger
        log_path = os.path.join(config.save_path, 'logs')
        os.makedirs(log_path, exist_ok=True)
        init_logger(log_path)
    else:
        logger = None

    # train
    final_weights = train(rank, agent, manager, logger, config)

    # final evaluation
    if rank == 0:
        model = agent.build_model()
        model.set_weights(final_weights)
        save_path = Path(config.save_path) / 'recordings' / 'final'

        scores = eval(agent, model, config.train.eval_n_episode, save_path, config)
        print('final score: ', np.mean(scores))


def train(rank, agent, manager, logger, config):
    # launch for the main process
    if rank == 0:
        workers, server_lst = start_workers(agent, manager, config)
    else:
        workers, server_lst = None, None

    # train
    storage_server, replay_buffer_server, watchdog_server, batch_storage = server_lst

    if config.ddp.training_size == 1:
        final_weights, final_model = agent.train(rank, replay_buffer_server, storage_server, batch_storage, logger)
    else:
        from ez.agents.base import train_ddp
        time.sleep(1)
        train_workers = [
            train_ddp.remote(
                agent, rank * config.ddp.training_size + rank_i,
                replay_buffer_server, storage_server, batch_storage, logger
            ) for rank_i in range(config.ddp.training_size)
        ]
        time.sleep(1)
        final_weights, final_model = ray.get(train_workers)

    epi_scores = eval(agent, final_model, 10, Path(config.save_path) / 'evaluation' / 'final', config,
                           max_steps=27000, use_pb=False, verbose=config.eval.verbose)
    print(f'final_mean_score={epi_scores.mean():.3f}')

    # join process
    if rank == 0:
        print(f'[main process] master worker finished')
        time.sleep(1)
        join_workers(workers, server_lst)

    # return
    dist.destroy_process_group()
    return final_weights


if __name__ == '__main__':
    main()
