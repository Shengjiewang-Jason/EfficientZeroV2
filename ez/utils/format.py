# Copyright (c) EVAR Lab, IIIS, Tsinghua University.
#
# This source code is licensed under the GNU License, Version 3.0
# found in the LICENSE file in the root directory of this source tree.

import copy
import os
import cv2
import time
import torch
import random
import logging
import numpy as np

import subprocess as sp

from ray.util.queue import Queue

class RayQueue(object):
    def __init__(self, threshold=15, size=20):
        self.threshold = threshold
        self.queue = Queue(maxsize=size)

    def push(self, batch):
        if self.queue.qsize() <= self.threshold:
            self.queue.put(batch)

    def pop(self):
        if self.queue.qsize() > 0:
            return self.queue.get()
        else:
            return None

    def get_len(self):
        return self.queue.qsize()


class PreQueue(object):
    def __init__(self, threshold=15, size=20):
        self.threshold = threshold
        self.queue = Queue(maxsize=size)

    def push(self, batch):
        if self.queue.qsize() <= self.threshold:
            self.queue.put(batch)

    def pop(self):
        if self.queue.qsize() > 0:
            return self.queue.get()
        else:
            return None

    def get_len(self):
        return self.queue.qsize()



class LinearSchedule(object):
    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        """Linear interpolation between initial_p and final_p over
        schedule_timesteps. After this many timesteps pass final_p is
        returned.
        Parameters
        ----------
        schedule_timesteps: int
            Number of timesteps for which to linearly anneal initial_p
            to final_p
        initial_p: float
            initial output value
        final_p: float
            final output value
        """
        self.schedule_timesteps = schedule_timesteps
        self.final_p = final_p
        self.initial_p = initial_p

    def value(self, t):
        """See Schedule.value"""
        fraction = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)


def transform_one2(x):
    return torch.sign(x) * (torch.sqrt(torch.abs(x) + 1.0) - 1) + 0.001 * x

def transform_one(x):
    return np.sign(x) * (np.sqrt(np.abs(x) + 1.0) - 1) + 0.001 * x

def atanh(x):
    return 0.5 * (torch.log(1 + x + 1e-6) - torch.log(1 - x + 1e-6))

def symlog(x):
    return torch.sign(x) * torch.log(torch.abs(x) + 1)

def symexp(x):
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)

class DiscreteSupport(object):
    def __init__(self, config=None):
        if config:
            # assert min < max
            self.env = config.env.env
            if self.env in ['DMC', 'Gym']:
                assert config.model.reward_support.bins == config.model.value_support.bins
                self.size = config.model.reward_support.bins
            else:
                assert config.model.reward_support.range[0] == config.model.value_support.range[0]
                assert config.model.reward_support.range[1] == config.model.value_support.range[1]
                assert config.model.reward_support.scale == config.model.value_support.scale
                self.min = config.model.reward_support.range[0]
                self.max = config.model.reward_support.range[1]
                self.scale = config.model.reward_support.scale
                self.range = np.arange(self.min, self.max + self.scale, self.scale)
                self.size = len(self.range)

    @staticmethod
    def scalar_to_vector(x, **kwargs):
        """ Reference from MuZerp: Appendix F => Network Architecture
        & Appendix A : Proposition A.2 in https://arxiv.org/pdf/1805.11593.pdf (Page-11)
        """
        env = kwargs['env']
        x_min = kwargs['range'][0]
        x_max = kwargs['range'][1]

        epsilon = 0.001

        if env in ['DMC', 'Gym']:
            x_min = transform_one(x_min)
            x_max = transform_one(x_max)
            bins = kwargs['bins']
            scale = (x_max - x_min) / (bins - 1)
            x_range = np.arange(x_min, x_max + scale, scale)
            sign = torch.ones(x.shape).float().to(x.device)
            sign[x < 0] = -1.0
            x = sign * (torch.sqrt(torch.abs(x) + 1) - 1) + epsilon * x
            x = x / scale

            x.clamp_(x_min / scale, x_max / scale - 1e-5)
            x = x - x_min / scale
            x_low_idx = x.floor()
            x_high_idx = x.ceil()
            p_high = x - x_low_idx
            p_low = 1 - p_high

            target = torch.zeros(tuple(x.shape) + (bins,), dtype=p_high.dtype).to(x.device)
            target.scatter_(len(x.shape), x_high_idx.long().unsqueeze(-1), p_high.unsqueeze(-1))
            target.scatter_(len(x.shape), x_low_idx.long().unsqueeze(-1), p_low.unsqueeze(-1))
        else:
            scale = kwargs['scale']
            x_range = np.arange(x_min, x_max + scale, scale)
            x_size = len(x_range)

            # transform
            # assert scale == 1
            sign = torch.ones(x.shape).float().to(x.device)
            sign[x < 0] = -1.0
            x = sign * (torch.sqrt(torch.abs(x / scale) + 1) - 1 + epsilon * x / scale)

            # to vector
            x.clamp_(x_min, x_max)
            x_low = x.floor()
            x_high = x.ceil()
            p_high = x - x_low
            p_low = 1 - p_high

            target = torch.zeros(x.shape[0], x.shape[1], x_size).to(x.device)
            x_high_idx, x_low_idx = x_high - x_min / scale, x_low - x_min / scale
            target.scatter_(2, x_high_idx.long().unsqueeze(-1), p_high.unsqueeze(-1))
            target.scatter_(2, x_low_idx.long().unsqueeze(-1), p_low.unsqueeze(-1))
        return target

    @staticmethod
    def vector_to_scalar(logits, **kwargs):
        """ Reference from MuZerp: Appendix F => Network Architecture
        & Appendix A : Proposition A.2 in https://arxiv.org/pdf/1805.11593.pdf (Page-11)
        """
        x_min = kwargs['range'][0]
        x_max = kwargs['range'][1]
        env = kwargs['env']
        epsilon = 0.001

        if env in ['DMC', 'Gym']:
            x_min = transform_one(x_min)
            x_max = transform_one(x_max)
            bins = kwargs['bins']
            scale = (x_max - x_min) / (bins - 1)
            x_range = np.arange(x_min, x_max + scale, scale)

            # Decode to a scalar
            value_probs = torch.softmax(logits, dim=-1)  # training & test
            # value_probs = logits  # debug
            value_support = torch.ones(value_probs.shape)
            value_support[:, :] = torch.from_numpy(np.array([x for x in x_range]))
            value_support = value_support.to(device=value_probs.device)
            value = (value_support * value_probs).sum(-1, keepdim=True) / scale

            sign = torch.ones(value.shape).float().to(value.device)
            sign[value < 0] = -1.0
            output = (((torch.sqrt(1 + 4 * epsilon * (torch.abs(value) * scale + 1 + epsilon)) - 1) / (
                    2 * epsilon)) ** 2 - 1)
            output = sign * output
        else:
            scale = kwargs['scale']
            x_range = np.arange(x_min, x_max + scale, scale)
            value_probs = torch.softmax(logits, dim=-1)  # training & test
            # value_probs = logits  # debug
            value_support = torch.ones(value_probs.shape)
            value_support[:, :] = torch.from_numpy(np.array([x for x in x_range]))
            value_support = value_support.to(device=value_probs.device)
            value = (value_support * value_probs).sum(-1, keepdim=True) / scale

            sign = torch.ones(value.shape).float().to(value.device)
            sign[value < 0] = -1.0
            output = (((torch.sqrt(1 + 4 * epsilon * (torch.abs(value) + 1 + epsilon)) - 1) / (2 * epsilon)) ** 2 - 1)
            output = sign * output * scale

            nan_part = torch.isnan(output)
            output[nan_part] = 0.
            output[torch.abs(output) < epsilon] = 0.
        return output


def arr_to_str(arr):
    """
    To reduce memory usage, we choose to store the jpeg strings of image instead of the numpy array in the buffer.
    This function encodes the observation numpy arr to the jpeg strings.
    :param arr:
    :return:
    """
    img_str = cv2.imencode('.jpg', arr)[1].tobytes()

    return img_str


def str_to_arr(s, gray_scale=False):
    """
    To reduce memory usage, we choose to store the jpeg strings of image instead of the numpy array in the buffer.
    This function decodes the observation numpy arr from the jpeg strings.
    :param s: string of inputs
    :param gray_scale: bool, True -> the inputs observation is gray instead of RGB.
    :return:
    """
    nparr = np.frombuffer(s, np.uint8)
    if gray_scale:
        arr = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        arr = np.expand_dims(arr, -1)
    else:
        arr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    return arr


def formalize_obs_lst(obs_lst, image_based, already_prepare=False):
    # if not already_prepare:
    # obs_lst = prepare_obs_lst(obs_lst, image_based)
    obs_lst = np.asarray(obs_lst)
    if image_based:
        obs_lst = torch.from_numpy(obs_lst).cuda().float() / 255.
        obs_lst = torch.moveaxis(obs_lst, -1, 2)
        shape = obs_lst.shape
        obs_lst = obs_lst.reshape((shape[0], -1, shape[-2], shape[-1]))
    else:
        obs_lst = torch.from_numpy(obs_lst).cuda().float()
        shape = obs_lst.shape
        obs_lst = obs_lst.reshape((shape[0], -1))
    return obs_lst


def prepare_obs_lst(obs_lst, image_based):
    """Prepare the observations to satisfy the input fomat of torch
    [B, S, W, H, C] -> [B, S x C, W, H]
    batch, stack num, width, height, channel
    """
    if image_based:
        # B, S, W, H, C -> B, S x C, W, H
        obs_lst = np.asarray(obs_lst)
        obs_lst = np.moveaxis(obs_lst, -1, 2)

        shape = obs_lst.shape
        obs_lst = obs_lst.reshape((shape[0], -1, shape[-2], shape[-1]))
    else:
        # B, S, H
        obs_lst = np.asarray(obs_lst)
        shape = obs_lst.shape
        obs_lst = obs_lst.reshape((shape[0], -1))

    return obs_lst


def normalize_state(tensor, first_dim=1):
    # normalize the tensor (states)
    if first_dim < 0:
        first_dim = len(tensor.shape) + first_dim
    flat_tensor = tensor.view(*tensor.shape[:first_dim], -1)
    max = torch.max(flat_tensor, first_dim, keepdim=True).values
    min = torch.min(flat_tensor, first_dim, keepdim=True).values
    flat_tensor = (flat_tensor - min) / (max - min)

    return flat_tensor.view(*tensor.shape)


def softmax(logits):
    logits = np.asarray(logits)

    logits -= logits.max()
    logits = np.exp(logits)
    logits = logits / logits.sum()

    return logits


def pad_and_mask(trajectories, pad_value=0, is_action=False):
    """
    Pads the trajectories to the same length and creates an attention mask.
    """
    # Get the maximum length of trajectories in the batch
    max_len = max([len(t) for t in trajectories])

    # Initialize masks with zeros
    masks = torch.ones(len(trajectories), max_len).cuda().bool()

    # Pad trajectories and create masks
    padded_trajectories = []
    for i, traj in enumerate(trajectories):
        if is_action:
            padded_traj = torch.nn.functional.pad(traj, (0, max_len - len(traj)), value=pad_value)
        else:
            padded_traj = torch.nn.functional.pad(traj, (0, 0, 0, 0, 0, 0, 0, max_len - len(traj)), value=pad_value)
        masks[i, :len(traj)] = False
        padded_trajectories.append(padded_traj)

    try:
        padded_trajectories = torch.stack(padded_trajectories)
    except:
        import ipdb
        ipdb.set_trace()
        print('false')
    return padded_trajectories, masks


def init_logger(base_path):
    formatter = logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s][%(filename)s>%(funcName)s] ==> %(message)s')
    for mode in ['Train', 'Eval']:
        file_path = os.path.join(base_path, mode + '.log')
        logger = logging.getLogger(mode)
        handler = logging.FileHandler(file_path, mode='a')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)


def get_ddp_model_weights(ddp_model):
    """
    Get weights of a DDP model
    """
    return {'.'.join(k.split('.')[2:]): v.cpu() for k, v in ddp_model.state_dict().items()}


def allocate_gpu(rank, gpu_lst, worker_name):
    # set the gpu it resides on according to remaining memory
    time.sleep(3)
    available_memory_list = get_gpu_memory()
    for i in range(len(available_memory_list)):
        if i not in gpu_lst:
            available_memory_list[i] = -1
    available_memory_list[0] -= 4000  # avoid using gpu 0, which is left for training
    available_memory_list[1] -= 6000  # avoid using gpu 1, which is left for training
    max_index = available_memory_list.index(max(available_memory_list))
    if available_memory_list[max_index] < 2000:
        print(f"[{worker_name} worker GPU]******************* Warning: Low video ram (max remaining "
              f"{available_memory_list[max_index]}) *******************")
    torch.cuda.set_device(max_index)
    print(f"[{worker_name} worker GPU] {worker_name} worker GPU {rank} at process {os.getpid()}"
          f" will use GPU {max_index}. Remaining memory before allocation {available_memory_list}")


def get_gpu_memory():
    """
    Returns available gpu memory for each available gpu
    https://stackoverflow.com/questions/59567226/how-to-programmatically-determine-available-gpu-memory-with-tensorflow
    """

    # internal tool function
    def _output_to_list(x):
        return x.decode('ascii').split('\n')[:-1]

    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = _output_to_list(sp.check_output(command.split()))[1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values


def set_seed(seed):
    # set seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def profile(func):
    from line_profiler import LineProfiler
    def wrapper(*args, **kwargs):
        lp = LineProfiler()
        lp_wrapper = lp(func)
        result = lp_wrapper(*args, **kwargs)
        lp.print_stats()

        return result
    return wrapper


if __name__=='__main__':
    support = DiscreteSupport()
    # dict = {
    #     'range': [-2, 2],
    #     'scale': 0.01,
    #     'env': 'DMC',
    #     'bins': 51
    # }
    dict = {
        'range': [-299, 299],
        'scale': 0.01,
        'env': 'DMC',
        'bins': 51
    }
    # dict = {
    #     'range': [-300, 300],
    #     'scale': 1,
    #     'env': 'Atari',
    #     # 'bins': 51
    # }
    value = np.ones((2, 1)) * 15
    value = torch.from_numpy(value).float().cuda()
    print(f'input={value}')
    vec = support.scalar_to_vector(value, **dict).squeeze()
    vec2 = support.scalar_to_vector2(value, **dict).squeeze()
    print(f'input={value}')
    print(f'support={vec}, support2={vec2}')
    val = support.vector_to_scalar(vec, **dict)
    val2 = support.vector_to_scalar2(vec, **dict)
    print(f'support={vec}, support2={vec2}')
    print(f'input={value}')
    print(f'val={val}')

