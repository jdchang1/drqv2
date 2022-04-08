import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

from pathlib import Path

import hydra
import numpy as np
import torch
from dm_env import specs

import dmc
import utils
from logger import Logger
from replay_buffer import ReplayBufferStorage, make_replay_loader
from video import TrainVideoRecorder, VideoRecorder

from tqdm import tqdm
import numpy as np
from PIL import Image
import os.path as osp
import lmdb
import pickle
from PIL import Image
torch.backends.cudnn.benchmark = True

#linear(1.0,0.1,500000)


def main(task, checkpoint, db_dirs, eps, file_path):
    chkpt = torch.load(checkpoint)
    agent = chkpt['agent'] # assuming gpu if not just change map_location
    global_step = chkpt['_global_step']
    frame_stack = 3
    action_repeat = 2
    env = dmc.make(task, frame_stack, action_repeat, 0)

    spec = env.action_spec()
    action_shape = spec.shape
    action_min = spec.minimum
    action_max = spec.maximum

    def deserialize(serialized_img):
        img = np.frombuffer(
            pickle.loads(serialized_img), dtype=np.uint8
        ).reshape(3, 84, 84)
        return img

    def get_log_prob(obs, action, eps_greedy):
        action_dist = agent.get_dist(obs, global_step)
        log_prob = action_dist.log_prob(action.cuda())
        prob = eps_greedy/action_shape[-1] + (1-eps_greedy)*torch.exp(log_prob.cpu())
        return torch.sum(torch.log(prob), dim=-1).detach().numpy()

    # get next actions with appropriate policy 
    for db, ep in zip(db_dirs, eps):
        print(f'DB: {db} | Eps: {ep}')
        obs_db = lmdb.open(osp.join(file_path, db, 'frames.lmdb'), max_readers=1, readonly=True, lock=False,
                            readahead=False, meminit=False)
        data = np.load(osp.join(file_path, db, 'data.npz'), allow_pickle=True)
        idx_mapping = data['idx_mapping'].item()
        actions = np.load(osp.join(file_path, db, 'next_actions_comp.npy'))
        log_probs = []
        for idx in tqdm(range(actions.shape[0])):
            obs_idxs = [idx_mapping[idx] + j for j in range(1, 4)]
            images = []
            with obs_db.begin(write=False) as txn:
                for obs_idx in obs_idxs:
                    serialized_im = txn.get(f'{obs_idx}'.encode('ascii'))
                    im = deserialize(serialized_im)
                    images.append(im) # move channel first NOTE: prob not necessary...
            obs = np.concatenate(images, axis=0)
            act = torch.from_numpy(actions[idx]).unsqueeze(0)
            log_prob = get_log_prob(obs, act, ep)
            log_probs.append(log_prob)
        print(np.concatenate(log_probs, axis=0).shape)
        with open(osp.join(file_path, db, 'log_probs.npy'), 'wb') as f:
            np.save(f, np.concatenate(log_probs, axis=0))


if __name__ == '__main__':
    #db_dirs = ['d_target', 'd_offline_1', 'd_offline_2', \
    #    'd_offline_3', 'd_offline_4', 'd_offline_5', 'd_offline_6']
    db_dirs = ['d_target', 'd_offline_2', 'd_offline_4', 'd_offline_6']
    #eps = [0.0, 0.2, 0.6, 1.0]
    eps = [0.0, 0.015, 0.05, 0.2]
    #eps = [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]

    root_dir = '/home/jdc396/offline_representation/drq_data'
    task_dirs = {
        'cartpole_swingup': 'cartpole-swingup',
        'cup_catch': 'cup-catch',
        'finger_turn_hard': 'finger-turn-hard',
        'humanoid_run': 'humanoid-run',
        'humanoid_stand': 'humanoid-stand',
        'quadruped_walk': 'quadruped-walk',
        'walker_walk': 'walker-walk',
        'cheetah_run': 'cheetah-run'
    }
    #tasks = ['cartpole_swingup', 'cup_catch', 'finger_turn_hard', 'humanoid_run', 'humanoid_stand', 'quadruped_walk', 'walker_walk']
    #tasks = ['cheetah_run', 'finger_turn_hard', 'quadruped_walk']
    tasks = ['humanoid_stand']

    checkpoints = {
        'cartpole_swingup': './exp_local/2021.12.19/015554_task=cartpole_swingup/snapshot.pt',
        'cup_catch': './exp_local/2021.12.19/051258_task=cup_catch/snapshot.pt',
        'finger_turn_hard': './exp_local/2021.12.18/133831_task=finger_turn_hard/snapshot.pt',
        'humanoid_run': './exp_local/2021.12.18/133530_task=humanoid_run/snapshot.pt',
        'humanoid_stand': './exp_local/2021.12.18/133539_task=humanoid_stand/snapshot.pt',
        'quadruped_walk': './exp_local/2021.12.18/133623_task=quadruped_walk/snapshot.pt',
        'walker_walk': './exp_local/2021.12.19/025259_task=walker_walk/snapshot.pt',
        'cheetah_run': 'exp_local/2021.12.04/111419_task=cheetah_run/snapshot.pt'
    }

    for task in tasks:
        print(f'Getting Log Probs for {task}')
        main(task, checkpoints[task], db_dirs, eps, osp.join(root_dir, task_dirs[task]))
    #for task in tasks:
    #    if task != 'cartpole_swingup':
    #        print(f'Getting Comparison Next Actions for {task}')
    #        main(task, checkpoints[task], osp.join(root_dir, task_dirs[task]), db_dirs, eps)




