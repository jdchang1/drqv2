import numpy as np
import os.path as osp
from PIL import Image
import cv2

def min_max_traj(root_dir, target_dir, task='cheetah-run'):
    datafile = osp.join(root_dir, target_dir, 'data.npz')
    gamma = 0.99

    data = np.load(datafile, allow_pickle=True)

    # Grab 100 trajectories
    data_len = np.where(data['dones'])[0]
    starts = np.concatenate([[0], np.where(data['dones'])[0][:-1] + 1], axis=0) # get the starts
    dones = data['dones']
    rewards = data['rewards']

    scores, ep_reward = [], 0
    for d, r in zip(dones, rewards):
        ep_reward += r
        if d:
            scores.append(ep_reward)
            ep_reward = 0
    min_score, max_score, mean_score = np.min(scores), np.max(scores), np.mean(scores)
    min_idx, max_idx = np.argmin(scores), np.argmax(scores)
    min_idx_range = list(range(starts[min_idx], starts[min_idx+1]))
    max_idx_range = list(range(starts[max_idx], starts[max_idx+1]))

    idx_mapping = data['idx_mapping'].item()

    def get_traj(idx_range):
        traj = []
        for idx in idx_range:
            im = np.asarray(Image.open(osp.join(root_dir, target_dir, 'big_frames', f'{idx}.png')))
            traj.append(im)
        return np.stack(traj)

    def create_mp4(frames, video_name, fps=30):
        codec = cv2.VideoWriter_fourcc(*'mp4v')
        size = frames[0].shape[:-1] # assuming (H, W, C)
        grayscale = frames[0].shape[-1] == 1
        video = cv2.VideoWriter(osp.join(root_dir, target_dir, f'{video_name}.mp4'), codec, fps, size)
        for frame in frames:
            if grayscale:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video.write(frame) # Note assumes non grayscale is BGR.....should convert
        video.release()

    min_traj = get_traj(min_idx_range)
    max_traj = get_traj(max_idx_range)
    meta_data = {'min_score': min_score, 'max_score': max_score, 'mean_score': mean_score}

    with open(osp.join(root_dir, target_dir, 'trajs.npz'), 'wb') as f:
        np.savez(f, min_traj=min_traj, max_traj=max_traj, meta=meta_data)
    print(min_score, max_score, mean_score)
    print(min_traj.shape)
    print(max_traj.shape)

    print('Creating Videos')
    create_mp4(min_traj, video_name=f'{task}_{int(min_score)}')
    create_mp4(max_traj, video_name=f'{task}_{int(max_score)}')


if __name__ == '__main__':
    data_dir = '/home/jdc396/offline_representation/drq_data/cheetah-run'
    #target_dir = 'd_offline'
    target_dirs = ['d_offline_686', 'd_offline_491', 'd_offline_367', 'd_offline_62', 'd_offline_7', 'd_offline', 'd_target']
    for target_dir in target_dirs:
        min_max_traj(data_dir, target_dir)
