import numpy as np
import random
import torch
from PIL import Image
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict
import pickle

class CalvinDataset(Dataset):
    """Naive implementation of dataset to store
    calvin debug dataset, may be changed to WDS for the full dataset
    """

    def __init__(self, dataset_path, obs_horizon, pred_horizon, is_train=True) -> None:
        super().__init__()
        self.dataset_path = dataset_path
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon

        tag = "training" if is_train else "validation"
        self.file_prefix = f"{self.dataset_path}/{tag}"
        self.anns = np.load(
            f"{self.file_prefix}/lang_annotations/auto_lang_ann.npy", allow_pickle=True
        ).item()

    def __len__(self):
        return len(self.anns["info"]["indx"])

    def __getitem__(self, index):
        rgb_static = []
        depth_static = []
        rgb_gripper = []
        depth_gripper = []
        actions = []
        next_actions = []

        task = self.anns["language"]["task"][index]
        text = self.anns["language"]["ann"][index]
        st, ed = self.anns["info"]["indx"][index]
        # CJ: randomly sample a datapoint in the episode
        frame_idx = random.randint(st, ed - self.obs_horizon - self.pred_horizon + 1)
        for i in range(self.obs_horizon):
            frame = np.load(
                f"{self.file_prefix}/episode_{(frame_idx + i):07d}.npz"
            )  # , allow_pickle=True (lazy load)
            rgb_static.append(Image.fromarray(frame["rgb_static"]))
            depth_static.append(Image.fromarray(frame["depth_static"]))
            rgb_gripper.append(Image.fromarray(frame["rgb_gripper"]))
            depth_gripper.append(Image.fromarray(frame["depth_gripper"]))
            action = np.array(frame["rel_actions"])
            action[..., 6:] = (action[..., 6:] + 1) // 2
            actions.append(action)

        for i in range(self.pred_horizon):
            frame = np.load(
                f"{self.file_prefix}/episode_{(frame_idx + self.obs_horizon + i):07d}.npz"
            )
            action = np.array(frame["rel_actions"])
            action[..., 6:] = (action[..., 6:] + 1) // 2
            next_actions.append(action)

        return { 'rgb_static' : np.array(rgb_static), 
                'depth_static' : np.array(depth_static), 
                'rgb_gripper' : np.array(rgb_gripper), 
                'depth_gripper' : np.array(depth_gripper), 
                'text' : np.array(text), 
                'actions' : np.array(actions),
                'next_actions' : np.array(next_actions) }