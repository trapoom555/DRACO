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

    def __init__(self, dataset_path, is_train=True) -> None:
        super().__init__()
        self.dataset_path = dataset_path

        tag = "training" if is_train else "validation"
        self.file_prefix = f"{self.dataset_path}/{tag}"
        self.anns = np.load(
            f"{self.file_prefix}/lang_annotations/auto_lang_ann.npy", allow_pickle=True
        ).item()

    def __len__(self):
        return len(self.anns["info"]["indx"])

    def __getitem__(self, index):
        task = self.anns["language"]["task"][index]
        text = self.anns["language"]["ann"][index]
        st, ed = self.anns["info"]["indx"][index]
        # CJ: randomly sample a datapoint in the episode
        frame = random.randint(st, ed)
        frame = np.load(
            f"{self.file_prefix}/episode_{frame:07d}.npz"
        )  # , allow_pickle=True (lazy load)
        rgb_static = Image.fromarray(frame["rgb_static"])
        depth_static = Image.fromarray(frame["depth_static"])
        rgb_gripper = Image.fromarray(frame["rgb_gripper"])
        depth_gripper = Image.fromarray(frame["depth_gripper"])
        actions = np.array(frame["rel_actions"])
        
        actions[..., 6:] = (actions[..., 6:] + 1) // 2
        return rgb_static, depth_static, rgb_gripper, depth_gripper, text, actions