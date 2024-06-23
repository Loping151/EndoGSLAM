import glob
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from natsort import natsorted
import cv2

from .basedataset import GradSLAMDataset


class C3VDDataset(GradSLAMDataset):
    def __init__(
        self,        
        config_dict,
        basedir,
        sequence,
        stride: Optional[int] = None,
        start: Optional[int] = 0,
        end: Optional[int] = -1,
        desired_height: Optional[int] = 540,
        desired_width: Optional[int] = 675,
        load_embeddings: Optional[bool] = False,
        embedding_dir: Optional[str] = "embeddings",
        embedding_dim: Optional[int] = 512,
        train_or_test: Optional[str] = 'all',
        **kwargs,
    ):
        self.input_folder = os.path.join(basedir, sequence)
        self.pose_path = os.path.join(self.input_folder, "pose.txt")
        self.mode = train_or_test
        super().__init__(
            config_dict,
            stride=stride,
            start=start,
            end=end,
            desired_height=desired_height,
            desired_width=desired_width,
            load_embeddings=load_embeddings,
            embedding_dir=embedding_dir,
            embedding_dim=embedding_dim,
            **kwargs,
        )
        
    def train_test_split(self, stride):
        all_idx = set(range(self.end))
        train_idx = None
        eval_idx = set(range(self.start + 7, self.end, 8)) # we don't expect eval in early frames
        train_idx = all_idx - eval_idx
        eval_idx = sorted(list(eval_idx))
        train_idx = sorted(list(train_idx))
            
        if self.mode == 'test':
            self.color_paths = [self.color_paths[i] for i in eval_idx]
            self.depth_paths = [self.depth_paths[i] for i in eval_idx]
            if self.load_embeddings:
                self.embedding_paths = [self.embedding_paths[i] for i in eval_idx]
            self.poses = [self.poses[i] for i in eval_idx]
            self.retained_inds = torch.arange(self.num_imgs)[eval_idx]
        elif self.mode == 'train':
            self.color_paths = [self.color_paths[i] for i in train_idx]
            self.depth_paths = [self.depth_paths[i] for i in train_idx]
            if self.load_embeddings:
                self.embedding_paths = [self.embedding_paths[i] for i in train_idx]
            self.poses = [self.poses[i] for i in train_idx]
            self.retained_inds = torch.arange(self.num_imgs)[train_idx]
        
            self.color_paths = self.color_paths[self.start : len(self.color_paths) : stride]
            self.depth_paths = self.depth_paths[self.start : len(self.depth_paths) : stride]
            if self.load_embeddings:
                self.embedding_paths = self.embedding_paths[self.start : len(self.embedding_paths) : stride]
            self.poses = self.poses[self.start : len(self.poses) : stride]
            # Tensor of retained indices (indices of frames and poses that were retained)
            self.retained_inds = torch.arange(self.num_imgs)[self.start : len(self.retained_inds) : stride]
        else:
            super().train_test_split(stride)

    def get_filepaths(self):
        color_paths = natsorted(glob.glob(f"{self.input_folder}/color/*.png"))
        depth_paths = natsorted(glob.glob(f"{self.input_folder}/depth/*.tiff"))
        embedding_paths = None
        if self.load_embeddings:
            embedding_paths = natsorted(glob.glob(f"{self.input_folder}/{self.embedding_dir}/*.pt"))
        return color_paths, depth_paths, embedding_paths
    
    def load_poses(self):
        poses = []
        with open(self.pose_path, "r") as f:
            lines = f.readlines() # Skip the header
        for i in range(self.num_imgs):
            line = lines[i]
            pose = list(map(float, line.split(sep=',')))
            pose = torch.Tensor(pose).reshape(4, 4).float().transpose(0, 1)
            poses.append(pose)
        return poses

    def read_embedding_from_file(self, embedding_file_path):
        embedding = torch.load(embedding_file_path)
        return embedding.permute(0, 2, 3, 1)  # (1, H, W, embedding_dim)
