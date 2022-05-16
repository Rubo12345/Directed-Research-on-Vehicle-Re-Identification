from curses import def_shell_mode
from dataclasses import dataclass
from fileinput import filename
from multiprocessing import BoundedSemaphore
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os.path as osp
from PIL import Image
import pickle
from glob import glob
from itertools import islice
from Datasets import veri_train, Rotation

class Self_Supervised_Learning:
    def __init__(self):
        V = veri_train.VeRi()
        self.dataset_dir = '/home/rutu/WPI/Directed_Research/ReID_Datasets/VeRi'
        self.train_dir = osp.join(dataset_dir, 'image_train')
        self.train_list = None   # I think data gets shuffled using this  # 37778
        self.test_dir = osp.join(dataset_dir,'image_test')
        self.test_list = None
        self.Dsl_path = osp.join(dataset_dir,'Dsl')
        self.Dsl_test_path = osp.join(dataset_dir, 'Dsl_test')
        self.root_dir = osp.join(dataset_dir,'Dsl')