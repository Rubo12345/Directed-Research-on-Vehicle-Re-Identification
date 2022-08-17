from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import re
import os.path as osp
import sys

from matplotlib.pyplot import imshow # My change
sys.path.append('/home/rutu/WPI/Directed_Research/My_Approach_To_DR/') # My change
from  Datasets.utils.base import BaseImageDataset

class VeRi(BaseImageDataset):
    """
    VeRi
    Reference:
    Liu, X., Liu, W., Ma, H., Fu, H.: Large-scale vehicle re-identification in urban surveillance videos. In: IEEE   %
    International Conference on Multimedia and Expo. (2016) accepted.

    Dataset statistics:
    # identities: 776 vehicles(576 for training and 200 for testing)
    # images: 37778 (train) + 11579 (query)
    """
    dataset_dir = '/home/rutu/WPI/Directed_Research/ReID_Datasets/VeRi'   #Dataset path

    def __init__(self, root='datasets', verbose=True, **kwargs):
        super(VeRi, self).__init__(root)
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'image_train')
        # self.train_list = None
        self.train_list = osp.join(self.dataset_dir, 'name_train.txt')
        # self.query_dir = osp.join(self.dataset_dir, 'image_query')        #Purpose Comment
        # self.query_list = osp.join(self.dataset_dir, 'name_query.txt')    #Purpose Comment
        # self.gallery_dir = osp.join(self.dataset_dir, 'image_test')       #Purpose Comment
        # self.gallery_list = osp.join(self.dataset_dir, 'name_test.txt')   #Purpose Comment

        self.check_before_run()

        train = self.process_dir(self.train_dir, self.train_list, relabel=True)
        # query = self.process_dir(self.query_dir, self.query_list, relabel=False)         #Purpose Comment
        # gallery = self.process_dir(self.gallery_dir, self.gallery_list, relabel=False)   #Purpose Comment

        if verbose:
            print('=> VeRi loaded')
            # self.print_dataset_statistics(train, query, gallery) #Purpose Comment
            # self.print_dataset_statistics(train)

        self.train = train
        # self.query = query      #Purpose Comment
        # self.gallery = gallery  #Purpose Comment

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
       
    def check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError('"{}" is not available'.format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError('"{}" is not available'.format(self.train_dir))
     
    def process_dir(self, dir_path, list_path=None, relabel=False):
        if list_path is None:
            img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        else:
            name_list = open(list_path).readlines()
            img_paths = []
            for index in range(len(name_list)):
                img_paths.append(osp.join(dir_path, name_list[index].strip()))
        pattern = re.compile(r'([-\d]+)_c([-\d]+)')

        pid_container = set()
        for img_path in img_paths:
            if pattern.search(img_path) is None:   #My change
                return None                                  #My change
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue  # junk images are just ignored
            assert 0 <= pid <= 1501  # pid == 0 means background
            assert 1 <= camid <= 20
            camid -= 1  # index starts from 0
            if relabel:
                pid = pid2label[pid]
            
            dataset.append((img_path, pid, camid)) 
        return dataset

    def data_image_labels(dataset_dir, train_dir, train_list):
        train_data = VeRi.process_dir(train_dir,train_list, relabel=True)
        Train_Images = [];Train_Labels = [];Train_Cams = []
        for image in range(len(train_data)):
            Train_Images.append(train_data[image][0])
            Train_Labels.append(train_data[image][1])
            Train_Cams.append(train_data[image][2])
        return Train_Images, Train_Labels, Train_Cams

