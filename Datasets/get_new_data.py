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
import sys
sys.path.append('/home/rutu/WPI/Directed_Research/Directed-Research-on-Vehicle-Re-Identification/Datasets/')
import veri_train
import xml.etree.ElementTree as ET

def directory_paths():
    V = veri_train.VeRi()
    dataset_dir = '/home/rutu/WPI/Directed_Research/ReID_Datasets/VeRi'
    train_dir = osp.join(dataset_dir, 'image_train')
    train_list = None   # I think data gets shuffled using this  # 37778
    test_dir = osp.join(dataset_dir,'image_test')
    test_list = None
    Dsl_path = osp.join(dataset_dir,'Dsl/')
    Dsl_test_path = osp.join(dataset_dir, 'Dsl_test/')
    root_dir = osp.join(dataset_dir,'Dsl')
    return V, dataset_dir, train_dir, train_list, test_dir, train_list, test_list, Dsl_path, Dsl_test_path, root_dir

V, dataset_dir, train_dir, train_list, test_dir, train_list, test_list, Dsl_path, Dsl_test_path, root_dir = directory_paths()

def data_image_labels(train_dir, train_list):
        train_data = V.process_dir(train_dir,train_list, relabel=True)
        Train_Images = [];Train_Labels = [];Train_Cams = []
        for image in range(len(train_data)):
            Train_Images.append(train_data[image][0])
            Train_Labels.append(train_data[image][1])
            Train_Cams.append(train_data[image][2])
        return Train_Images, Train_Labels, Train_Cams

def image_to_pixel(image):
    pix_val = list(image.getdata())
    pix_val_flat = [x for sets in pix_val for x in sets]
    return pix_val_flat

def input_to_4d_tensor(I):
    ''' Function converts the image into a tensor as well as size it'''
    preprocess = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_image = Image.open(I)
    Tensor = preprocess(input_image)
    Tensor = torch.reshape(Tensor,(1,3,Tensor.shape[1],Tensor.shape[2]))
    return Tensor

def Data_List_Train(Train_Images,Data_Size):  #for new train
    Dsl = []; Dsl_Label = []
    with open('/home/rutu/WPI/Directed_Research/ReID_Datasets/VeRi/train_label.xml','r') as f:
        root = ET.fromstring(f.read())
    
    classes = [1, 3, 4, 7, 8, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 28, 29, 31, 32, 33, 34, 36, 37, 39, 40, 41, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 62, 64, 67, 68, 69, 70, 71, 72, 73, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 87, 88, 91, 92, 93, 94, 95, 97, 98, 99, 100, 103, 107, 109, 111, 112, 114, 115, 116, 119, 120, 121, 123, 124, 125, 127, 128, 131, 133, 136, 137, 138, 139, 140, 141, 146, 147, 148, 149, 152, 153, 155, 156, 157, 158, 159, 160, 161, 163, 164, 165, 167, 168, 169, 170, 171, 175, 176, 178, 181, 184, 185, 186, 187, 189, 190, 191, 193, 194, 195, 198, 199, 200, 201, 202, 203, 204, 206, 208, 209, 210, 211, 212, 213, 214, 215, 217, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 232, 233, 234, 235, 236, 238, 239, 242, 243, 244, 245, 246, 248, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 274, 275, 276, 277, 278, 279, 280, 282, 283, 284, 285, 286, 287, 289, 290, 291, 292, 293, 297, 301, 302, 303, 304, 305, 307, 308, 309, 311, 312, 313, 314, 316, 317, 320, 321, 323, 324, 325, 328, 329, 330, 331, 333, 334, 335, 336, 338, 339, 340, 341, 342, 343, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 358, 359, 360, 362, 363, 364, 366, 367, 370, 372, 374, 375, 376, 377, 378, 379, 381, 382, 383, 384, 386, 387, 390, 393, 394, 395, 396, 397, 399, 400, 401, 403, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 419, 422, 423, 424, 425, 426, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 447, 448, 449, 451, 452, 453, 454, 455, 457, 458, 459, 460, 462, 463, 464, 465, 466, 468, 469, 470, 471, 472, 474, 475, 478, 479, 481, 483, 484, 487, 488, 490, 491, 492, 493, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506, 508, 509, 511, 512, 513, 514, 516, 517, 519, 520, 521, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532, 533, 534, 535, 536, 537, 538, 539, 540, 542, 544, 547, 548, 549, 550, 551, 552, 553, 555, 556, 557, 558, 559, 560, 561, 562, 563, 564, 565, 567, 568, 571, 572, 573, 574, 575, 577, 578, 579, 580, 583, 587, 589, 591, 592, 594, 595, 596, 599, 600, 601, 603, 604, 605, 607, 608, 611, 613, 616, 617, 618, 619, 620, 621, 626, 627, 628, 629, 632, 633, 635, 636, 637, 638, 639, 640, 641, 643, 644, 645, 647, 648, 649, 650, 651, 655, 656, 658, 661, 664, 665, 666, 667, 669, 670, 671, 673, 674, 675, 678, 679, 680, 681, 682, 683, 684, 686, 688, 689, 690, 691, 692, 693, 694, 695, 697, 700, 701, 702, 703, 704, 705, 706, 707, 708, 709, 710, 713, 714, 715, 716, 718, 719, 722, 723, 724, 725, 726, 728, 730, 731, 732, 733, 734, 735, 736, 737, 738, 739, 740, 741, 743, 744, 745, 746, 747, 748, 749, 750, 751, 752, 754, 755, 756, 757, 758, 759, 760, 762, 763, 764, 765, 766, 767, 769]
    class_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511, 512, 513, 514, 515, 516, 517, 518, 519, 520, 521, 522, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532, 533, 534, 535, 536, 537, 538, 539, 540, 541, 542, 543, 544, 545, 546, 547, 548, 549, 550, 551, 552, 553, 554, 555, 556, 557, 558, 559, 560, 561, 562, 563, 564, 565, 566, 567, 568, 569, 570, 571, 572, 573, 574]

    for i in range(Data_Size):
        _4d_tensor = input_to_4d_tensor(Train_Images[i])
        Dsl.append(_4d_tensor)
        Index = classes.index(int(root[0][i].attrib['vehicleID']))
        label = class_index[Index]
        Dsl_Label.append(label)
    Dsl_Label = torch.Tensor(Dsl_Label)

    return Dsl,Dsl_Label

def Data_List_Test(Test_Images,Data_Size):  #for new train
    Dsl_test = []; Dsl_Label_test = []
    with open('/home/rutu/WPI/Directed_Research/ReID_Datasets/VeRi/test_label.xml','r') as f:
        root = ET.fromstring(f.read())
    
    classes = [2, 5, 6, 9, 14, 27, 30, 35, 38, 42, 61, 63, 65, 66, 74, 86, 89, 90, 96, 101, 102, 104, 105, 106, 108, 110, 113, 117, 118, 122, 126, 129, 130, 132, 134, 135, 142, 143, 144, 145, 150, 151, 154, 162, 166, 172, 173, 174, 177, 179, 180, 182, 183, 188, 192, 196, 197, 205, 207, 216, 218, 219, 231, 237, 240, 241, 247, 249, 262, 273, 281, 288, 294, 295, 296, 298, 299, 300, 306, 310, 315, 318, 319, 322, 326, 327, 332, 337, 344, 357, 361, 365, 368, 369, 371, 373, 380, 385, 388, 391, 392, 398, 402, 404, 405, 416, 417, 418, 420, 421, 427, 446, 450, 456, 461, 467, 473, 476, 477, 480, 482, 485, 486, 489, 494, 507, 510, 515, 518, 522, 541, 543, 545, 546, 554, 566, 569, 570, 576, 581, 582, 584, 585, 586, 588, 590, 593, 597, 598, 602, 606, 609, 610, 612, 614, 615, 622, 623, 624, 625, 630, 631, 634, 642, 646, 652, 653, 654, 657, 659, 660, 662, 663, 668, 672, 676, 677, 685, 687, 696, 698, 699, 711, 712, 717, 720, 721, 727, 729, 742, 753, 761, 768, 770, 771, 772, 773, 774, 775, 776]
    class_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199]

    for i in range(Data_Size):
        _4d_tensor = input_to_4d_tensor(Test_Images[i])
        Dsl_test.append(_4d_tensor)
        Index = classes.index(int(root[0][i].attrib['vehicleID']))
        label = class_index[Index]
        Dsl_Label_test.append(label)
    Dsl_Label_test = torch.Tensor(Dsl_Label_test)

    return Dsl_test,Dsl_Label_test

def get_data(No_of_Train_Images, No_of_Test_Images):
    Train_Images, Train_Labels, Train_Cams = data_image_labels(train_dir, train_list)
    Dsl,Dsl_Label = Data_List_Train(Train_Images,No_of_Train_Images)
    Test_Images, Test_Labels, Test_Cams = data_image_labels(test_dir,test_list)
    Dsl_test,Dsl_Label_test = Data_List_Test(Test_Images,No_of_Test_Images)
    return Dsl, Dsl_Label, Dsl_test, Dsl_Label_test

Dsl, Dsl_Label, Dsl_test, Dsl_Label_test = get_data(560,560)  #4000,1120

def save_pkl(D,path):
    with open(path, 'wb') as f:
        pickle.dump(D, f)

def read_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def save_pkl_folder(Data, Data_Label, path):
    for i in range(len(Data)):
        D = {}
        D['image'] = Data[i]
        D['label'] = Data_Label[i]
        tmp = path + f'{i}.pkl'
        save_pkl(D,tmp)

save_pkl_folder(Dsl,Dsl_Label,Dsl_path)
save_pkl_folder(Dsl_test,Dsl_Label_test, Dsl_test_path)

class Veri(Dataset):
    """dataset."""
    def __init__(self, root_dir, transform=None):
        self.files = glob(f'{root_dir}*.pkl')
        self.transform = transform
        # self.class_names = ['0','1','2','3']  # 0 for '0', 1 for '90', 2 for '180', 3 for '270'
        # Have a look at the class names

    def __len__(self):
        return len(self.files)  # partial data, return = 10

    def __getitem__(self, idx):
        file = self.files[idx]
        D = read_pkl(file)
        return {
            'image': torch.tensor(D['image']),
            'label': torch.tensor(D['label'], dtype=torch.long),
            # 'class_names': self.class_names
            # Have a look at the class names
        }

def data_loader(path,batch_size):
    veri = Veri(path)
    loader = torch.utils.data.DataLoader(veri, batch_size, shuffle=True)
    return loader,veri 
