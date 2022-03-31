import tensorflow as tf

# new_list=[145,56,89,56]
# print(type(new_list))
# con_lis = tf.convert_to_tensor(new_list)
# print("Convert list to tensor:",con_lis)

from PIL import Image
im = Image.open("/home/rutu/WPI/Directed_Research/ReID_Datasets/VeRi/image_train/0496_c017_00035340_0.jpg")
im = Image.open('/home/rutu/WPI/Directed_Research/ReID_Datasets/VeRi/image_train/0381_c004_00031195_0.jpg')
# print(im)
pix_val = list(im.getdata())
# print(pix_val)
# print(len(pix_val))  # length is 24336, but as it is RGB it should be 73008 
pix_val_flat = [x for sets in pix_val for x in sets]
# print(pix_val_flat)
# print(len(pix_val_flat))   # length is 73008, flattened (Also check the sequence of colors R G B)

import torch
Tensor = torch.Tensor(pix_val_flat)
Tensor = torch.reshape(Tensor, (1,3,156,156))
print(Tensor)