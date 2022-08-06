
import numpy as np
import torch
from torch import nn
from torch.nn import Module, Conv2d, Parameter, Softmax
import torchvision.transforms as transforms
from PIL import Image
x = torch.randn(28,512,28,28)

def nms_pytorch(P : torch.tensor ,thresh_iou : float):

    """
    Apply non-maximum suppression to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        boxes: (tensor) The location preds for the image 
            along with the class predscores, Shape: [num_boxes,5].
        thresh_iou: (float) The overlap thresh for suppressing unnecessary boxes.
    Returns:
        A list of filtered boxes, Shape: [ , 5]
    """

    # we extract coordinates for every 
    # prediction box present in P
    x1 = P[:, 0]
    y1 = P[:, 1]
    x2 = P[:, 2]
    y2 = P[:, 3]

    # we extract the confidence scores as well
    scores = P[:, 4]

    # calculate area of every block in P
    areas = (x2 - x1) * (y2 - y1)
    
    # sort the prediction boxes in P
    # according to their confidence scores
    order = scores.argsort()

    # initialise an empty list for 
    # filtered prediction boxes
    keep = []
    

    while len(order) > 0:
        
        # extract the index of the 
        # prediction with highest score
        # we call this prediction S
        idx = order[-1]

        # push S in filtered predictions list
        keep.append(P[idx])

        # remove S from P
        order = order[:-1]

        # sanity check
        if len(order) == 0:
            break
        
        # select coordinates of BBoxes according to 
        # the indices in order
        xx1 = torch.index_select(x1,dim = 0, index = order)
        xx2 = torch.index_select(x2,dim = 0, index = order)
        yy1 = torch.index_select(y1,dim = 0, index = order)
        yy2 = torch.index_select(y2,dim = 0, index = order)

        # find the coordinates of the intersection boxes
        xx1 = torch.max(xx1, x1[idx])
        yy1 = torch.max(yy1, y1[idx])
        xx2 = torch.min(xx2, x2[idx])
        yy2 = torch.min(yy2, y2[idx])

        # find height and width of the intersection boxes
        w = xx2 - xx1
        h = yy2 - yy1
        
        # take max with 0.0 to avoid negative w and h
        # due to non-overlapping boxes
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)

        # find the intersection area
        inter = w*h

        # find the areas of BBoxes according the indices in order
        rem_areas = torch.index_select(areas, dim = 0, index = order) 

        # find the union of every prediction T in P
        # with the prediction S
        # Note that areas[idx] represents area of S
        union = (rem_areas - inter) + areas[idx]
        
        # find the IoU of every prediction in P with S
        IoU = inter / union

        # keep the boxes with IoU less than thresh_iou
        mask = IoU < thresh_iou
        order = order[mask]
    
    return keep

a = torch.randn((28,512,28,28),dtype = torch.float32)
identity = a

pd = (2,2,2,2)
a = torch.nn.functional.pad(a, pd, mode='constant', value=0)
m = nn.Softmax(dim = -1)
for i in range(28):
    for c in range(512):
        for h in range(28):
            for w in range(28):
                input = a[i,c,h:h+5,w:w+5]
                output = m(input)
                a[i,c,h:h+5,w:w+5] = output
transform = transforms.CenterCrop((28,28))
M = transform(a) # We got the M

a = identity

Lnuv = torch.tensor(-2)
for i in range(28):
    for h in range(28):
        for w in range(28):
            m = a[i,0:,h,w]
            _,max_ = torch.max(m,0)
            Lnuv = a[i,max_,h,w]
            for c in range(512):
                Lcuv = a[i,c,h,w]
                G = Lcuv/Lnuv
                a[i,c,h,w] = G
G = a   # We got the G

Q = torch.multiply(M,G)
print(Q.shape)

Q_Dash = torch.max(Q[0:,0:,0:,0:],1)
print(Q_Dash.values.shape)

'''a = torch.tensor(([[[[1,1,1,1],
				    [2,2,2,2],
                    [3,3,3,3],
                    [4,4,4,4]],
				   [[5,5,5,5],
				    [6,6,6,6],
                    [7,7,7,7],
                    [8,8,8,8]]],
                  [[[9,9,9,9],
				    [6,6,6,6],
                    [4,4,4,4],
                    [1,1,1,1]],
				   [[7,7,7,7],
				    [8,8,8,8],
                    [3,3,3,3],
                    [5,5,5,5]]]]),dtype = torch.float32)

print(a[0,0,3,3])
'''