1) Get the Dataset - Done (VeRi)

2) Filter out the Dataset - Done (Train, Test, Query)

3) Rotate the images from the Dataset to generate new dataset

4) Feed the images into ResNet18 

5) To predict rotation class, high level representations need to be further condensed from ResNet18.

6) To do the above step another subnetwork consisting of two basic ResNet Blocks are appended.

7) High Dimensional Embedding vector is obtained: Fsl(Xi,r) = GAP[fse(fae(Xi,r;0ae);0se)]

8) To generate more compact clusters in embedded space, the cosine classifier (CC) is employed to assign the rotation class.

9) The learnable parameters of cc is Wcc = [W1,...,Wj,....Wb], b = 4. 

10) The probabilities of assigning the input image into each class can be represented as P(Xi,r) = [p1, p2, p3,..pb], where each element is pj = softmax[cos(Fsl(xi,r),wj)]. 

11) Cross Entropy Loss = [CE(P(Xi,r),Yr)]

