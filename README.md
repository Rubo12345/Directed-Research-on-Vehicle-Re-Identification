# Directed-Research-on-Vehicle-Re-Identification

Self Supervised Geometric Features Discovery vis Interpretable Attention

Branches:
1) Global Branch (GB): Encode robust global features codes from an input image.
2) Self-Supervised Learning Branch (SLB): SLB performs the auxiliary self-supervised representation learning
3) Geometrics Features Branch (GFB): By sharing its encoder with SLB, GFB discovers discriminative features from automatically discovered geometric locations without corresponding supervision.

Problem Setup:
Input -> Query Image
Output -> Ranking list of all gallery images according to similarities between query and gallery image. (Similarity score is obtained by cosine similarity)

Self supervised learning for highlighting geometric features:
Self-supervised learning is equivalent to optimizing deep network under the supervion of machine generated pseudo labels.
Image rotation degree prediction: rotating image by a random angle and training a classifier to predict it.
Vehicle ReID can be regarded as an instance level classification problem, i.e., all images contain the same species but many instance. Thus, salient object in each images has similar geometry properties, e.g., shape, outline, and skeleton.
A network to predict the rotation degree of a randomly rotated vehicle image encourages it to focus on these reliable and shared geometric properties, which can help to easily recognize the rotation of an object.

Steps: 1) Rotate an image Xi from Dataset by 0,90,180,270 degreees, to generate a new dataset Dsl = {Xi,r, Yr}. 
       2) Feed the image Xi,r into a shared encoder ResNet18 (Orange).
       3) To predict rotation class, high level representations need to be further condensed from ResNet18. To do this another subnetwork consisting of two basic ResNet           blocks are appended.
       4) High dimensional embedding vector is obtained: Fsl(Xi,r) = GAP[fse(fae(Xi,r;0ae);0se)]
       5) To generate more compact clusters in embedded space, the consine classifier (CC) is employed to assign the rotation class.
       6) The learnable parameters of CC is Wcc = [W1,...,Wj,....Wb], b = 4.
       7) The probabilities of assigning the input image into each class canbe represented as P(Xi,r) = [p1, p2, p3,..pb], 
          where each element is pj = softmax[cos(Fsl(xi,r),wj)]. 
       8) Cross Entropy Loss = [CE(P(Xi,r),Yr)]
