# FsNet: Feature Selection Network on High-dimensional Biological Data

Feature Selection Network (FsNet) is a scalable concrete neural network architecture for Wide data. Wide data consists of high-dimensional and small number of samples.
Specifically, FsNet consists of a selector layer that uses a concrete random variable for discrete feature selection and a supervised deep neural network regularized with the reconstruction loss.
Because a large number of parameters in the selector and reconstruction layer can easily cause overfitting under a limited number of samples, we use two tiny networks to predict the large virtual weight matrices of the selector and reconstruction layers. 
