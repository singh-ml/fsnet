# FsNet: Feature Selection Network on High-dimensional Biological Data

Feature Selection Network (FsNet) is a scalable concrete neural network architecture for Wide data. Wide data consists of high-dimensional and small number of samples.
Specifically, FsNet consists of a selector layer that uses a concrete random variable for discrete feature selection and a supervised deep neural network regularized with the reconstruction loss.
Because a large number of parameters in the selector and reconstruction layer can easily cause overfitting under a limited number of samples, we use two tiny networks to predict the large virtual weight matrices of the selector and reconstruction layers. 

For more details, see the accompanying paper: ["FsNet: Feature Selection Network on High-dimensional Biological Data"](https://arxiv.org/abs/2001.08322), *arXive*, and please use the citation below.

```
@article{singh2020fsnet,
      title={FsNet: Feature Selection Network on High-dimensional Biological Data}, 
      author={Dinesh Singh and Héctor Climente-González and Mathis Petrovich and Eiryo Kawakami and Makoto Yamada},
      year={2020},
      eprint={2001.08322},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
