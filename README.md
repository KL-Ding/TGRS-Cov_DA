# Cov-DA: A Stepwise Domain Adaptive Segmentation Network with Covariate shift Alleviation for Remote Sensing Imagery,TGRS,2022

[Jiaojiao Li](https://scholar.google.com/citations?user=Ccu3-acAAAAJ&amp;hl=zh-CN&amp;oi=sra ), [Shunyao Zi](https://github.com/KL-Ding/ ), [Rui Song](https://scholar.google.com/citations?user=_SKooBYAAAAJ&amp;hl=zh-CN), Yunsong Li, [Yinlin Hu](https://scholar.google.com/citations?hl=zh-CN&amp;user=dhmdaoQAAAAJ  ), [Qian Du](https://scholar.google.com/citations?user=0OdKQoQAAAAJ&amp;hl=zh-CN).

------

Code for the paper: 

[A Stepwise Domain Adaptive Segmentation Network with Covariate shift Alleviation for Remote Sensing Imagery](https://ieeexplore.ieee.org/document/9716091).

![](https://github.com/KL-Ding/Cov-DA/tree/kaile/Image/Cov-DA.png)

Fig. 1. Overview of Cov-DA for RSI. This network has the following three innovative parts: CMUM, MFJM, and PPAM. CMUM finds the closest k-pair color transformation matrix Mi{i = 1, 2, ..., k} and the corresponding weight αi in the database according to the histogram of each input image and calculates the unique transformation matrix. MJEM completes the second stage of intra-class DA by calculating the gray level co-occurrence matrix of an image edge and entropy information of an image and evaluating the complexity of each image in the target domain. PPAM extracts multi-scale features through image pyramids and inputs them into the coordinate attention module one by one to extract feature information from different angles to improve the contextual information perception of the network.



## Pre-requsites

- Python 3.7
- Pytorch >= 0.4.1
- CUDA 9.0 or higher

## Training and Test Process

1. Please prepare the training and test data as operated in the paper. The datasets are ISPRS, Aerial, Mars_seg dataset.

2. The model definition code is placed in the folder **.\UDA\ model**.

3. You can run eval.py to validate multiple models saved through the domain adaptation training method, or you can test individual models in your own projects.

## References

If you find this code helpful, please kindly cite:

[1] J. Li, S. Zi, R. Song, Y. Li, Y. Hu and Q. Du, "A Stepwise Domain Adaptive Segmentation Network With Covariate Shift Alleviation for Remote Sensing Imagery," in IEEE Transactions on Geoscience and Remote Sensing, vol. 60, pp. 1-15, 2022, Art no. 5618515, doi: 10.1109/TGRS.2022.3152587.

## Citation Details

BibTeX entry:

```
@ARTICLE{9716091,  author={Li, Jiaojiao and Zi, Shunyao and Song, Rui and Li, Yunsong and Hu, Yinlin and Du, Qian},  
journal={IEEE Transactions on Geoscience and Remote Sensing},   
title={A Stepwise Domain Adaptive Segmentation Network With Covariate Shift Alleviation for Remote Sensing Imagery},   
year={2022},  
volume={60},  
number={},  
pages={1-15},  
doi={10.1109/TGRS.2022.3152587}}
```

## License

Copyright (C) 2022 Shunyao Zi

You should have received a copy of the GNU General Public License along with this program.
