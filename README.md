## Selective Hourglass Mapping for Universal Image Restoration Based on Diffusion Model <br><sub>Official PyTorch Implementation of DiffUIR. </sub>

[Project Page](https://isee-laboratory.github.io/DiffUIR/) | [Paper]() | [Personal HomePage](https://zhengdian1.github.io)

### Updates
[**2024.03.17**] The **whole training and testing codes** are released!!!
[**2024.03.16**] The **pretrained weights** of DiffUIR are released in [link1](https://drive.google.com/drive/folders/1vIFrSe8Bfy9neNSQjO51OKyEKNV83BLW?usp=drive_link)
[**2024.02.27**]  🎉🎉🎉 Our DiffUIR paper was accepted by CVPR 2024 🎉🎉🎉 <br>

## Introduction

The main challenge of universal image restoration tasks is handling different degradation images at once. In this work, we propose a selective hourglass mapping strategy based on conditional diffusion model to learn the shared information between different tasks. Specifically, we integrate a 
shared distribution term to the diffusion algorithm elegantly and naturally, achieving to map the different distributions to a shared one and could further guide the shared distribution to the task-specific clean image. By only modify
the mapping strategy, we outperform large-scale universal methods with at least five times less computational costing.


### Framework comparison
![image](Images/diffuir.png)

# How to use

## Environment
* Python 3.79
* Pytorch 1.12

## Install

### Create a virtual environment and activate it.

```
conda create -n diffuir python=3.7
conda activate diffuir
```
### Dependencies

```
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch -c nvidia
pip install opencv-python
pip install scikit-image
pip install tensorboard
pip install matplotlib 
pip install tqdm
```

## Data Preparation
Download [Scene Flow Datasets](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)

## Train 
We train the five image restoration tasks at once, you can change train.py-Line42 to change the task type. <br>
Note that the result of our paper can be reimplemented on RTX 4090, using 3090 or other gpus may cause performance drop.
```
python train.py
```

## Test and Calculate the Metric
Note that the dataset of SOTS can not calculate the metric online as the number of input and gt images is different. 
Please use eval/SOTS.m. 
```
python test.py
```

For Under-Camera real-world dataset, as the resolution is high, we split the image into several pathes and merge them after model.
```
python test_udc.py
```

## Visualize
Here you can test our model in your personal image. Note that if you want to test low-light image, please use the code src/visualization-Line1279-1281
```
python visual.py
```

## Qualitative results on four restoration tasks

![image](Images/four.png)

## Analysis of the shared information

The distribution before and after our SDT, SDT map the different degradation images to a shared distribution.
![image](Images/tsne.png)

The attention of the feature map, our method could focus on the degradation type (rain and fog), validating that we learn the useful shared information.
![image](Images/attention.png)

# Citation

If you find this project helpful in your research, welcome to cite the paper.

```
@inproceedings{zheng2024selective,
  title={Selective Hourglass Mapping for Universal Image Restoration Based on Diffusion Model},
  author={Zheng, Dian and Wu, Xiao-Ming and Yang, Shuzhou and Zhang, Jian and Hu, Jian-Fang and Zheng, Wei-shi},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2024}
}

```

# Acknowledgements

Thanks to Jiawei Liu for opening source of his excellent works RDDM. Our work is inspired by these works and part of codes are migrated from [RDDM](https://github.com/nachifur/RDDM).

# Acknowledgements

Please contact Dian Zheng if there is any question (1423606603@qq.com or zhengd35@mail2.sysu.edu.cn).