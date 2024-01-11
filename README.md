# 301RoseDataSet
 we collected Gram-stained bacteria images from lower respiratory tract specimens of patients with lung infections in Chinese PLA General Hospital obtained by M-ROSE from 2018 to 2022 and desensitized these images to produce 1705 images (4,912 × 3,684 pixels). 
## DataSet
https://doi.org/10.5281/zenodo.10474234
## DataSet specification
BigPic: Total 1705 pieces of raw big data.
<br>
DeepDataSet:Select some data from the original big picture data to verify the quality of the dataset.
<br>
640DataSet:Selected some large maps from the original large data, cut them into 640 sizes, manually marked the detection box, and segmented the data.
<br>
DetectionDataSet: Converts 640 data sets to detection data sets suitable for YoloV5 networks.
<br>
SplitDataSet: Transforms 640 datasets into split datasets suitable for Unet networks
<br>
* G-cocci(0_G) : 3371
* G+cocci(0_G+) : 1462
* G-bacilli(1_G) : 5799
* G+bacilli(1_G+) : 1192

## 检测
![image](https://github.com/Quanlab-Bioimage/301RoseDataSet/blob/main/images/Detection.png)


## 分割
![[](https://github.com/Quanlab-Bioimage/301RoseDataSet/blob/main/images/Split.png)https://github.com/Quanlab-Bioimage/301RoseDataSet/blob/main/images/Split.png]
