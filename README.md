# Clinical bacteria DataSet
**This dataset comprises Gram-stained bacteria images obtained using M-ROSE from lower respiratory tract specimens of patients with lung infection from 2018 to 2022 at the Chinese PLA General Hospital. 1705 images (4,912 × 3,684 pixels) were desensitized and constituted RawImageDataSet. A total of 4,833 cocci and 6,991 bacilli were manually labeled and differentiated into negative and positive categories. 
Additionally, we have also employed a cascade detection and semantic segmentation network for bacteria identification that achieved over 85% accuracy. DeepDataSet was constructed to verify the dataset quality, which includes 640DataSet, DetectionDataSet, and SegmentationDataSet applied to train deep learning networks directly. The data and benchmark algorithms we provide may contribute to the study of automated bacterial identification in clinical specimens.**
## DataSet
https://doi.org/10.5281/zenodo.10526360
## DataSet specification
* RawImageDataSet: 1705 original images obtained from clinical specimens using the micro-optical system. 
* DeepDataSet: partial image selected randomly from RawImageDataSet, used to verify the quality of the dataset.
  * 640DataSet: partially extracted from corresponding raw images in a random manner and cropped into 640×640 pixels, then labeled them manually with detection box and segmentation boxes.
  * DetectionDataSet: converted from 640DataSet to detection DataSet applied for YOLOv5 networks.
  * SegmentationDataSet: converted from 640DataSet to Segmentation DataSet applied for Unet networks.
  <br>
Accumulated annotation a total of 3371 Gram-negative cocci (G-cocci), 1462 Gram-positive cocci (G+cocci), 5799 Gram-negative bacilli (G-bacilli), and 1192 Gram-positive bacilli (G+bacilli ).

## Detection
**The effect of target detection network recognition is as follows: 0 represents negative cocci, 1 represents positive cocci, 2 represents negative bacilli and 3 represents positive bacilli.**
<br>
![image](https://github.com/Quanlab-Bioimage/301RoseDataSet/blob/main/images/Detection.png)


## Segmentation
**The effect of the segmentation network is as follows, with red representing positive bacteria and green representing negative bacteria.**
<br>
![image](https://github.com/Quanlab-Bioimage/301RoseDataSet/blob/main/images/Split.png)
