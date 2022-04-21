# Indoor-segmentation
## Introduction
  This is an implementation of TensorFlow-based (TF1) DeepLab-ResNet for Indoor-scene segmentation. The provided model is trained on the [ade20k](http://sceneparsing.csail.mit.edu/) dataset. The code is inherited from [tensorflow-deeplab-resnet](https://github.com/DrSleep/tensorflow-deeplab-resnet) by [Drsleep](https://drsleep.github.io/). Since this model is for `robot navigating`, we `re-label 150 classes into 27 classes` in order to easily classify obstacles and road.  

### Re-label list: 
```
1 (wall)      <- 9(window), 15(door), 33(fence), 43(pillar), 44(sign board), 145(bullertin board)
4 (floor)     <- 7(road), 14(ground, 30(field), 53(path), 55(runway)
5 (tree)      <- 18(plant)
8 (furniture) <- 8(bed), 11(cabinet), 14(sofa), 16(table), 19(curtain), 20(chair), 25(shelf), 34(desk) 
7 (stairs)    <- 54(stairs)
26(others)    <- class number larger than 26
```

  
## Quick Start 
### Install dependency 
The codes are test on `Python 3.7`. Please run the following script to install the packages.
```bash
pip install -r requirements.txt
```

### Download pretrained model
Run the following script to download the provided pretrained model from Google Drive.
```bash
./download_models.sh
```
Or directly get the pretrained model from [Google Drive](https://drive.google.com/file/d/1o7QrlNxH6BX6uYatlR06-A_cutWD9sNg/view?usp=sharing).

### Demo
Run the following sample command for inference
```
python inference.py --img_path input/IMG_0416_640x480.png --restore_from=pretrained_models/ResNet101/
```

## Result
### Video
[![Demo video](https://img.youtube.com/vi/4OqW3M-eqaQ/0.jpg)](https://youtu.be/4OqW3M-eqaQ)
### Image
Input image                |  Output image
:-------------------------:|:-------------------------:
![](https://github.com/hellochick/Indoor-segmentation/blob/master/input/IMG_0416_640x480.png)  |  ![](https://github.com/hellochick/Indoor-segmentation/blob/master/output/IMG_0416_640x480.png)
![](https://github.com/hellochick/Indoor-segmentation/blob/master/input/IMG_0417_640x480.png)  |  ![](https://github.com/hellochick/Indoor-segmentation/blob/master/output/IMG_0417_640x480.png)
![](https://github.com/hellochick/Indoor-segmentation/blob/master/input/IMG_0418_640x480.png)  |  ![](https://github.com/hellochick/Indoor-segmentation/blob/master/output/IMG_0418_640x480.png)
