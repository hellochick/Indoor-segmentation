# Indoor-segmentation
## Introduction
  This is an implementation of DeepLab-ResNet in TensorFlow for Indoor-scene segmentation on the [ade20k](http://sceneparsing.csail.mit.edu/) dataset. Since this model is for `robot navigating`, we `re-label 150 classes into 27 classes` in order to easily classify obstacles and road.  

### Re-label list: 
```
1 (wall)      <- 9(window), 15(door), 33(fence), 43(pillar), 44(sign board), 145(bullertin board)
4 (floor)     <- 7(road), 14(ground, 30(field), 53(path), 55(runway)
5 (tree)      <- 18(plant)
8 (furniture) <- 8(bed), 11(cabinet), 14(sofa), 16(table), 19(curtain), 20(chair), 25(shelf), 34(desk) 
7 (stairs)    <- 54(stairs)
26(others)    <- class number larger than 26
```

  
## Install 
First get restore checkpoint from [Google Drive](https://drive.google.com/drive/folders/0B9CKOTmy0DyaQ2oxUHdtYUd2Mm8?usp=sharing) and put into `restore_weights` directory.

Run `inference.py` with `--img_path` and `--restore_from`
```
python inference --img_path=FILENAME --restore_from=CHECKPOINT_DIR
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
