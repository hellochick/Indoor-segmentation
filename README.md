# Indoor-segmentation
Indoor segmentation based on deeplab model, implemented on tensorflow

## Install 
First get restore checkpoint from [Google Drive](https://drive.google.com/drive/folders/0B9CKOTmy0DyaQ2oxUHdtYUd2Mm8?usp=sharing) and put into `restore_weights` directory.

Run `inference.py` with `--img_path` and `--restore-from`
```
python inference --img_path=FILENAME --restore-from=CHECKPOINT_DIR
```
## Result
### Video
![Demo video](https://www.youtube.com/watch?v=4OqW3M-eqaQ&feature=youtu.be)
### Image
Input image                |  Output image
:-------------------------:|:-------------------------:
![](https://github.com/hellochick/Indoor-segmentation/blob/master/input/IMG_0416_640x480.png)  |  ![](https://github.com/hellochick/Indoor-segmentation/blob/master/output/IMG_0416_640x480.png)
![](https://github.com/hellochick/Indoor-segmentation/blob/master/input/IMG_0417_640x480.png)  |  ![](https://github.com/hellochick/Indoor-segmentation/blob/master/output/IMG_0417_640x480.png)
![](https://github.com/hellochick/Indoor-segmentation/blob/master/input/IMG_0418_640x480.png)  |  ![](https://github.com/hellochick/Indoor-segmentation/blob/master/output/IMG_0418_640x480.png)
