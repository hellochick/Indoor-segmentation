from __future__ import print_function

import argparse
import os
import sys
import time
import scipy.io as sio
from PIL import Image

import tensorflow as tf
import numpy as np

from model import DeepLabResNetModel

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

NUM_CLASSES = 27
SAVE_DIR = './output/'
RESTORE_PATH = './restore_weights/'
matfn = 'color150.mat'

def get_arguments():
    parser = argparse.ArgumentParser(description="Indoor segmentation parser.")
    parser.add_argument("--img_path", type=str, default='',
                        help="Path to the RGB image file.")
    parser.add_argument("--restore_from", type=str, default=RESTORE_PATH,
                        help="checkpoint location")

    return parser.parse_args()

def read_labelcolours(matfn):
    mat = sio.loadmat(matfn)
    color_table = mat['colors']
    shape = color_table.shape
    color_list = [tuple(color_table[i]) for i in range(shape[0])]

    return color_list

def decode_labels(mask, num_images=1, num_classes=150):
    label_colours = read_labelcolours(matfn)

    n, h, w, c = mask.shape
    assert(n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' % (n, num_images)
    outputs = np.zeros((num_images, h, w, 3), dtype=np.uint8)
    for i in range(num_images):
      img = Image.new('RGB', (len(mask[i, 0]), len(mask[i])))
      pixels = img.load()
      for j_, j in enumerate(mask[i, :, :, 0]):
          for k_, k in enumerate(j):
              if k < num_classes:
                  pixels[k_,j_] = label_colours[k]
      outputs[i] = np.array(img)
    return outputs

def load(saver, sess, ckpt_path):
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))

def main():
    args = get_arguments()
    filename = args.img_path.split('/')[-1]
    file_type = filename.split('.')[-1]

    if os.path.isfile(args.img_path):
        print('successful load img: {0}'.format(args.img_path))
    else:
        print('not found file: {0}'.format(args.img_path))
        sys.exit(0)

    # Prepare image.
    if file_type.lower() == 'png':
        img = tf.image.decode_png(tf.read_file(args.img_path), channels=3)
    elif file_type.lower() == 'jpg':
        img = tf.image.decode_jpeg(tf.read_file(args.img_path), channels=3)
    else:
        print('cannot process {0} file.'.format(file_type))

    # Convert RGB to BGR.
    img_r, img_g, img_b = tf.split(axis=2, num_or_size_splits=3, value=img)
    img = tf.cast(tf.concat(axis=2, values=[img_b, img_g, img_r]), dtype=tf.float32)
    # Extract mean.
    img -= IMG_MEAN

    # Create network.
    net = DeepLabResNetModel({'data': tf.expand_dims(img, dim=0)}, is_training=False, num_classes=NUM_CLASSES)

    # Which variables to load.
    restore_var = tf.global_variables()

    # Predictions.
    raw_output = net.layers['fc_out']
    raw_output_up = tf.image.resize_bilinear(raw_output, tf.shape(img)[0:2,])
    raw_output_up = tf.argmax(raw_output_up, dimension=3)
    pred = tf.expand_dims(raw_output_up, dim=3)

    # Set up TF session and initialize variables.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()

    sess.run(init)

    # Load weights.
    ckpt = tf.train.get_checkpoint_state(args.restore_from)

    if ckpt and ckpt.model_checkpoint_path:
        loader = tf.train.Saver(var_list=restore_var)
        load_step = int(os.path.basename(ckpt.model_checkpoint_path).split('-')[1])
        load(loader, sess, ckpt.model_checkpoint_path)
    else:
        print('No checkpoint file found.')
        load_step = 0

    # Perform inference.
    preds = sess.run(pred)

    msk = decode_labels(preds, num_classes=NUM_CLASSES)
    im = Image.fromarray(msk[0])
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    im.save(SAVE_DIR + filename)

    print('The output file has been saved to {0}'.format(SAVE_DIR + filename))


if __name__ == '__main__':
    main()
