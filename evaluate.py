from __future__ import print_function
import argparse
from datetime import datetime
import os
import sys
import time

from PIL import Image
import tensorflow as tf
import numpy as np

from deeplab_resnet import DeepLabResNetModel, ImageReader, decode_labels, prepare_label

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

DATASET = 'ade20k' # change to 'ade20k' or 'nyu'
SNAPSHOT_DIR = './snapshots_ADE_od16'
SAVE_DIR = './output/'
IS_SAVE = False
MEASURE_TIME = False

def get_arguments():
    parser = argparse.ArgumentParser(description="DeepLab Network")
    parser.add_argument("--dataset", type=str, default=DATASET,
                        help="type of dataset")
    parser.add_argument("--restore_dir", type=str, default=SNAPSHOT_DIR,
                        help="location of restore weights")
    parser.add_argument("--measure_time", action="store_true",
                        help="whether to measure inference time")
    parser.add_argument("--is_save", action="store_true",
                        help="whether to save output")

    return parser.parse_args()

def initialize(dataset_name, measure_time):
    global DATA_DIRECTORY, DATA_LIST_PATH, IGNORE_LABEL, NUM_CLASSES, NUM_STEPS
    global ALLtime, ASPPtime, block5time, block4time, block3time, block2time, block1time, othertime

    if dataset_name == 'cityscape':
        DATA_DIRECTORY = '/data/cityscapes_dataset/cityscape'
        DATA_LIST_PATH = '/data/cityscapes_dataset/cityscape/list/eval_list.txt'
        IGNORE_LABEL = 255
        NUM_CLASSES = 19
        NUM_STEPS = 500 # Number of images in the validation set.
    elif dataset_name == 'nyu':
        DATA_DIRECTORY = ''
        DATA_LIST_PATH = '/home/yaaaaa0127/nyu_dataset/NYU/list/val_list.txt'
        IGNORE_LABEL = 0
        NUM_CLASSES = 4
        NUM_STEPS = 277 # Number of images in the validation set.
    elif dataset_name == 'ade20k':
        DATA_DIRECTORY = ''
        DATA_LIST_PATH = '/home/yaaaaa0127/ADEChallengeData2016/list/val_list.txt'
        IGNORE_LABEL = 0
        NUM_CLASSES = 27
        NUM_STEPS = 2000 # Number of images in the validation set.
    else:
        print('cannot find dataset {}'.format(dataset))

    if measure_time == True:
        ALLtime = []
        ASPPtime = []
        block5time = []
        block4time = []
        block3time = []
        block2time = []
        block1time = []
        othertime = []

def load(saver, sess, ckpt_path):
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))

def calculate_time(sess, All, ASPP, block5, block4, block3, block2, block1):
    start_time = time.time()
    _ = sess.run(All)
    time0 = time.time() - start_time

    start_time = time.time()
    _ = sess.run(ASPP)
    time1 = time.time() - start_time
    ASPP_time = time0 - time1

    start_time = time.time()
    _ = sess.run(block5)
    time2 = time.time() - start_time
    block5_time = time1 - time2

    start_time = time.time()
    _ = sess.run(block4)
    time3 = time.time() - start_time
    block4_time = time2 - time3

    start_time = time.time()
    _ = sess.run(block3)
    time4 = time.time() - start_time
    block3_time = time3 - time4

    start_time = time.time()
    _ = sess.run(block2)
    time5 = time.time() - start_time
    block2_time = time4 - time5

    start_time = time.time()
    _ = sess.run(block1)
    time6 = time.time() - start_time
    block1_time = time5 - time6

    ALLtime.append(time0)
    ASPPtime.append(ASPP_time)
    block5time.append(block5_time)
    block4time.append(block4_time)
    block3time.append(block3_time)
    block2time.append(block2_time)
    block1time.append(block1_time)
    othertime.append(time6)

def print_TimeInfo(num_step):
    totol_time = sum(ALLtime)/(num_step-1)
    ASPP_time = sum(ASPPtime)/(num_step-1)
    block5_time = sum(block5time)/(num_step-1)
    block4_time = sum(block4time)/(num_step-1)
    block3_time = sum(block3time)/(num_step-1)
    block2_time = sum(block2time)/(num_step-1)
    block1_time = sum(block1time)/(num_step-1)
    other_time = sum(othertime)/(num_step-1)

    print('total time:  {}'.format(totol_time))
    print('ASPP time:   {}'.format(ASPP_time))
    print('block5 time: {}'.format(block5_time))
    print('block4 time: {}'.format(block4_time))
    print('block3 time: {}'.format(block3_time))
    print('block2 time: {}'.format(block2_time))
    print('block1 time: {}'.format(block1_time))
    print('other time:  {}'.format(other_time))
    print('sum: {}'.format(
    ASPP_time + block1_time + block2_time + block3_time + block4_time + block5_time + other_time))

    print('average_time: {}'.format(totol_time - other_time))

def main():
    """Create the model and start the evaluation process."""
    args = get_arguments()

    initialize(args.dataset, args.measure_time)
    # Create queue coordinator.
    coord = tf.train.Coordinator()

    tf.reset_default_graph()
    # Load reader.
    with tf.name_scope("create_inputs"):
        reader = ImageReader(
            DATA_DIRECTORY,
            DATA_LIST_PATH,
            None, # No defined input size.
            False, # No random scale.
            False, # No random mirror.
            IGNORE_LABEL,
            IMG_MEAN,
            coord)
        image, label = reader.image, reader.label
    image_batch, label_batch = tf.expand_dims(image, dim=0), tf.expand_dims(label, dim=0) # Add one batch dimension.

    # Create network.
    net = DeepLabResNetModel({'data': image_batch}, is_training=False, num_classes=NUM_CLASSES)

    # Which variables to load.
    restore_var = tf.global_variables()

    # Predictions.
    raw_output = net.layers['fc_out']
    raw_output = tf.image.resize_bilinear(raw_output, tf.shape(image_batch)[1:3,])
    raw_output = tf.argmax(raw_output, dimension=3)
    pred = tf.expand_dims(raw_output, dim=3) # Create 4-d tensor.

    # mIoU
    pred_flatten = tf.reshape(pred, [-1,])
    gt = tf.reshape(label_batch, [-1,])

    if args.dataset == 'cityscape':
        mask = tf.less_equal(gt, NUM_CLASSES - 1)
    elif args.dataset == 'ade20k':
        less_equal_class = tf.less_equal(gt, NUM_CLASSES - 1)
        not_equal_zero = tf.not_equal(gt, 0)
        mask = tf.logical_and(less_equal_class, not_equal_zero)

    weights = tf.cast(mask, tf.int32) # Ignoring all labels greater than or equal to n_classes.
    mIoU, update_op = tf.contrib.metrics.streaming_mean_iou(predictions=pred_flatten, labels=gt, num_classes=NUM_CLASSES, weights=weights)

    All = net.layers['fc_out']
    ASPP = net.layers['res5c_relu']
    block5 = net.layers['res5a_relu']
    block4 = net.layers['res4b1_relu']
    block3 = net.layers['res3a_relu']
    block2 = net.layers['res2a_relu']
    block1 = net.layers['data']

    # Set up tf session and initialize variables.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()

    sess.run(init)
    sess.run(tf.local_variables_initializer())

    # Load weights.
    loader = tf.train.Saver(var_list=restore_var)

    ckpt = tf.train.get_checkpoint_state(args.store_dir)

    if ckpt and ckpt.model_checkpoint_path:
        loader = tf.train.Saver(var_list=restore_var)
        load_step = int(os.path.basename(ckpt.model_checkpoint_path).split('-')[1])
        load(loader, sess, ckpt.model_checkpoint_path)
    else:
        print('No checkpoint file found.')
        load_step = 0

    # Start queue threads.
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    for step in range(NUM_STEPS):
        preds, _ = sess.run([pred, update_op])

        if step > 0 and args.measure_time == True:
            calculate_time(sess, All, ASPP, block5, block4, block3, block2, block1)

        if args.is_save == True:
            msk = decode_labels(preds, num_classes=args.num_classes)
            im = Image.fromarray(msk[0])
            filename = 'mask' + str(step) + '.png'
            im.save(SAVE_DIR + filename)

        if step % 10 == 0:
            print('step {0} mIoU: {1}'.format(step, mIoU.eval(session=sess)))

    if args.measure_time == True:
        print_TimeInfo(NUM_STEPS)

    coord.request_stop()
    coord.join(threads)

if __name__ == '__main__':
    main()
