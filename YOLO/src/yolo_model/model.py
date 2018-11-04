import tensorflow
import os
import numpy as np
import pandas
import imageio
import codecs, json
from skimage.io import imread
from pathlib import Path

import data_processing.parser as parser

from yolo_model import B_BOX_SIDE as B_BOX_SIDE, CLASSES
from yolo_model import IMAGE_HEIGHT as IMAGE_HEIGHT
from yolo_model import IMAGE_WIDTH as IMAGE_WIDTH

 
from tensorflow.python.ops import array_ops
def conv2d_3x3(filters):
    return tensorflow.layers.Conv2D(filters, kernel_size=(3,3), activation=tensorflow.nn.relu, padding='same')

def max_pool():
    return tensorflow.layers.MaxPooling2D((2,2), strides=2, padding='same') 

def conv2d_transpose_2x2(filters):
    return tensorflow.layers.Conv2DTranspose(filters, kernel_size=(2, 2), strides=(2, 2), padding='same')

def concatenate(branches):
    return array_ops.concat(branches, 3)

num_classes = len(CLASSES)
def createModel(features, labels, mode):
    input_layer = tensorflow.reshape(features["x"], [-1, IMAGE_HEIGHT, IMAGE_WIDTH, 3])
    #Model taken from:
    #https://www.kaggle.com/piotrczapla/tensorflow-u-net-starter-lb-0-34
    c1 = conv2d_3x3(8) (input_layer)
    c1 = conv2d_3x3(8) (c1)
    p1 = max_pool() (c1)

    c2 = conv2d_3x3(16) (p1)
    c2 = conv2d_3x3(16) (c2)
    p2 = max_pool() (c2)

    c3 = conv2d_3x3(32) (p2)
    c3 = conv2d_3x3(32) (c3)
    p3 = max_pool() (c3)

    c4 = conv2d_3x3(64) (p3)
    c4 = conv2d_3x3(64) (c4)
    p4 = max_pool() (c4)

    c5 = conv2d_3x3(128) (p4)
    c5 = conv2d_3x3(128) (c5)

    u6 = conv2d_transpose_2x2(64) (c5)
    u6 = concatenate([u6, c4])
    c6 = conv2d_3x3(64) (u6)
    c6 = conv2d_3x3(64) (c6)

    u7 = conv2d_transpose_2x2(32) (c6)
    u7 = concatenate([u7, c3])
    c7 = conv2d_3x3(32) (u7)
    c7 = conv2d_3x3(32) (c7)

    u8 = conv2d_transpose_2x2(16) (c7)
    u8 = concatenate([u8, c2])
    c8 = conv2d_3x3(16) (u8)
    c8 = conv2d_3x3(16) (c8)

    u9 = conv2d_transpose_2x2(8) (c8)
    u9 = concatenate([u9, c1])
    c9 = conv2d_3x3(8) (u9)
    c9 = conv2d_3x3(8) (c9)

    c15 = tensorflow.layers.Conv2D(1, (1, 1)) (c9)
    c15 = tensorflow.layers.Flatten()(c15)
    #dense = tensorflow.layers.Dense(units = 1280)(c15)
    #dropout = tensorflow.layers.Dropout(rate=0.2)(dense)
    
    preds = tensorflow.layers.Dense(units = int( (IMAGE_HEIGHT/B_BOX_SIDE) * (IMAGE_WIDTH/B_BOX_SIDE) * (4+num_classes) ), activation=tensorflow.nn.sigmoid, kernel_initializer=tensorflow.contrib.layers.xavier_initializer() )(c15)
    predictions = {
        "preds": preds,
        }
    
    if mode == tensorflow.estimator.ModeKeys.PREDICT :
        return tensorflow.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    
    #loss = tensorflow.losses.mean_squared_error(labels=labels, predictions=preds)
    #How are the preds reshaped.
    reshapedPreds = tensorflow.reshape(preds, (-1, int(IMAGE_HEIGHT/B_BOX_SIDE), int(IMAGE_WIDTH/B_BOX_SIDE), num_classes+4))
    reshapedLabels = tensorflow.reshape(labels, (-1, int(IMAGE_HEIGHT/B_BOX_SIDE), int(IMAGE_WIDTH/B_BOX_SIDE), num_classes+4))
    
    #Cost calculation taken from https://stackoverflow.com/questions/48938120/make-tensorflow-ignore-values
    #This excludes bounding boxes that are 
    mask = tensorflow.tile(reshapedLabels[:, :, :, 0:1], [1, 1, 1, num_classes+4]) #repeating the first item 5 times
    mask_first = tensorflow.tile(reshapedLabels[:, :, :, 0:1], [1, 1, 1, 1]) #repeating the first item 1 time

    mask_of_ones = tensorflow.ones(tensorflow.shape(mask_first))

    full_mask = tensorflow.concat([tensorflow.to_float(mask_of_ones), tensorflow.to_float(mask[:, :, :, 1:])], 3)

    terms = tensorflow.multiply(full_mask, tensorflow.to_float(tensorflow.subtract(reshapedLabels, reshapedPreds, name="loss")))
    
    #Number of bounding boxes in the prediction.
    num_boxes = tensorflow.cast(tensorflow.count_nonzero(mask_first), dtype=tensorflow.float32)
    
    #Number of segments in the image.
    num_segments = tensorflow.cast(tensorflow.size(input = mask_first, out_type = tensorflow.int32), dtype=tensorflow.float32)
    
    #Number of terms that are counted (total output size minus x, y, w, h of segments with no bounding box).
    non_zeros = tensorflow.add(num_segments, tensorflow.multiply(num_boxes, 4.0))
    
    loss = tensorflow.div((tensorflow.reduce_sum(tensorflow.square(terms))), non_zeros, "loss_calc")
    
    if mode == tensorflow.estimator.ModeKeys.TRAIN:
        optimizer = tensorflow.train.AdamOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tensorflow.train.get_global_step())
        return tensorflow.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
        
    eval_metric_ops = {
        "accuracy": tensorflow.metrics.accuracy(
        labels=labels, predictions=preds)}
    return tensorflow.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
    
moddir = "saved_models"
current_model = tensorflow.estimator.Estimator(
    model_fn = createModel, model_dir=moddir)

def trainModel(unused_argv):
    imgs = []
    dims = []
    labels = []
    if not Path('imgs.npy').is_file():
        print("Creating input")
        imgs, dims, labels = parser.read_data()
        np.save("imgs.npy", np.array(imgs))
        np.save("dims.npy", np.array(dims))
        np.save("labels.npy", np.array(labels))
    else:
        print("Loading input")
        imgs = np.reshape(np.asarray(np.load('imgs.npy')).astype(np.float32), (-1, IMAGE_HEIGHT, IMAGE_WIDTH, 3))
        dims = np.reshape(np.asarray(np.load('dims.npy')).astype(np.int32), (-1))
        labels = np.reshape(np.asarray(np.load('labels.npy')).astype(np.float32), (-1, int(IMAGE_HEIGHT/B_BOX_SIDE)*int(IMAGE_WIDTH/B_BOX_SIDE)*(num_classes+4)))
        print()
    
    train_set_size = 5000
    train_imgs = imgs[0:train_set_size]
    train_dims = dims[0:train_set_size]
    train_labels = labels[0:train_set_size]
    
    test_imgs = imgs[train_set_size:]
    test_dims = dims[train_set_size:]
    test_labels = labels[train_set_size:]
    
    tensors_to_log = {}
    logging_hook = tensorflow.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)
    train_input_fn = tensorflow.estimator.inputs.numpy_input_fn(
        x={"x": np.array(train_imgs).astype(np.float32)},
        y=np.array(train_labels).astype(np.float32),
        batch_size=100,
        num_epochs=20,
        shuffle=True)

    current_model.train(
        input_fn=train_input_fn,
        steps=20000,
        hooks=[logging_hook])
    
    eval_input_fn = tensorflow.estimator.inputs.numpy_input_fn(
        x={"x": np.array(test_imgs).astype(np.float32)},
        y=np.array(test_labels).astype(np.float32),
        num_epochs=1,
        shuffle=False)
    current_model.evaluate(input_fn=eval_input_fn)
        
tensorflow.logging.set_verbosity(tensorflow.logging.INFO)
tensorflow.app.run(trainModel)
        
        