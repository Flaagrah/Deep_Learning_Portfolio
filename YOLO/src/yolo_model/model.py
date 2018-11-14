import tensorflow as tf
import os
import numpy as np
import pandas
import imageio
import codecs, json
from skimage.io import imread
from pathlib import Path

import data_processing.parser as parser
import yolo_model.normalization as normalization
import yolo_model.process_boxes as process_boxes
import yolo_model.data_augmentation as data_aug
from yolo_model import B_BOX_SIDE as B_BOX_SIDE, CLASSES, B_BOX_SIDE
from yolo_model import IMAGE_HEIGHT as IMAGE_HEIGHT
from yolo_model import IMAGE_WIDTH as IMAGE_WIDTH

from tensorflow.python.ops import array_ops
from datashape.coretypes import float32
from tensorflow.contrib.model_pruning.python.pruning import _weight_mask_variable

def conv2d_3x3(filters):
    return tf.layers.Conv2D(filters, kernel_size=(3,3), activation=tf.nn.relu, padding='same')

def max_pool():
    return tf.layers.MaxPooling2D((2,2), strides=2, padding='same') 

def conv2d_transpose_2x2(filters):
    return tf.layers.Conv2DTranspose(filters, kernel_size=(2, 2), strides=(2, 2), padding='same')

def concatenate(branches):
    return array_ops.concat(branches, 3)

num_classes = len(CLASSES)
batch=50
def createModel(features, labels, mode):
    input_layer = tf.reshape(features["x"], [-1, IMAGE_HEIGHT, IMAGE_WIDTH, 3])
    #Model taken from:
    #https://www.kaggle.com/piotrczapla/tf-u-net-starter-lb-0-34
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

    c15 = tf.layers.Conv2D(1, (1, 1)) (c9)
    c15 = tf.layers.Flatten()(c15)
    #dense = tf.layers.Dense(units = 1280)(c15)
    #dropout = tf.layers.Dropout(rate=0.2)(dense)
    
    preds = tf.layers.Dense(units = int( (IMAGE_HEIGHT/B_BOX_SIDE) * (IMAGE_WIDTH/B_BOX_SIDE) * (4+num_classes) ), activation=tf.nn.sigmoid, kernel_initializer=tf.contrib.layers.xavier_initializer() )(c15)
    predictions = {
        "preds": preds,
        }
    
    if mode == tf.estimator.ModeKeys.PREDICT :
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    
    #How are the preds reshaped.
    reshapedPreds = tf.reshape(preds, (-1, int(IMAGE_HEIGHT/B_BOX_SIDE), int(IMAGE_WIDTH/B_BOX_SIDE), num_classes+4))
    reshapedLabels = tf.reshape(labels, (-1, int(IMAGE_HEIGHT/B_BOX_SIDE), int(IMAGE_WIDTH/B_BOX_SIDE), num_classes+4))
    
    mask_sub1 = tf.reduce_sum(input_tensor=reshapedLabels, axis=3, keepdims=True)

    mask_sub2 = tf.clip_by_value(t=mask_sub1, clip_value_max=tf.constant(1.0), clip_value_min=tf.constant(0.0))
    mask_sub3 = tf.ceil(mask_sub2) 
    mask = tf.tile(mask_sub3[:, :, :, 0:1], [1, 1, 1, 4])
    num_terms = tf.reduce_sum(tf.reduce_sum(tf.reshape(mask, (-1, 4)), axis=1))
    
    #Weights for the flags (probability of detection)
    flag_weights = tf.multiply(tf.ones([num_classes], dtype=tf.float32), tf.constant(1.0))
    #Weights for the dimensions of image (x, y, width, height)
    dim_weights = tf.constant([1.0, 1.0, 1.0, 1.0])
    #Create weight mask to cancel out unwanted dimension units (x,y,w,h) in predicates
    weight_mask = tf.reshape(tf.concat([flag_weights, dim_weights], axis=0), shape=(1,1,1,num_classes+4))

    #Multiply square difference and apply mask.
    squared_diff = tf.multiply(tf.square(tf.subtract(reshapedPreds, reshapedLabels)), weight_mask)
    #Apply mask to cancel out unwanted dimension units.
    masked_boxes = tf.multiply(squared_diff[:,:,:,num_classes:], mask)
    masked_labels = tf.concat((squared_diff[:,:,:,0:num_classes], masked_boxes), axis=3)
    shape_linear = tf.reshape(masked_labels, (-1, int(IMAGE_HEIGHT/B_BOX_SIDE)*int(IMAGE_WIDTH/B_BOX_SIDE)*(num_classes+4)))
    
    #Calculate the number of units that are being summed
    num_class_labels = tf.convert_to_tensor(tf.constant(batch*num_classes*int(IMAGE_HEIGHT/B_BOX_SIDE)*int(IMAGE_WIDTH/B_BOX_SIDE), tf.float32))
    denominator = tf.add(num_terms, num_class_labels)
    #Calculate loss
    loss = tf.divide(tf.reduce_sum(tf.reduce_sum(shape_linear)), denominator)
    
    #Train
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
    
    #Evaluate
    if mode == tf.estimator.ModeKeys.EVAL:
        eval_metric_ops = {
            "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=preds)
            }
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
    
    
    
moddir = "saved_models"
current_model = tf.estimator.Estimator(
    model_fn = createModel, model_dir=moddir)

def trainModel(unused_argv):
    imgs = []
    dims = []
    labels = []
    #Create input if it doesn't exist already.
    if not Path('imgs.npy').is_file():
        print("Creating input")
        imgs, dims, labels = parser.read_data()
        np.save("imgs.npy", imgs)
        np.save("dims.npy", dims)
        np.save("labels.npy", labels)
    else:
        print("Loading input")
        imgs = np.reshape(np.asarray(np.load('imgs.npy')).astype(np.float32), (-1, IMAGE_HEIGHT, IMAGE_WIDTH, 3))
        dims = np.reshape(np.asarray(np.load('dims.npy')).astype(np.int32), (-1))
        labels = np.reshape(np.asarray(np.load('labels.npy')).astype(np.float32), (-1, int(IMAGE_HEIGHT/B_BOX_SIDE)*int(IMAGE_WIDTH/B_BOX_SIDE)*(num_classes+4)))
        print()
    
    #labels = normalization.NormalizeWidthHeightForAll(labels)
    #Add label smoothing so that the flags are between 0.1 and 0.9
    fLabels = np.reshape(labels, (-1, int(IMAGE_HEIGHT/B_BOX_SIDE), int(IMAGE_WIDTH/B_BOX_SIDE), num_classes+4))
    flags = fLabels[:,:,:,0:num_classes]
    flags = np.multiply(flags, 0.8)
    flags = np.add(flags, 0.1)
    fLabels[:,:,:,0:num_classes] = flags
    
    labels = np.reshape(fLabels, (-1, int(IMAGE_HEIGHT/B_BOX_SIDE)*int(IMAGE_WIDTH/B_BOX_SIDE)*(num_classes+4)))
    
    #Add data augmentations
    r1Imgs, r1Labs = data_aug.rotate_colour(np.copy(imgs), np.copy(labels))
    r2Imgs, r2Labs = data_aug.rotate_colour(np.copy(imgs), np.copy(labels))
    inverse_img, inverse_lab = data_aug.invert_colour(np.copy(imgs), np.copy(labels))
    flip_hor_img, flip_hor_lab = data_aug.flip_horizontal(np.copy(imgs), np.copy(labels))
    flip_ver_img, flip_ver_lab = data_aug.flip_vertical(np.copy(imgs), np.copy(labels))
    
    imgs = np.concatenate((imgs, r1Imgs, r2Imgs, inverse_img, flip_hor_img, flip_ver_img), axis=0)
    labels = np.concatenate((labels, r1Labs, r2Labs, inverse_lab, flip_hor_lab, flip_ver_lab), axis=0)
    
    train_set_size = 5000
    eval_size = 200
    #Train images and labels
    train_imgs = imgs[0:train_set_size]
    train_dims = dims[0:train_set_size]
    train_labels = labels[0:train_set_size]
    
    #Test images and labels
    test_imgs = imgs[train_set_size:train_set_size+eval_size]
    test_dims = dims[train_set_size:train_set_size+eval_size]
    test_labels = labels[train_set_size:train_set_size+eval_size]
    
    #Predictions
    pred_ind = train_set_size+eval_size
    pred_imgs = imgs[pred_ind:]
    pred_dims = dims[pred_ind:]
    pred_labels = labels[pred_ind:] 
    
    tensors_to_log = {}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": np.array(train_imgs).astype(np.float32)},
        y=np.array(train_labels).astype(np.float32),
        batch_size=batch,
        num_epochs=200,
        shuffle=True)
    #Train model
    current_model.train(input_fn=train_input_fn,steps=40000,hooks=[logging_hook])
    
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": np.array(test_imgs).astype(np.float32)},
        y=np.array(test_labels).astype(np.float32),
        shuffle=False)
    current_model.evaluate(input_fn=eval_input_fn)
    
    pred_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": np.array(pred_imgs).astype(np.float32)},
        shuffle=False)
    predicates = current_model.predict(input_fn=pred_input_fn)
    preds = []
    for pred in predicates:
        p = pred["preds"]
        p = np.reshape(p, (int(IMAGE_HEIGHT/B_BOX_SIDE), int(IMAGE_WIDTH/B_BOX_SIDE), num_classes+4))
        preds.append(p)
    #preds = normalization.unNormalizeAll(np.array(preds))
    boxes = process_boxes.getBoxes(preds)
    #Print out img # 5202 to 5211
    print("Box")
    print(boxes[1])
    print(boxes[2])
    print(boxes[3])
    print(boxes[4])
    print(boxes[5])
    print(boxes[6])
    print(boxes[7])
    print(boxes[8])
    print(boxes[9])
    print(boxes[10])
        
tf.logging.set_verbosity(tf.logging.INFO)
tf.app.run(trainModel)
        