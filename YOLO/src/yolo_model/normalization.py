import tensorflow as tf
import os
import numpy as np
import pandas

from yolo_model import B_BOX_SIDE as B_BOX_SIDE
from yolo_model import IMAGE_HEIGHT as IMAGE_HEIGHT
from yolo_model import IMAGE_WIDTH as IMAGE_WIDTH
from yolo_model import CLASSES as CLASSES

num_classes = len(CLASSES)
#Normalize the width and height by square rooting. The purpose is to make smaller values more visible.
def NormalizeWidthHeight(labels):
    
    rLabels = np.reshape(labels, (-1, int(IMAGE_HEIGHT/B_BOX_SIDE), int(IMAGE_WIDTH/B_BOX_SIDE), num_classes+4))
    widthHeight = rLabels[:,:,:,num_classes+2:]
    otherLabels = rLabels[:,:,:,0:num_classes+2]
    
    widthHeight = np.sqrt(widthHeight)
    
    normalizedVars = np.concatenate([otherLabels, widthHeight], axis = -1)
    normalizedVars = normalizedVars.flatten()
    normalizedVars = np.asarray(normalizedVars)
    return normalizedVars

def NormalizeWidthHeightForAll(allLabels):
    normLabels = []
    normalized = None
    for i in range(0, len(allLabels)):
        normalized = NormalizeWidthHeight(allLabels[i])
        normLabels.append(normalized)
    
    return np.asarray(normLabels).astype(np.float32)

#Undo normalization.
def unNormalize(labels):
    widthHeight = labels[:,:,num_classes+2:]
    otherLabels = labels[:,:,0:num_classes+2]
    widthHeight = np.multiply(widthHeight, widthHeight)
    unNormalLabels = np.concatenate([otherLabels, widthHeight], axis = -1)
    unNormalLabels = unNormalLabels.flatten()
    unNormalLabels = np.asarray(unNormalLabels)
    return unNormalLabels

def unNormalizeAll(labels):
    normLabels = []
    for i in range(0, len(labels)):
        normLabels.append(unNormalize(labels[i]))
    return normLabels

reshapedLabels = np.zeros((3, int(IMAGE_HEIGHT/B_BOX_SIDE), int(IMAGE_WIDTH/B_BOX_SIDE), num_classes+4))
#reshapedPreds = np.random.uniform(0.0, 1.0, (3, int(IMAGE_HEIGHT/B_BOX_SIDE), int(IMAGE_WIDTH/B_BOX_SIDE), num_classes+4))
reshapedPreds = np.zeros(((3, int(IMAGE_HEIGHT/B_BOX_SIDE), int(IMAGE_WIDTH/B_BOX_SIDE), num_classes+4)))

reshapedPreds[0][3][4][5]=1.0
reshapedPreds[0][3][4][10]=0.3
reshapedPreds[0][3][4][11]=0.5
reshapedPreds[0][3][4][12]=0.4
reshapedPreds[0][3][4][13]=0.7

reshapedPreds[0][6][7][8]=1.0
reshapedPreds[0][6][7][10]=0.4
reshapedPreds[0][6][7][11]=0.5
reshapedPreds[0][6][7][12]=0.7
reshapedPreds[0][6][7][13]=0.8

reshapedPreds[1][5][2][3]=1.0
reshapedPreds[1][5][2][10]=0.4
reshapedPreds[1][5][2][11]=0.2
reshapedPreds[1][5][2][12]=0.2
reshapedPreds[1][5][2][13]=0.4

reshapedPreds[2][6][1][2]=1.0
reshapedPreds[2][6][1][10]=0.8
reshapedPreds[2][6][1][11]=0.3
reshapedPreds[2][6][1][12]=0.3
reshapedPreds[2][6][1][13]=0.6

normed_preds = NormalizeWidthHeightForAll(reshapedPreds)
normed_preds = np.reshape(normed_preds, (-1,8,8,14))
print(normed_preds[0][3][4][10])
print(normed_preds[0][3][4][11])
print(normed_preds[0][3][4][12])
print(normed_preds[0][3][4][13])
unNormed = unNormalizeAll(normed_preds)
unNormed = np.reshape(unNormed, (-1,8,8,14))
print(unNormed[0][3][4][10])
print(unNormed[0][3][4][11])
print(unNormed[0][3][4][12])
print(unNormed[0][3][4][13])

