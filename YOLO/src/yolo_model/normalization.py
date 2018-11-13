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



