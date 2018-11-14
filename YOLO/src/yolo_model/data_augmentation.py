from yolo_model import B_BOX_SIDE, IMAGE_HEIGHT, IMAGE_WIDTH, CLASSES
import numpy as np

ver_boxes = int(IMAGE_HEIGHT/B_BOX_SIDE)
hor_boxes = int(IMAGE_WIDTH/B_BOX_SIDE)

num_classes = len(CLASSES)
#Flip the image horizontally
def flip_horizontal(imgs, labels):
    labels = np.reshape(labels, (-1, ver_boxes, hor_boxes, num_classes+4))
    imgs = np.reshape(imgs, (-1, IMAGE_HEIGHT, IMAGE_WIDTH, 3))
    flipped_imgs = np.flip(imgs, 2)
    flipped_labels = np.flip(labels, 2)
    flipped_labels[:,:,:,num_classes+0]=(1.0-1.0/B_BOX_SIDE)-flipped_labels[:,:,:,num_classes+0]
    return flipped_imgs, np.reshape(flipped_labels, (-1, ver_boxes*hor_boxes*(num_classes+4)))

#Flip the image vertically
def flip_vertical(imgs, labels):
    labels = np.reshape(labels, (-1, ver_boxes, hor_boxes, num_classes+4))
    imgs = np.reshape(imgs, (-1, IMAGE_HEIGHT, IMAGE_WIDTH, 3))
    flipped_imgs = np.flip(imgs, 1)
    flipped_labels = np.flip(labels, 1)
    flipped_labels[:,:,:,num_classes+1]=(1.0-1.0/B_BOX_SIDE)-flipped_labels[:,:,:,num_classes+1]
    return flipped_imgs, np.reshape(flipped_labels, (-1, ver_boxes*hor_boxes*(num_classes+4)))

#Rotate the colours RGB
def rotate_colour(imgs, labels):
    new_imgs = np.copy(np.reshape(imgs, (-1, IMAGE_HEIGHT, IMAGE_WIDTH, 3)))
    red = imgs[:,:,:,0]
    green = imgs[:,:,:,1]
    blue = imgs[:,:,:,2]
    new_imgs[:,:,:,0]=blue
    new_imgs[:,:,:,1]=red
    new_imgs[:,:,:,2]=green
    return new_imgs, labels

#Invert the colour
def invert_colour(imgs, labels):
    new_imgs = 1.0-imgs
    return new_imgs, labels


    