Disclaimer

Introduction

This project uses the "VOC 2006 Database" found here: http://host.robots.ox.ac.uk/pascal/VOC/databases.html
This project is an implementation of the YOLO architecture for object detection. The 10 VOC object classes
are grouped into 3 classes for the sake of simplicity (vehicles, animals, and people). Because of the small
size of the dataset, learning 10 different object types seemed to be difficult for the network. 
A description of the Yolo architecture can be found here: https://pjreddie.com/media/files/papers/yolo.pdf

Network Model

The model is a UNET architecture that is based on the following research paper: https://arxiv.org/pdf/1505.04597.pdf

The implementation of the model was taken from: https://www.kaggle.com/piotrczapla/tensorflow-u-net-starter-lb-0-34
	
The loss function takes into account the flag of every segment (which indicates that there is a hit within the segment) and the
width, height, x, y of every segment that has a hit (ie, every segment with an object in it). This ignores the width/height/x/y 
of segments that don't have hits because they are not relevant to the accuracy. A simple mean squared error loss function was 
used. Flags, coordinates, and dimensions can be given different weights within the loss function depending on what needs to be
emphasized in training. For example, if the network is having trouble learning the width and height of objects, the weights for
width and height errors can be increased (or coordinate and flag weights can be decreased).
	
All resulting values in the output layer are between 0.0 and 1.0. The width and height of each segment are represented
by the square root of the width/height of the image itself (ie, if an object is 10% the height of the image height,
then height is the square root of 0.1). The x and y coordinates are represented by the coordinates relative to the segment divided by the 
segment width/height. For example, if the centre of the object is in the center of the segment, then the x and y would be 0.5 
and 0.5 respectively. If the object is in the top left of the segment, then x and y would be 0.0 and 0.0 respectively. 
	
The output of the model ends being 8X8X7 in size. 8X8 grid segments with each segment having three flags, a width variable,
height variable, x, and y. Output size is 384.
	
Data

The data is stored in a file after it is initially preprocessed. This is because it takes a long time for the images 
to be processed and so it speeds up testing if the processed data is stored in a file for future use.
	
Data Augmentation

Data augmentation is done by flipping the images horizontally and vertically, switching RGB values, and inverting RGB values.
flipping the images will allow the network to learn to detect shapes in different positions. Switching RGB values or inverting
allows the network to learn the outline of the shapes rather than memorizing common colours.
	
Normalization
	
Normalization takes the square root of the width and height of the images (all values are between 0.0 and 0.1). This is magnifies
the width/height error of smaller objects. The error in smaller objects is more significant than for bigger objects. For example, 
an error of 2 pixels has a greater effect on a objects of size 10 (20% miss), than a objects that is 100 pixels (2% miss). 
Therefore the model should be more sensitive to error on smaller objects.
	
Results and suggestions for improvement
After training for 200 epochs, the results for images 5202 to 5011 are the following:
[[0, 0.55037916, 16.0, 24.0, 0.0, 0.0]]
[[1, 0.92321914, 24.0, 32.0, 1.0, 15.0]]
[[2, 0.92931944, 25.0, 37.0, 3.0, 3.0], [0, 0.79125583, 32.0, 24.0, 2.0, 3.0], [0, 0.71798825, 40.0, 32.0, 0.0, 1.0]]
[[0, 0.77663624, 25.0, 32.0, 42.0, 59.0], [1, 0.7725394, 33.0, 33.0, 26.0, 30.0]]
[[1, 0.7779547, 27.0, 32.0, 31.0, 43.0]]
[[1, 0.59979737, 32.0, 25.0, 39.0, 23.0], [1, 0.5561943, 24.0, 16.0, 0.0, 0.0], [0, 0.5044279, 48.0, 40.0, 0.0, 0.0]]
[[0, 0.82546175, 36.0, 33.0, 6.0, 7.0], [0, 0.77222204, 32.0, 28.0, 38.0, 43.0]]
[[0, 0.55437344, 32.0, 35.0, 6.0, 20.0]]
[[0, 0.76426727, 24.0, 32.0, 18.0, 2.0], [0, 0.7504982, 33.0, 40.0, 64.0, 61.0], [0, 0.58262265, 48.0, 40.0, 0.0, 0.0], [0, 0.52472174, 8.0, 40.0, 0.0, 0.0], [1, 0.50002253, 24.0, 16.0, 0.0, 0.0]]
[[0, 0.6670166, 16.0, 32.0, 0.0, 0.0], [2, 0.6653167, 8.0, 32.0, 0.0, 0.0], [0, 0.6036443, 16.0, 24.0, 1.0, 0.0]]

Each line represents the predicted bounding boxes for each images.
The format of the bounding boxes is [flag_type, probability, x, y, w, h]
0 represents vehicle, 1 represents animal, and 2 represents person.
The x/y/w/h are relative to a 64X64 image (ie, x=32 would mean the x center is halfway across the width
and w=16 would mean the object is predicted to have a width that is one quarter of the image width).
The model seems to do a decent job of recognizing the objects but does not seem to accurately detect the
height and width. This leads to the IOU removal algorithm being unable to see that some boxes are detecting
the same object (since boxes that should overlap are not overlapping because of underestimated width/height).
The x and y coordinates are not accurate, which means that the wrong boxes are being flagged.

A possible solution to all of the above may be to add data augmentation that moves the bounding boxes (and
the sub-section of the image within the bounding box) around the image so that every segment of the image
can be trained to detect the object. The inaccuracy of the coordinates suggests that the segments detect
signs of an object within its range (ie, wheels, legs, etc), but has difficulty understanding if the object
is centered within it. Creating more variation in the object locations could help solve this. It is possible,
if not likely, that the inaccuracy of the width and height is related to the inaccuracy of the center because
the wrong box is being flagged (ie, the width and height units know there is no object there).