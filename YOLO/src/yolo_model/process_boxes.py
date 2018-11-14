import numpy as np
from yolo_model import B_BOX_SIDE, IMAGE_HEIGHT, IMAGE_WIDTH, CLASSES

num_boxes_ver = int(IMAGE_HEIGHT/B_BOX_SIDE)
num_boxes_hor = int(IMAGE_WIDTH/B_BOX_SIDE)
num_classes = len(CLASSES)

hit_thresh = 0.5
IOU_thresh = 0.4

#Convert predicates to bounding boxes
def convertAllPredsToBoxes(preds):
    rPreds = np.reshape(preds, (-1, num_boxes_ver, num_boxes_hor, num_classes+4))
    
    boxes = []
    #for each predicate
    for p in range(0, len(rPreds)):
        pred = rPreds[p]
        imgBoxes = []
        #For each bounding box
        for i in range(0, num_boxes_ver):
            for j in range(0, num_boxes_hor):
                c = -1
                maxProb = 0.0
                #Get the class with the highest probability for each box
                for k in range(0, num_classes):
                    if pred[i][j][k]>maxProb:
                        maxProb=pred[i][j][k]
                        c=k
                #If the probability of the class is higher than the threshold, create a bounding box.
                if c>-1 and maxProb>=hit_thresh:
                    segmentY = i*B_BOX_SIDE
                    segmentX = j*B_BOX_SIDE
                    y = round(segmentY+pred[i][j][num_classes+1]*B_BOX_SIDE)
                    x = round(segmentX+pred[i][j][num_classes]*B_BOX_SIDE)
                    height = round(pred[i][j][num_classes+3]*IMAGE_HEIGHT)
                    width = round(pred[i][j][num_classes+2]*IMAGE_WIDTH)
                    imgBoxes.append([c, maxProb, x, y, width, height])
                    
        boxes.append(imgBoxes)
    return boxes

#Sort the bounding boxes by probability of detection. (Merge sort)
def sortByProb(b_boxes):
    size = len(b_boxes)
    if (size==0 or size==1):
        return b_boxes
    elif (size==2):
        if (b_boxes[0][1]<b_boxes[1][1]):
            temp = b_boxes[0]
            b_boxes[0]=b_boxes[1]
            b_boxes[1]=temp
        return b_boxes
    else:
        split_ind = int(size/2)
        left = b_boxes[0:split_ind]
        right = b_boxes[split_ind:]
        l_sort = sortByProb(left)
        r_sort = sortByProb(right)
        l_index = 0
        r_index = 0
        l_size = len(l_sort)
        r_size = len(r_sort)
        merged = []
        while l_index<l_size and r_index<r_size:
            if l_sort[l_index][1]>r_sort[r_index][1]:
                merged.append(l_sort[l_index])
                l_index = l_index+1
            else:
                merged.append(r_sort[r_index])
                r_index = r_index+1
                
        if l_index<l_size:
            merged = merged+l_sort[l_index:]
        if r_index<r_size:
            merged = merged+r_sort[r_index:]
        
        return merged

#Evaluate IOU 
def evalIOU(box1, box2):
    x1 = box1[2]
    y1 = box1[3]
    w1 = box1[4]
    h1 = box1[5]
    
    x2 = box2[2]
    y2 = box2[3]
    w2 = box2[4]
    h2 = box2[5]
    
    x1min = x1-int(w1/2)
    x1max = x1+round(w1/2)
    y1min = y1-int(h1/2)
    y1max = y1+round(h1/2)
    
    x2min = x2-int(w2/2)
    x2max = x2+round(w2/2)
    y2min = y2-int(h2/2)
    y2max = y2+round(h2/2)
    
    #Calculate overlap between line segments
    def overlap(max1, min1, max2, min2):
        top = - 1
        bottom = - 1
        
        if max1<=max2 and max1>=min2:
            top = max1
        elif max2<=max1 and max2>=min1:
            top = max2
        
        if not top == - 1:
            if min1>min2:
                bottom = min1
            else:
                bottom = min2
            return top-bottom
            
        return 0
    #Get vertical overlap
    ver_overlap = overlap(y1max, y1min, y2max, y2min)
    #Get horizontal overlap
    hor_overlap = overlap(x1max, x1min, x2max, x2min)
    #Intersection of bounding boxes
    intersection = ver_overlap*hor_overlap
    #Union of bounding boxes
    union = (y1max-y1min)*(x1max-x1min)+(y2max-y2min)*(x2max-x2min)-intersection
    #Avoiding error just in case
    if union==0:
        return 0
    return intersection/union

#Remove boxes that have high IOU scores
def removeBoxes(b_boxes):
    k=0
    while k<len(b_boxes):
        box = b_boxes[k]
        remove_indices = []
        for j in range(k+1, len(b_boxes)):
            iou = evalIOU(box, b_boxes[j])
            #Add to list of boxes to be removed
            if iou>IOU_thresh and not j in remove_indices:
                remove_indices.append(j)
        
        #Remove all of the indicated boxes
        new_b_boxes = []
        for i in range(0, len(b_boxes)):
            if not i in remove_indices:
                new_b_boxes.append(b_boxes[i])
        b_boxes = new_b_boxes
        k=k+1
        
    return b_boxes
                
#Get predicted bounding boxes from raw predicates
def getBoxes(preds):
    boxes = convertAllPredsToBoxes(preds)
    for i in range(0, len(boxes)):
        boxes[i] = sortByProb(boxes[i])
        boxes[i] = removeBoxes(boxes[i])
    return boxes
        
  
