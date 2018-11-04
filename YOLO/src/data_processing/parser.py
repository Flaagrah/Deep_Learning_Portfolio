import imageio
import os
import numpy as np
from yolo_model import IMAGE_HEIGHT, IMAGE_WIDTH, B_BOX_SIDE, CLASSES, SINGULAR
import scipy.misc
from os.path import isfile, join

  
def read_data():
    imgs_dir = '../data/images/'
    anns_dir = '../data/labels/'
    file_list = os.listdir(imgs_dir)
    
    dims = []
    imgs = []
    anns = []
    
    for i in range(0, len(file_list)):
        f = file_list[i]
        img_dir = join(imgs_dir, f)
        ann_dir = f.replace(".png", ".txt")
        ann_reader = open(join(anns_dir, ann_dir))
        if not isfile(img_dir):
            continue
        
        img = imageio.imread(img_dir)
        print(img.shape)
        ann = parse_annotation(ann_reader)
        
        dims.append(img.shape)
        imgs.append(img)
        anns.append(ann)
    
    img_input = create_img_inputs(imgs)
    label_create = create_labels(dims, anns)
    
    return img_input, dims, label_create
        
        
def parse_bound_box_tuple(tuple):
    skin = tuple[1:len(tuple)-1]
    comm_index = skin.index(",")
    x = skin[:comm_index]
    y = skin[comm_index+2:]
    return int(x), int(y)

#Parse one annotation
def parse_annotation(reader):
    objs = []
    for line in reader:
        if "Bounding box" == line[0:12]:
            b_box = line[line.index(':')+2:]
            first_q = line[line.index("\"")+1:]
            obj_name = first_q[0:first_q.index("\"")]
            print(obj_name)
            print(b_box)
            
            sep_index = b_box.index("-")
            first_tuple = b_box[:sep_index-1]
            second_tuple = b_box[sep_index+2:len(b_box)-1]
            
            x1, y1 = parse_bound_box_tuple(first_tuple)
            x2, y2 = parse_bound_box_tuple(second_tuple)
            xCenter = int((x1+x2)/2)
            yCenter = int((y1+y2)/2)
            width = x2-x1
            height = y2-y1
            objs.append([obj_name, xCenter, yCenter, width, height])
            print(str(obj_name)+","+str(xCenter)+","+str(yCenter)+","+str(width)+","+str(height))
    return objs

#Divide input by 255 and flatten.
def create_img_inputs(imgs):
    reshaped_imgs = []
    for i in range(0, len(imgs)):
        img = imgs[i].astype(np.float32)
        img = scipy.misc.imresize(img, (IMAGE_HEIGHT, IMAGE_WIDTH, 3))
        #if (i == 0):
         #   scipy.misc.imsave("something.png", img)
        img = img/255.0
        reshaped_imgs.append(img.flatten())
    return reshaped_imgs

#Bounding boxes are layed out along a horizontal line.    
def create_labels(dims, parsed_anns):
    num_classes = len(CLASSES)
    label_units = (int(IMAGE_HEIGHT/B_BOX_SIDE)*int(IMAGE_WIDTH/B_BOX_SIDE)*(num_classes+4))
    labels = np.empty(shape=(0, label_units))
    #label_units = int((len(CLASSES)+4)*(IMAGE_HEIGHT/B_BOX_SIDE)*(IMAGE_WIDTH/B_BOX_SIDE))
    for j in range(0, len(parsed_anns)):
        ann = parsed_anns[j]
        label = np.zeros((1, int(IMAGE_HEIGHT/B_BOX_SIDE), int(IMAGE_WIDTH/B_BOX_SIDE), num_classes+4), dtype=np.float32)
        #create individual label.
        for i in range(0, len(ann)):
            print()
            b_box = ann[i]
            dim = dims[j]
            h_factor = IMAGE_HEIGHT/dim[0]
            w_factor = IMAGE_WIDTH/dim[1]
            print(ann)
            b_box[2] = int(b_box[2]*h_factor)
            b_box[4] = int(b_box[4]*h_factor)
            b_box[1] = int(b_box[1]*w_factor)
            b_box[3] = int(b_box[3]*w_factor)
            
            y_grid = int(b_box[2]/B_BOX_SIDE)
            x_grid = int(b_box[1]/B_BOX_SIDE)
            
            boxes_per_line = IMAGE_WIDTH/B_BOX_SIDE
            lab_index = boxes_per_line*y_grid
            lab_index += x_grid
            
            for k in range(0, len(CLASSES)):
                if SINGULAR[k] in b_box[0]:
                    label[0, y_grid, x_grid, k] = 1.0
                    label[0, y_grid, x_grid, num_classes] = (b_box[1]%B_BOX_SIDE)/B_BOX_SIDE
                    label[0, y_grid, x_grid, num_classes+1] = (b_box[2]%B_BOX_SIDE)/B_BOX_SIDE
                    label[0, y_grid, x_grid, num_classes+2] = b_box[3]/IMAGE_WIDTH
                    label[0, y_grid, x_grid, num_classes+3] = b_box[4]/IMAGE_HEIGHT
                    
        label = np.reshape(label, (1, label_units))
        labels = np.concatenate([labels, label], axis=0)
    
    return labels

def convert_to_output(dims, preds):
    pred_boxes = np.reshape(preds, (-1, 16, 16, len(CLASSES)+4))
    print()
         
    
