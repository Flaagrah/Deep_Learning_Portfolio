'''
This is derived from the Tutorial on how to create a GAN
from the course "Complete Guide to TensorFlow for Deep Learning with Python"
on Udemy.
https://www.udemy.com/complete-guide-to-tensorflow-for-deep-learning-with-python/
'''

import tensorflow as tf
import pickle
import numpy as np
import os
from PIL import Image 

#Extract the data
def unpickle():
    dict1 = pickle.load(open("../Data/CIFAR/data_batch_1", 'rb'), encoding='bytes')
    dict2 = pickle.load(open("../Data/CIFAR/data_batch_2", 'rb'), encoding='bytes')
    dict3 = pickle.load(open("../Data/CIFAR/data_batch_3", 'rb'), encoding='bytes')
    dict4 = pickle.load(open("../Data/CIFAR/data_batch_4", 'rb'), encoding='bytes')
    dict5 = pickle.load(open("../Data/CIFAR/data_batch_5", 'rb'), encoding='bytes')
    meta = pickle.load(open("../Data/CIFAR/batches.meta", 'rb'), encoding='bytes')
    print(dict1.keys())
    print(meta)
    np.frombuffer(dict1[b'data'], dtype=np.uint16)
    data = np.concatenate((dict1[b'data'], dict2[b'data'], dict3[b'data'], dict4[b'data'], dict5[b'data']), axis=0)
    labels = np.concatenate((dict1[b'labels'], dict2[b'labels'], dict3[b'labels'], dict4[b'labels'], dict5[b'labels']), axis=0)
    label_names = meta[b'label_names']
    return data, labels, label_names


data_dict, labels, names = unpickle()
print(data_dict[0])
images = data_dict
batch_size=100
set_size=50000

#Prepare the data by getting pictures of a certain object type.
def prepareData(data_dict, labels, names, wanted_pics):
    index = None
    for i in range(0, len(names)):
        if wanted_pics == names[i]:
            index = i
            break
    if index == None:
        return None
    total = np.empty((0, 3072))
    for i in range(0, len(data_dict)):
        if (labels[i])==(index):
            total = np.append(total, np.reshape(data_dict[i], newshape=(1, 3072)), axis=0)
    return total

#Get pictures of horses
images = prepareData(data_dict, labels, names, b'horse')
print("length")
print(len(images))
set_size=len(images)

#Generator that creates the artificial images.
def generator(z, reuse=None):
    with tf.variable_scope('gen',reuse=reuse):
        first_layer = tf.layers.dense(inputs=z, units=2048, activation=tf.nn.leaky_relu)
        reshaped = tf.reshape(first_layer, shape=(batch_size, 2, 2, 512))
        conv1 = tf.layers.conv2d_transpose(inputs=reshaped, filters=128, kernel_size=(2,2), strides=(2,2), activation=tf.nn.leaky_relu)
        conv2 = tf.layers.conv2d_transpose(inputs=conv1, filters=128, kernel_size=(2,2), strides=(2,2), activation=tf.nn.leaky_relu)
        conv3 = tf.layers.conv2d_transpose(inputs=conv2, filters=128, kernel_size=(2,2), strides=(2,2), activation=tf.nn.leaky_relu)
        conv4 = tf.layers.conv2d_transpose(inputs=conv3, filters=3, kernel_size=(2,2), strides=(2,2), activation=tf.nn.leaky_relu)
        flat = tf.layers.flatten(conv4)
        reshaped_output = tf.reshape(flat, shape=(batch_size, 32, 32, 3))
        return reshaped_output
 
#Discriminator that is used to distinguish between the artificial images and the real images.   
def discriminator(G, reuse=None):
    with tf.variable_scope('dis', reuse=reuse):
        reshaped_input = tf.reshape(G, shape=(batch_size, 32, 32, 3))
        normed = tf.layers.batch_normalization(inputs=reshaped_input, axis=3)
        
        conv1 = tf.layers.conv2d(inputs=normed, filters=128, kernel_size=(5,5), strides=(1,1), padding="same", activation=tf.nn.leaky_relu)
        norm1 = tf.layers.batch_normalization(inputs=conv1, axis=3)

        conv2 = tf.layers.conv2d(inputs=norm1, filters=128, kernel_size=(5,5), strides=(1,1), padding="same", activation=tf.nn.leaky_relu)
        norm2 = tf.layers.batch_normalization(inputs=conv2, axis=3)

        conv3 = tf.layers.conv2d(inputs=norm2, filters=128, kernel_size=(5,5), strides=(1,1), padding="same", activation=tf.nn.leaky_relu)
        norm3 = tf.layers.batch_normalization(inputs=conv3, axis=3)
 
        conv4 = tf.layers.conv2d(inputs=norm3, filters=128, kernel_size=(5,5), strides=(1,1), padding="same", activation=tf.nn.leaky_relu)
        norm4 = tf.layers.batch_normalization(inputs=conv4, axis=3)

        conv5 = tf.layers.conv2d(inputs=norm4, filters=128, kernel_size=(5,5), strides=(1,1), padding="same", activation=tf.nn.leaky_relu)
        norm5 = tf.layers.batch_normalization(inputs=conv5, axis=3)
        #Since the loss function is sigmoid cross entropy, there is no need to have an activation function here.
        logits = tf.layers.dense(inputs=norm5, units=1)
        return tf.nn.sigmoid(logits), logits

#Loss function is a sigmoid cross entropy
def loss_func(logits_in, labels_in):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=logits_in,labels=labels_in))
    

#Define the tensors that call the generator and discriminator.
real_images=tf.placeholder(tf.float32,shape=[None,3072])
z=tf.placeholder(tf.float32,shape=[None,100])
G = generator(z)
D_output_data, D_logits_data = discriminator(real_images)
D_output_gen, D_logits_gen = discriminator(G,reuse=True)

#Define the losses of the discriminator from the real images and the artificial images.
#Real = 1 and fake = 0.
#Label smoothing sets Real=0.9 and Fake=0.1 for labels. Prevents saturation of nodes.
D_real_loss = loss_func(D_logits_data,tf.ones_like(D_logits_data)*0.9)
D_fake_loss = loss_func(D_logits_gen,tf.zeros_like(D_logits_gen)+0.1)

#Total discriminator loss.
D_loss = D_real_loss + D_fake_loss
D_loss_print = tf.Print(D_loss, [D_loss], message="Disc loss: ")
#Total generator loss.
G_loss = loss_func(D_logits_gen,tf.ones_like(D_logits_gen)*0.9)
G_loss_print = tf.Print(G_loss, [G_loss], message="Gen loss: ")

#Generator has a lower learning rate than discriminator to make it trail the discriminator.
D_learning_rate = 0.001
G_learning_rate = 0.0005

tvars = tf.trainable_variables()
d_vars = [var for var in tvars if 'dis' in var.name]
g_vars = [var for var in tvars if 'gen' in var.name]
#Training for the Discriminator and Generator
D_trainer = tf.train.AdamOptimizer(D_learning_rate).minimize(D_loss, var_list=d_vars)
G_trainer = tf.train.AdamOptimizer(G_learning_rate).minimize(G_loss, var_list=g_vars)

epochs=200

init = tf.global_variables_initializer()
#Reshape the images to a different format (axis 3 is made to be the RGB of each pixel)
samples=[]
def reshape_images(imgs):
    red_channel = np.reshape(a=imgs[:, 0:1024], newshape=(set_size, 32, 32, 1))
    green_channel = np.reshape(a=imgs[:, 1024:2048], newshape=(set_size, 32, 32, 1))
    blue_channel = np.reshape(a=imgs[:, 2048:3072], newshape=(set_size, 32, 32, 1))
    
    colour_img = np.concatenate((red_channel, green_channel, blue_channel), axis=3)
    return colour_img

#Create an image given the output of the generator
def create_image(img, name):
    img = np.reshape(a=img, newshape=(32, 32, 3))
    #print("before")
    #print(img)
    img = np.multiply(np.divide(np.add(img, 1.0), 2.0),255.0).astype(np.int16)
    #print("after")
    #print(img)
    im = Image.fromarray(img.astype('uint8'), 'RGB')
    im.save(name)

reshaped_images = reshape_images(images)
model_file = "./model.ckpt"
saver = tf.train.Saver()
with tf.Session() as sess:
    #Restor a model if it already exists.
    if os.path.isfile(model_file+".meta"):
        print("Restoring")
        saver.restore(sess, model_file)
    else:
        print("Init")
        sess.run(init)
    
    for epoch in range(epochs):
        print("ON EPOCH {}".format(epoch))
        sample_z = np.random.uniform(-1,1,size=(batch_size,100))
        gen_sample = sess.run(G,feed_dict={z:sample_z})
        create_image(gen_sample[0], "img"+str(epoch)+".png")
        if epoch % 10 == 0:
            save_path = saver.save(sess, model_file)
        #np.random.shuffle(fname_img_train)
        num_batches=int(set_size/batch_size)
        for i in range(0, num_batches):
            batch = reshaped_images[i*batch_size:((i+1)*batch_size), :, :, :]
            batch_images = np.reshape(a=batch, newshape=(batch_size,3072))
            batch_images = (batch_images/255.0 )* 2 - 1
            batch_z = np.random.uniform(-1,1,size=(batch_size,100))
            #Run the trainers and print the losses at the start and end of the epoch.
            if i == 0 or i == num_batches - 1:
                loss = sess.run(fetches=[D_loss, D_trainer], feed_dict={real_images:batch_images,z:batch_z}) 
                print("Disc loss: "+str(loss))
                loss = sess.run(fetches=[G_loss, G_trainer], feed_dict={z:batch_z})
                print("Gen loss: "+str(loss))
            else:
                loss = sess.run(fetches=D_trainer, feed_dict={real_images:batch_images,z:batch_z}) 
                loss = sess.run(fetches=G_trainer, feed_dict={z:batch_z})
            
    