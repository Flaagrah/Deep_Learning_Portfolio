'''
This is code from the Udemy course:
"Complete Guide to TensorFlow for Deep Learning with Python"
It is from the section on GAN's

https://www.udemy.com/complete-guide-to-tensorflow-for-deep-learning-with-python/
'''
import tensorflow as tf
import numpy as np
import gzip
from PIL import Image
from tensorflow.examples.tutorials.mnist import input_data
import os.path

mnist = input_data.read_data_sets('../03-Convolutional-Neural-Networks/MNIST_data/train-images-idx3-ubyte.gz', one_hot=True)
def extract_data(filename, num_images):
        """Extract the images into a 4D tensor [image index, y, x, channels].
        Values are rescaled from [0, 255] down to [-0.5, 0.5].
        """
        print('Extracting', filename)
        with gzip.open(filename) as bytestream:
            bytestream.read(16)
            buf = bytestream.read(28 * 28 * num_images)
            data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
            #data = (data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
            data = data.reshape(num_images, 28, 28, 1)
            return data
        
fname_img_train = extract_data('../Data/MNIST/train-images-idx3-ubyte.gz', 60000)

def generator(z, reuse=None):
    with tf.variable_scope('gen',reuse=reuse):
        hidden1 = tf.layers.dense(inputs=z,units=128)
        alpha = 0.01
        hidden1=tf.maximum(alpha*hidden1,hidden1)
        hidden2=tf.layers.dense(inputs=hidden1,units=128)
        hidden2 = tf.maximum(alpha*hidden2,hidden2)
        output=tf.layers.dense(hidden2,units=784, activation=tf.nn.tanh)
        return output

def discriminator(X, reuse=None):
    with tf.variable_scope('dis',reuse=reuse):
        hidden1=tf.layers.dense(inputs=X,units=128)
        alpha=0.01
        hidden1=tf.maximum(alpha*hidden1,hidden1)
        hidden2=tf.layers.dense(inputs=hidden1,units=128)
        hidden2=tf.maximum(alpha*hidden2,hidden2)
        logits=tf.layers.dense(hidden2,units=1)
        output=tf.sigmoid(logits)
        return output, logits
    
data_images=tf.placeholder(tf.float32,shape=[None,784])
z=tf.placeholder(tf.float32,shape=[None,100])
G = generator(z)
D_output_data, D_logits_data = discriminator(data_images)
D_output_gen, D_logits_gen = discriminator(G,reuse=True)

def loss_func(logits_in,labels_in):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=logits_in,labels=labels_in))
    
D_real_loss = loss_func(D_logits_data,tf.ones_like(D_logits_data)*0.9)
D_fake_loss = loss_func(D_logits_gen,tf.zeros_like(D_logits_gen))

D_loss = D_real_loss + D_fake_loss
G_loss = loss_func(D_logits_gen,tf.ones_like(D_logits_gen))

learning_rate = 0.001

tvars = tf.trainable_variables()
d_vars = [var for var in tvars if 'dis' in var.name]
g_vars = [var for var in tvars if 'gen' in var.name]

D_trainer = tf.train.AdamOptimizer(learning_rate).minimize(D_loss, var_list=d_vars)
G_trainer = tf.train.AdamOptimizer(learning_rate).minimize(G_loss, var_list=g_vars)

batch_size=100
epochs=30
set_size=60000

init = tf.global_variables_initializer()
samples=[]
def create_image(img, name):
        img = np.reshape(img, (28, 28))
        print("before")
        print(img)
        img = (np.multiply(np.divide(np.add(img, 1.0), 2.0),255.0).astype(np.int16))
        print("after")
        print(img)
        im = Image.fromarray(img.astype('uint8'))
        im.save(name)
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(epochs):
        #np.random.shuffle(fname_img_train)
        #num_batches=int(set_size/batch_size)
        num_batches=(mnist.train.num_examples)//batch_size
        for i in range(num_batches):
            batch1 = fname_img_train[i*batch_size:((i+1)*batch_size)]
            batch_images1 = batch1.reshape((batch_size,784))
            batch_images1 = (batch_images1/255.0)*2-1
            batch = mnist.train.next_batch(batch_size)
            batch_images = batch[0].reshape((batch_size,784))
            batch_images = batch_images * 2 - 1
            
            
            
            batch_z = np.random.uniform(-1,1,size=(batch_size,100))
            _ = sess.run(D_trainer, feed_dict={data_images:batch_images1,z:batch_z})
            _ = sess.run(G_trainer,feed_dict={z:batch_z})
            
        print("ON EPOCH {}".format(epoch))
        sample_z = np.random.uniform(-1,1,size=(batch_size,100))
        gen_sample = sess.run(G,feed_dict={z:sample_z})
        create_image(gen_sample[0], "img"+str(epoch)+".png")