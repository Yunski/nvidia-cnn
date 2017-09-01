import numpy as np
import tensorflow as tf
import math
import os

from utils import INPUT_SHAPE, batch_generator

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

class DenseLayer(object):
    """
    Fully Connected layer
    mi input neurons, mo output neurons
    """
    def __init__(self, idx, mi, mo, f=tf.nn.elu):
        self.idx = idx
        shape = [mi, mo]
        self.shape = shape
        self.W = tf.get_variable("W_dense{}".format(idx), shape=shape, 
                                 initializer=tf.contrib.keras.initializers.glorot_uniform())
        self.b = tf.get_variable("b_dense{}".format(idx), shape=[mo],
                                 initializer=tf.zeros_initializer(), dtype=tf.float32)
        self.params = [self.W, self.b]
        self.f = f

        
    def forward(self, X):
        linear = tf.nn.bias_add(tf.matmul(X, self.W), self.b)
        return self.f(linear)
    
    
    def summary(self):
        return "shape: {}, activation: {}".format(self.shape, self.f)
        
        
    def get_weights(self):
        return self.W, self.b
        
        
class ConvLayer(object):
    """
    2D Convolution Layer 
    Filter dimensions ksize x ksize x mi x mo
    """
    def __init__(self, idx, mi, mo, ksize, stride, f=tf.nn.elu):
        self.idx = idx
        shape = [ksize, ksize, mi, mo]
        self.shape = shape
        self.W = tf.get_variable("W_conv{}".format(idx), shape=shape,
                                 initializer=tf.contrib.keras.initializers.glorot_uniform())
        self.b = tf.get_variable("b_conv{}".format(idx), shape=[mo],
                                 initializer=tf.zeros_initializer(), dtype=tf.float32)
        self.ksize = ksize
        self.stride = stride
        self.params = [self.W, self.b]
        self.f = f
        
        
    def forward(self, X):
        conv = tf.nn.conv2d(X, self.W, strides=[1,self.stride,self.stride,1], padding='VALID')
        conv = tf.nn.bias_add(conv, self.b)
        return self.f(conv)
    
    
    def summary(self):
        return "shape: {}, activation: {}".format(self.shape, self.f)
        
        
    def get_weights(self):
        return self.W, self.b
    
    
class CNN(object):
    """
    Convolutional Neural Network implementation based on NVIDIA 
    End to End Model
    """
    def __init__(self, conv_layer_sizes, dense_layer_sizes):
        self.dense_layer_sizes = dense_layer_sizes
        self.conv_layer_sizes = conv_layer_sizes

        height, width, channels = INPUT_SHAPE
        ow = width
        oh = height
        mi = channels
        self.params = []
        self.layers = []
                
        self.conv_layers = []
        for i, (mo, ksize, stride) in enumerate(self.conv_layer_sizes):
            c = ConvLayer(i+1, mi, mo, ksize, stride)
            self.conv_layers.append(c)
            self.params += c.params
            self.layers.append(c)
            mi = mo
            ow = math.ceil((ow - ksize + 1) / stride)
            oh = math.ceil((oh - ksize + 1) / stride)
 
        mi = self.conv_layer_sizes[-1][0]*ow*oh

        self.dense_layers = []
        for i, mo in enumerate(self.dense_layer_sizes):
            d = DenseLayer(i+1, mi, mo)
            if i == len(dense_layer_sizes)-1:
                d.f = tf.identity
            self.dense_layers.append(d)
            self.params += d.params
            self.layers.append(d)
            mi = mo
        
                
    def fit(self, X_train, X_valid, Y_train, Y_valid, data_dir, save_dir=None, learning_rate=1e-4, p=0.5, epochs=10, samples_per_epoch=20000, batch_size=40, num_validation_batches=18, debug=False):
        learning_rate = np.float32(learning_rate)
        p = np.float32(p)
        
        height, width, channels = INPUT_SHAPE
        tfX = tf.placeholder(tf.float32, shape=[None, height, width, channels], name='X')
        tfY = tf.placeholder(tf.float32, shape=[batch_size], name='Y')
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        
        pY = self.forward(tfX, keep_prob)
        mse = tf.losses.mean_squared_error(pY, tfY) 
        train = tf.train.AdamOptimizer(learning_rate).minimize(mse)
        errors = []
        num_batches = samples_per_epoch // batch_size
      
        saver = tf.train.Saver() 
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
                   
            if debug:
                weights = self.get_weights()
                for w, b in weights:
                    print(sess.run(w), sess.run(b))

            train_batch_generator = batch_generator(data_dir, X_train, Y_train, batch_size)
            validation_batch_generator = batch_generator(data_dir, X_valid, Y_valid, batch_size, is_training=False)
            
            best_validation_error = None            
        
            for i in range(epochs):
                training_error_for_epoch = 0
                validation_error_for_epoch = 0
                        
                for j in range(num_batches):
                    X_batch, Y_batch = next(train_batch_generator)
                    _, training_error = sess.run([train, mse], feed_dict={tfX: X_batch, tfY: Y_batch, keep_prob: p})
                    training_error_for_epoch += training_error
                    print("epoch {} iter {} - training error: {}".format(i+1, j+1, training_error))
            
                for j in range(num_validation_batches):
                    X_batch_valid, Y_batch_valid = next(validation_batch_generator)
                    if debug:
                        prediction = sess.run(pY, feed_dict={tfX:X_batch_valid, keep_prob: 1.0})
                        print(prediction)
                    validation_error = sess.run(mse, feed_dict={tfX: X_batch_valid, tfY: Y_batch_valid, keep_prob:1.0})
                    errors.append(validation_error)
                    validation_error_for_epoch += validation_error
            
                avg_training_error = training_error_for_epoch / num_batches
                avg_validation_error = validation_error_for_epoch / num_validation_batches
                print("END OF EPOCH {} - error: {}, validation error: {}".format(i+1, avg_training_error, avg_validation_error))
            
                if save_dir and (best_validation_error is None or avg_validation_error < best_validation_error):
                    best_validation_error = avg_validation_error
                    saver.save(sess, os.path.join(save_dir, 'epoch-{}'.format(i+1)))

            if debug:
                weights = self.get_weights()
                for w, b in weights:
                    print(sess.run(w), sess.run(b))

        return errors
    
    
    def forward(self, X, keep_prob):
        Z = tf.map_fn(lambda x: x / 127.5 - 1.0, X)
        for c in self.conv_layers:
            Z = c.forward(Z)
        Z_drop = tf.nn.dropout(Z, keep_prob)
        Z_shape = Z_drop.get_shape().as_list()
        Z_flat = tf.reshape(Z_drop, [-1, np.prod(Z_shape[1:])])
        for d in self.dense_layers:
            Z_flat = d.forward(Z_flat)
        return tf.reshape(Z_flat, [-1], name='output')
    
    
    def summary(self):
        for layer in self.layers:
            print(layer.summary())
                   
                                    
    def get_weights(self):
        return [(layer.get_weights()[0], layer.get_weights()[1]) for layer in self.layers]
