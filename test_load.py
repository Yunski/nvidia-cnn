import os
import argparse 
import tensorflow as tf
import numpy as np
from load import load_graph

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
"""
Adapted from https://gist.github.com/morgangiraud/4a062f31e8a7b71a030c2ced3277cc20#file-medium-tffreeze-3-py
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", dest='model_filename', type=str, help="frozen model .pb file to import")
    args = parser.parse_args()

    graph = load_graph(args.model_filename)

    for op in graph.get_operations():
        print(op.name)
        
    X = graph.get_tensor_by_name('prefix/X:0')
    keep_prob = graph.get_tensor_by_name('prefix/keep_prob:0')
    output = graph.get_tensor_by_name('prefix/output:0')
        
    with tf.Session(graph=graph) as sess:
        prediction = sess.run(output, feed_dict={
            X: np.random.randn(40, 66, 200, 3) * 100,
            keep_prob: 1.0
        })
        print(prediction)

