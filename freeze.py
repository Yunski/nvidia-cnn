import os
import argparse
import tensorflow as tf

from tensorflow.python.framework import graph_util

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

"""
Freeze tensorflow graph.
Adapted from https://gist.github.com/morgangiraud/249505f540a5e53a48b0c1a869d370bf#file-medium-tffreeze-1-py
"""

dir = os.path.dirname(os.path.realpath(__file__))

def freeze_graph(model_folder, file_name, output_node_names):
    checkpoint = tf.train.get_checkpoint_state(model_folder)
    input_checkpoint = checkpoint.model_checkpoint_path
    
    absolute_model_folder = "/".join(input_checkpoint.split('/')[:-1])
    output_graph = absolute_model_folder + "/" + file_name + ".pb"

    clear_devices = True
    
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()

    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)
        output_graph_def = graph_util.convert_variables_to_constants(
            sess, 
            input_graph_def, 
            output_node_names.split(",") 
        ) 

        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", dest="model_folder", type=str, help="export destination folder", default='saved_models/track1')
    parser.add_argument("-f", dest="model_filename", nargs='?', const=1, default='model', type=str, help="name of exported .pb")
    parser.add_argument("-o", dest="output_node_names", nargs='?', const=1, default='output', type=str, help="names of output nodes; separate with commas if multiple output nodes")
    args = parser.parse_args()

    if not os.path.exists(args.model_folder):
        raise Exception("Destination folder does not exist.")

    freeze_graph(args.model_folder, args.model_filename, args.output_node_names)


if __name__ == '__main__':
    main()
