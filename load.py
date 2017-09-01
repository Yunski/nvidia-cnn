import tensorflow as tf

def load_graph(frozen_graph_filename):
    """
    Written by morgangiraud.
    See https://gist.github.com/morgangiraud/5ef49adc3c608bf639164b1dd5ed3dab#file-medium-tffreeze-2-py
    """

    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def, 
            input_map=None, 
            return_elements=None, 
            name="prefix", 
            op_dict=None, 
            producer_op_list=None
        )
    return graph
