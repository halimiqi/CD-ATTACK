import tensorflow as tf
import numpy as np
np.random.seed(121)

def weight_variable_glorot(input_dim, output_dim, name=""):
    """Create a weight variable with Glorot & Bengio (AISTATS 2010)
    initialization.
    """
    #tf.set_random_seed(135)
    # overall_dim = input_dim + output_dim
    # overall_dim = tf.cast(overall_dim, dtype=tf.float32)
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = tf.random_uniform([input_dim, output_dim], minval=-init_range,
                                maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)
