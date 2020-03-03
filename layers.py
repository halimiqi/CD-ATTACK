from initializations import *
import tensorflow as tf
from ops import batch_normal
flags = tf.app.flags
FLAGS = flags.FLAGS

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs
    """
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def dropout_sparse(x, keep_prob, num_nonzero_elems):
    """Dropout for sparse tensors. Currently fails for very large sparse tensors (>1M elements)
    """
    noise_shape = [num_nonzero_elems]
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)

def zeros(shape, name=None):
    """All zeros."""
    initial = tf.zeros(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)


class Layer(object):
    """Base layer class. Defines basic API for all layer objects.

    # Properties
        name: String, defines the variable scope of the layer.

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
    """
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.issparse = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            outputs = self._call(inputs)
            return outputs


class GraphConvolution(Layer):
    """Basic graph convolution layer for undirected graph without edge labels."""
    def __init__(self, input_dim, output_dim, adj, dropout=0., act=tf.nn.relu, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(input_dim, output_dim, name="weights")
        self.dropout = dropout
        self.adj = adj
        self.act = act

    def _call(self, inputs):
        x = inputs
        x = tf.nn.dropout(x, 1-self.dropout)
        x = tf.matmul(x, self.vars['weights'])
        x = tf.sparse_tensor_dense_matmul(self.adj, x)
        outputs = self.act(x)
        return outputs


class GraphConvolutionSparse(Layer):
    """Graph convolution layer for sparse inputs."""
    def __init__(self, input_dim, output_dim, adj, features_nonzero, dropout=0., act=tf.nn.relu, **kwargs):
        super(GraphConvolutionSparse, self).__init__(**kwargs)
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(input_dim, output_dim, name="weights")
        self.dropout = dropout
        self.adj = adj
        self.act = act
        self.issparse = True
        self.features_nonzero = features_nonzero

    def _call(self, inputs):
        x = inputs
        x = dropout_sparse(x, 1-self.dropout, self.features_nonzero)
        x = tf.sparse_tensor_dense_matmul(x, self.vars['weights'])
        x = tf.sparse_tensor_dense_matmul(self.adj, x)
        outputs = self.act(x)
        return outputs

class GraphConvolution_denseadj(Layer):
    """Basic graph convolution layer for undirected graph without edge labels."""
    def __init__(self, input_dim, output_dim, adj, dropout=0., act=tf.nn.relu, **kwargs):
        super(GraphConvolution_denseadj, self).__init__(**kwargs)
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(input_dim, output_dim, name="weights")
        self.dropout = dropout
        self.adj = adj
        self.act = act

    def _call(self, inputs):
        x = inputs[0]
        new_adj = inputs[1]
        x = tf.nn.dropout(x, 1-self.dropout)
        x = tf.matmul(x, self.vars['weights'])
        x = tf.matmul(new_adj, x)
        outputs = self.act(x)
        return outputs


class GraphConvolutionSparse_denseadj(Layer):
    """Graph convolution layer for sparse inputs."""
    def __init__(self, input_dim, output_dim, adj, features_nonzero, dropout=0., act=tf.nn.relu, **kwargs):
        super(GraphConvolutionSparse_denseadj, self).__init__(**kwargs)
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(input_dim, output_dim, name="weights")
        self.dropout = dropout
        self.adj = adj
        self.act = act
        self.issparse = True
        self.features_nonzero = features_nonzero

    def _call(self, inputs):
        x = inputs[0]
        new_adj = inputs[1]
        x = dropout_sparse(x, 1-self.dropout, self.features_nonzero)
        x = tf.sparse_tensor_dense_matmul(x, self.vars['weights'])
        x = tf.matmul(new_adj, x)
        outputs = self.act(x)
        return outputs

class PPNP_Sparse_denseadj(Layer):
    """Graph convolution layer for sparse inputs."""
    def __init__(self, input_dim, output_dim, adj,num_nodes,  features_nonzero,alpha = 0.1, dropout=0., act=tf.nn.relu, **kwargs):
        super(PPNP_Sparse_denseadj, self).__init__(**kwargs)
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(input_dim, output_dim, name="weights")
        self.dropout = dropout
        self.adj = adj
        self.act = act
        self.issparse = True
        self.features_nonzero = features_nonzero
        self.alpha = alpha
        self.In = tf.eye(num_nodes)
    def _call(self, inputs):

        x = inputs[0]
        new_adj = inputs[1]
        new_adj = tf.linalg.inv(self.alpha * (self.In - (1 - self.alpha) * self.adj))
        x = dropout_sparse(x, 1-self.dropout, self.features_nonzero)
        x = tf.sparse_tensor_dense_matmul(x, self.vars['weights'])
        x = tf.matmul(new_adj, x)
        outputs = self.act(x)
        return outputs

class InnerProductDecoder(Layer):
    """Decoder model layer for link prediction."""
    def __init__(self, input_dim, dropout=0., act=tf.nn.sigmoid, **kwargs):
        super(InnerProductDecoder, self).__init__(**kwargs)
        self.dropout = dropout
        self.act = act

    def _call(self, inputs):
        inputs = tf.nn.dropout(inputs, 1-self.dropout)
        x = tf.transpose(inputs)
        x = tf.matmul(inputs, x)
        #x = tf.reshape(x, [-1])
        outputs = self.act(x)
        #outputs = tf.reshape(outputs, tf.sqrt())
        return outputs

class FullyConnect(Layer):
    def __init__(self, output_size , scope = None, stddev = 0.02, bias_start = 0.0, with_w = False,**kwargs):
        super(FullyConnect, self).__init__(**kwargs)
        #self.input = input
        self.output_size = output_size
        self.scope = scope
        self.stddev = stddev
        self.bias_start = bias_start
        self.with_w = with_w  # if the output compiled with w

    def _call(self, inputs):
        shape = inputs.get_shape().as_list()
        with tf.variable_scope(self.scope or "Linear"):

            matrix = tf.get_variable("Matrix", [shape[1], self.output_size], tf.float32,
                                     tf.random_normal_initializer(stddev=self.stddev))    # this is the weights w
            bias = tf.get_variable("bias", [self.output_size],
                                   initializer=tf.constant_initializer(self.bias_start))

            if self.with_w:
                return tf.matmul(inputs, matrix) + bias, matrix, bias
            else:

                return tf.matmul(inputs, matrix) + bias



class Scale(Layer):
    """Dense layer."""
    def __init__(self, input_dim, dropout=0., pos=False, sparse_inputs=False,
                 act=tf.nn.relu, bias=False, featureless=False, **kwargs):
        super(Scale, self).__init__(**kwargs)

        self.dropout = dropout
        self.act = act
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias

        with tf.variable_scope(self.name + '_vars'):
            self.vars['scale'] = zeros([1], name='weights')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs[0]
        y = inputs[1]

        return x * (1 - tf.nn.sigmoid(self.vars['scale'])) + y * tf.nn.sigmoid(self.vars['scale'])

class Dense(Layer):
    """Dense layer."""
    def __init__(self, input_dim, output_dim, dropout=0., pos=False, sparse_inputs=False,
                 act=tf.nn.relu, bias=False, featureless=False, **kwargs):
        super(Dense, self).__init__(**kwargs)

        self.dropout = dropout
        self.act = act
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(input_dim, output_dim, name='weights')
            if pos:
                self.vars['weights'] = tf.square(self.vars['weights'])
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        x = tf.nn.dropout(x, 1-self.dropout)
        output = tf.matmul(x, self.vars['weights'])

        # bias
        if self.bias:
            output += self.vars['bias']
        #output = batch_normal(output, scope = self.name + "_bn")
        return self.act(output)

class Graphite(Layer):
    """Graphite layer for undirected graph without edge labels."""
    def __init__(self, input_dim, output_dim, dropout=0., act=tf.nn.relu, **kwargs):
        super(Graphite, self).__init__(**kwargs)
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(input_dim, output_dim, name="weights")
        self.dropout = dropout
        self.act = act

    def _call(self, inputs):
        x = inputs[0]
        recon_1 = inputs[1]
        recon_2 = inputs[2]
        x = tf.matmul(x, self.vars['weights'])
        x = tf.matmul(recon_1, tf.matmul(tf.transpose(recon_1), x)) + tf.matmul(recon_2, tf.matmul(tf.transpose(recon_2), x))
        outputs = self.act(x)
        return outputs

class GraphiteSparse(Layer):
    """Graphite layer for sparse inputs."""
    def __init__(self, input_dim, output_dim, dropout=0., act=tf.nn.relu, **kwargs):
        super(GraphiteSparse, self).__init__(**kwargs)
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(input_dim, output_dim, name="weights")
        self.dropout = dropout
        self.act = act

    def _call(self, inputs):
        x = inputs[0]
        recon_1 = inputs[1]
        recon_2 = inputs[2]
        x = tf.sparse_tensor_dense_matmul(x, self.vars['weights'])
        x = tf.matmul(recon_1, tf.matmul(tf.transpose(recon_1), x)) + tf.matmul(recon_2, tf.matmul(tf.transpose(recon_2), x))
        outputs = self.act(x)
        return outputs


class Graphite_simple(Layer):
    """Graphite layer for undirected graph without edge labels."""
    def __init__(self, input_dim, output_dim, dropout=0., act=tf.nn.relu, **kwargs):
        super(Graphite_simple, self).__init__(**kwargs)
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(input_dim, output_dim, name="weights")
        self.dropout = dropout
        self.act = act

    def _call(self, inputs):
        x = inputs[0]
        adj = inputs[1]
        #recon_1 = inputs[1]
        #recon_2 = inputs[2]
        x = tf.matmul(x, self.vars['weights'])
        x = tf.sparse_tensor_dense_matmul(adj, x)
        #x = tf.matmul(recon_1, tf.matmul(tf.transpose(recon_1), x)) + tf.matmul(recon_2, tf.matmul(tf.transpose(recon_2), x))
        outputs = self.act(x)
        return outputs

class GraphiteSparse_simple(Layer):
    """Graphite layer for sparse inputs."""
    def __init__(self, input_dim, output_dim, dropout=0., act=tf.nn.relu, **kwargs):
        super(GraphiteSparse_simple, self).__init__(**kwargs)
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(input_dim, output_dim, name="weights")
        self.dropout = dropout
        self.act = act

    def _call(self, inputs):
        x = inputs[0]
        adj = inputs[1]
        #recon_1 = inputs[1]
        #recon_2 = inputs[2]
        x = tf.sparse_tensor_dense_matmul(x, self.vars['weights'])
        x = tf.sparse_tensor_dense_matmul(adj, x)
        #x = tf.matmul(recon_1, tf.matmul(tf.transpose(recon_1), x)) + tf.matmul(recon_2, tf.matmul(tf.transpose(recon_2), x))
        outputs = self.act(x)
        return outputs
