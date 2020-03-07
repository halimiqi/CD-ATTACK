import tensorflow as tf
#from utils import mkdir_p
from cdattack import cdattack
from optimizer import Optimizercdattack
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import normalize
import datetime
import numpy as np
import scipy.sparse as sp
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from tensorflow.python.client import device_lib
from preprocessing import preprocess_graph, construct_feed_dict, sparse_to_tuple, mask_test_edges,get_target_nodes_and_comm_labels, construct_feed_dict_trained
from ops import print_M1, print_M2
# set the random seed
np.random.seed(121)
tf.set_random_seed(121)
# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["KMP_WARNINGS"] = "FALSE"
flags.DEFINE_integer('n_clusters', 10, 'Number of clusters.')
#flags.DEFINE_string("target_index_list","10,35", "The index for the target_index")
flags.DEFINE_integer('epochs', 2000, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 32, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 16, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('hidden3', 32, 'Number of units in graphite hidden layers.')
flags.DEFINE_float('weight_decay', 0., 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('dropout', 0.3, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('g_scale_factor', 1- 0.75/2, 'the parametor for generate fake loss')
flags.DEFINE_float('d_scale_factor', 0.25, 'the parametor for community detection real loss')
flags.DEFINE_float('g_gamma', 1e-06, 'the parametor for generate loss, it has one term with encoder\'s loss')
flags.DEFINE_float('G_KL_r', 0.1, 'The r parameters for the G KL loss')
flags.DEFINE_float('mincut_r', 0.01, 'The r parameters for the cutmin loss orth loss')
flags.DEFINE_float('autoregressive_scalar', 0.2, 'the parametor for graphite generator')
flags.DEFINE_string('model', 'cdattack', 'Model string.')
flags.DEFINE_string('generator', 'dense', 'Which generator will be used') # the options are "inner_product", "graphite", "graphite_attention", "dense_attention" , "dense"
flags.DEFINE_string('dataset', 'dblp', 'Dataset string it is finance or dblp.')
flags.DEFINE_integer('features', 1, 'Whether to use features (1) or not (0).')
# seting from the vae gan
from tensorflow.python.client import device_lib
flags.DEFINE_integer("batch_size" , 64, "batch size")
flags.DEFINE_integer("max_iters" , 600000, "the maxmization epoch")
flags.DEFINE_integer("latent_dim" , 16, "the dim of latent code")
flags.DEFINE_float("learn_rate_init" , 1e-02, "the init of learn rate")
##PPNP parameters
flags.DEFINE_float('alpha', 0.1, 'the parametor for PPNP, the restart rate')
flags.DEFINE_string('dis_name', "PPNP", 'the name of community detection, it can be "GCN" and "PPNP"')
flags.DEFINE_string("trained_base_path", '191216023843', "The path for the trained base model")
flags.DEFINE_string("trained_our_path", '200304135412', "The path for the trained model")
flags.DEFINE_integer("k", 10, "The k edges to delete")
flags.DEFINE_integer('baseline_target_budget', 5, 'the parametor for graphite generator')
flags.DEFINE_integer("op", 1, "Training or Test")
flags.DEFINE_boolean('test', False, "Training or not")
###############################

placeholders = {
    'features': tf.sparse_placeholder(tf.float32),
    'adj': tf.sparse_placeholder(tf.float32),
    'adj_orig': tf.sparse_placeholder(tf.float32),
    'dropout': tf.placeholder_with_default(0., shape=()),
}

def get_new_adj(feed_dict, sess, model):
    new_adj = model.new_adj_without_norm.eval(session=sess, feed_dict=feed_dict)
    return new_adj

# Train model
def train(unused):
    if_drop_edge = True
    if_save_model = not FLAGS.test
    if_train_dis = False  # if train the community detection while training the generator part
    restore_trained_our = FLAGS.test
    showed_target_idx = 0   # the target index group of targets you want to show
    ##################################
    ### read and process the graph
    model_str = FLAGS.model
    dataset_str = FLAGS.dataset
    # Load data
    if FLAGS.dataset == "dblp":
        adj = sp.load_npz("data/dblp/dblp_medium_adj.npz")
        features = np.load("data/dblp/dblp_medium_features.npy")
        features_normlize = normalize(features, axis=0, norm='max')
        features = sp.csr_matrix(features_normlize)
        target_list = np.load("data/dblp/dblp_medium_label.npy")
    elif FLAGS.dataset == "finance":
        adj = sp.load_npz('./data/finance/Finance_large_adj.npz')
        features = np.load("data/finance/Finance_large_features.npy")
        features_normlize = normalize(features, axis=0, norm='max')
        features = sp.csr_matrix(features_normlize)
        target_list =  np.load("data/finance/Finance_large_label.npy")
    # Store original adjacency matrix (without diagonal entries) for later
    a = 1

    adj_orig = adj
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros()
    if FLAGS.features == 0:
        features = sp.identity(features.shape[0])  # featureless
    # Some preprocessing
    adj_norm, adj_norm_sparse = preprocess_graph(adj)
    num_nodes = adj.shape[0]
    features = sparse_to_tuple(features.tocoo())
    num_features = features[2][1]
    features_nonzero = features[1].shape[0]
    cost_val = []
    acc_val = []
    cost_val = []
    acc_val = []
    val_roc_score = []
    adj_label = adj_orig + sp.eye(adj.shape[0])
    adj_label_sparse = adj_label
    adj_label = sparse_to_tuple(adj_label)

    if_drop_edge = True
    ## set the checkpoint path
    checkpoints_dir_base = "./checkpoints"
    current_time = datetime.datetime.now().strftime("%y%m%d%H%M%S")
    checkpoints_dir = os.path.join(checkpoints_dir_base, current_time, current_time)

    tf.reset_default_graph()
    global_steps = tf.get_variable('global_step', trainable=False, initializer=0)
    new_learning_rate = tf.train.exponential_decay(FLAGS.learn_rate_init, global_step=global_steps, decay_steps=10000,
                                                   decay_rate=0.98)
    new_learn_rate_value = FLAGS.learn_rate_init
    ## set the placeholders
    placeholders = {
        'features': tf.sparse_placeholder(tf.float32, name= "ph_features"),
        'adj': tf.sparse_placeholder(tf.float32,name= "ph_adj"),
        'adj_orig': tf.sparse_placeholder(tf.float32, name = "ph_orig"),
        'dropout': tf.placeholder_with_default(0., shape=(), name = "ph_dropout"),
    }
    # build models
    model = None
    if model_str == "cdattack":
        model = cdattack(placeholders, num_features, num_nodes, features_nonzero, new_learning_rate, target_list, FLAGS.alpha, FLAGS.dis_name)
        model.build_model()
    pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
    opt = 0
    # Optimizer
    with tf.name_scope('optimizer'):
        if model_str == 'cdattack':
            opt = Optimizercdattack(preds=tf.reshape(model.x_tilde, [-1]),
                                  labels=tf.reshape(
                                      tf.sparse_tensor_to_dense(placeholders['adj_orig'], validate_indices=False),
                                      [-1]),
                                  model=model,
                                  num_nodes=num_nodes,
                                  pos_weight=pos_weight,
                                  norm=norm,
                                  target_list= target_list ,
                                  global_step=global_steps,
                                  new_learning_rate = new_learning_rate
                                  )
    # init the sess
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = ""
    var_list = tf.global_variables()
    saver = tf.train.Saver(var_list, max_to_keep=10)
    if if_save_model:
        os.mkdir(os.path.join(checkpoints_dir_base, current_time))
        saver.save(sess, checkpoints_dir)  # save the graph

    if restore_trained_our:
        checkpoints_dir_our = "./checkpoints"
        checkpoints_dir_our = os.path.join(checkpoints_dir_our, FLAGS.trained_our_path)
        saver.restore(sess,tf.train.latest_checkpoint(checkpoints_dir_our))
        print("model_load_successfully")
    feed_dict = construct_feed_dict(adj_norm, adj_label, features, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})
    pred_dis_res = model.vaeD_tilde.eval(session=sess, feed_dict=feed_dict)
    modified_adj = get_new_adj(feed_dict,sess, model)
    modified_adj = sp.csr_matrix(modified_adj)
    #####################################################
    G_loss_min = 1000
    if FLAGS.test == False:
        for epoch in range(FLAGS.epochs):
            t = time.time()
            if restore_trained_our:
                sess.run(opt.G_min_op, feed_dict=feed_dict)
            else: # it is the new model
                if epoch >= int(FLAGS.epochs / 2):
                    sess.run(opt.G_min_op, feed_dict=feed_dict)
                    if if_train_dis == True:
                        sess.run(opt.D_min_op, feed_dict=feed_dict)
                # run D optimizer
                if epoch < int(FLAGS.epochs / 2):
                    sess.run(opt.D_min_op_clean, feed_dict=feed_dict)

            if epoch % 50 == 0:
                print("Epoch:", '%04d' % (epoch + 1),
                      "time=", "{:.5f}".format(time.time() - t))
                D_loss_clean, D_loss, G_loss,new_learn_rate_value = sess.run([opt.D_mincut_loss_clean,opt.D_mincut_loss, opt.G_comm_loss,new_learning_rate],feed_dict=feed_dict)

                new_adj = model.new_adj_output.eval(session = sess, feed_dict = feed_dict)
                temp_pred = new_adj.reshape(-1)
                temp_ori = adj_label_sparse.todense().A.reshape(-1)
                print("Step %d:D_clean:loss = %.7f ,  D: loss = %.7f G: loss=%.7f , LR=%.7f" % (epoch,D_loss_clean, D_loss, G_loss,new_learn_rate_value))
                ## check the D_loss_min
                if (G_loss < G_loss_min) and (epoch > int(FLAGS.epochs / 2) + 1) and (if_save_model):
                    saver.save(sess, checkpoints_dir, global_step=epoch, write_meta_graph=False)
                    print("min G_loss new")
                if G_loss < G_loss_min:
                    G_loss_min = G_loss
            if (epoch % 200 ==0) and if_save_model:
                saver.save(sess,checkpoints_dir, global_step = epoch, write_meta_graph = False)
                print("Save the model at epoch:", '%04d' % (epoch + 1))
    if if_save_model:
        saver.save(sess, checkpoints_dir, global_step=FLAGS.epochs, write_meta_graph=False)
    print("Optimization Finished!")
    new_adj = get_new_adj(feed_dict,sess, model)
    ##### The final results ######
    feed_dict.update({placeholders['dropout']: 0})
    pred_dis_res = model.vaeD_tilde.eval(session=sess, feed_dict=feed_dict)
    print("*" * 15)
    print("The modified matrics")
    print_M1(target_list, pred_dis_res, FLAGS.n_clusters)
    print("*" * 15)
    print_M2(target_list, pred_dis_res, FLAGS.n_clusters)
    print("*" * 15)
    new_adj = get_new_adj(feed_dict,sess, model)
    x_tilde_out = model.new_adj_output.eval(session=sess, feed_dict=feed_dict)
    temp_pred = new_adj.reshape(-1)
    temp_ori = adj_norm_sparse.todense().A.reshape(-1)
    return

FLAGS = flags.FLAGS
if __name__ == "__main__":
    tf.app.run(train)
