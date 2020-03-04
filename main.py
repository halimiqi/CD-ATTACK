import tensorflow as tf
#from utils import mkdir_p
from gaegan import gaegan
from optimizer import Optimizergaegan
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import normalize
import datetime
import numpy as np
import scipy.sparse as sp
import time
import os
#import sklearn.metrics.normalized_mutual_info_score as normalized_mutual_info_score
from sklearn.metrics import normalized_mutual_info_score
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
from preprocessing import preprocess_graph, construct_feed_dict, sparse_to_tuple, mask_test_edges,get_target_nodes_and_comm_labels, construct_feed_dict_trained
from ops import print_mu, print_mu2
# set the random seed
np.random.seed(121)
tf.set_random_seed(121)
# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
# flags.DEFINE_float('learning_rate', 0.005, 'Initial learning rate.')
flags.DEFINE_integer('n_clusters', 20, 'Number of epochs to train.')
flags.DEFINE_string("target_index_list","10,35", "The index for the target_index")
flags.DEFINE_integer('epochs', 2000, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 32, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 16, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('hidden3', 32, 'Number of units in graphite hidden layers.')
flags.DEFINE_float('weight_decay', 0., 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('dropout', 0.3, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('g_scale_factor', 1- 0.75/2, 'the parametor for generate fake loss')
flags.DEFINE_float('d_scale_factor', 0.25, 'the parametor for discriminator real loss')
flags.DEFINE_float('g_gamma', 1e-06, 'the parametor for generate loss, it has one term with encoder\'s loss')
flags.DEFINE_float('G_KL_r', 0.1, 'The r parameters for the G KL loss')
flags.DEFINE_float('mincut_r', 0.01, 'The r parameters for the cutmin loss orth loss')
flags.DEFINE_float('autoregressive_scalar', 0.2, 'the parametor for graphite generator')
flags.DEFINE_string('model', 'gae_gan', 'Model string.')
flags.DEFINE_string('generator', 'dense', 'Which generator will be used') # the options are "inner_product", "graphite", "graphite_attention", "dense_attention" , "dense"
flags.DEFINE_string('dataset', 'dblp', 'Dataset string it is qq or dblp.')
flags.DEFINE_integer('features', 1, 'Whether to use features (1) or not (0).')
# seting from the vae gan
from tensorflow.python.client import device_lib
flags.DEFINE_integer("batch_size" , 64, "batch size")
flags.DEFINE_integer("max_iters" , 600000, "the maxmization epoch")
flags.DEFINE_integer("latent_dim" , 16, "the dim of latent code")
flags.DEFINE_float("learn_rate_init" , 1e-02, "the init of learn rate")
##PPNP parameters
flags.DEFINE_float('alpha', 0.1, 'the parametor for PPNP, the restart rate')
flags.DEFINE_string('dis_name', "PPNP", 'the name of discriminator, it can be "GCN" and "PPNP"')
#Please set this num of repeat by the size of your datasets.
flags.DEFINE_integer("repeat", 1000, "the numbers of repeat for your datasets")
flags.DEFINE_string("trained_base_path", '191216023843', "The path for the trained model")
flags.DEFINE_string("trained_our_path", '191215231708', "The path for the trained model")
flags.DEFINE_integer("k", 10, "The k edges to delete")
flags.DEFINE_integer('baseline_target_budget', 5, 'the parametor for graphite generator')
flags.DEFINE_integer("op", 1, "Training or Test")
###############################
if_drop_edge = True
if_save_model = False
# if train the discriminator
if_train_dis = False
restore_trained_our = False
showed_target_idx = 0   # the target index group of targets you want to show
##################################
### read and process the graph
model_str = FLAGS.model
dataset_str = FLAGS.dataset
# Load data
if FLAGS.dataset == "dblp":
    adj = sp.load_npz("./data/dblp/dblp_adj_sparse_small.npz")
    features = np.load("./data/dblp/dblp_features_small.npy")
    features_normlize = normalize(features, axis=0, norm='max')
    features = sp.csr_matrix(features_normlize)
    target_list = np.load("./data/dblp/dblp_target_label_small.npy")
elif FLAGS.dataset == "qq":
    adj = sp.load_npz('./data/1215_qq_data_10_3/qq_adj_all_csr_5000_1215_10_3.npz')
    features = np.load("data/1215_qq_data_10_3/qq_features_5000_1215_10_3.npy")
    features_normlize = normalize(features, axis=0, norm='max')
    features = sp.csr_matrix(features_normlize)
    target_list =  np.load("data/1215_qq_data_10_3/qq_target_label_5000_1215_10_3.npy")
# Store original adjacency matrix (without diagonal entries) for later
a = 1


adj_orig = adj
adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
adj_orig.eliminate_zeros()

# adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
# adj = adj_train
if FLAGS.features == 0:
    features = sp.identity(features.shape[0])  # featureless
# Some preprocessing
adj_norm, adj_norm_sparse = preprocess_graph(adj)

placeholders = {
    'features': tf.sparse_placeholder(tf.float32),
    'adj': tf.sparse_placeholder(tf.float32),
    'adj_orig': tf.sparse_placeholder(tf.float32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    #'comm_label': tf.placeholder(tf.float32)
}

num_nodes = adj.shape[0]
features = sparse_to_tuple(features.tocoo())
num_features = features[2][1]
features_nonzero = features[1].shape[0]

# Create model

    #session part
cost_val = []
acc_val = []

cost_val = []
acc_val = []
val_roc_score = []

adj_label = adj_orig + sp.eye(adj.shape[0])
adj_label_sparse = adj_label
adj_label = sparse_to_tuple(adj_label)

def get_roc_score(edges_pos, edges_neg,feed_dict,sess, model, emb=None):
    if emb is None:
        feed_dict.update({placeholders['dropout']: 0})
        adj_rec = sess.run(model.x_tilde, feed_dict=feed_dict)

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Predict on test set of edges
    preds = []
    pos = []
    for e in edges_pos:
        #preds.append(sigmoid(adj_rec[e[0], e[1]]))
        preds.append(adj_rec[e[0], e[1]])
        pos.append(adj_orig[e[0], e[1]])
    preds_neg = []
    neg = []
    for e in edges_neg:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
        neg.append(adj_orig[e[0], e[1]])
    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)
    return roc_score, ap_score

def get_new_adj(feed_dict, sess, model):
    new_adj = model.new_adj_without_norm.eval(session=sess, feed_dict=feed_dict)
    return new_adj

# def train_dis_base():
#     tf.reset_default_graph()
#     if_drop_edge = False
#     ## set the placeholde
#     placeholders = {
#         'features': tf.sparse_placeholder(tf.float32, name= "ph_features"),
#         'adj': tf.sparse_placeholder(tf.float32,name= "ph_adj"),
#         'adj_orig': tf.sparse_placeholder(tf.float32, name = "ph_orig"),
#         'dropout': tf.placeholder_with_default(0., shape=(), name = "ph_dropout"),
#         #'comm_label': tf.placeholder(tf.float32, name = "ph_comm_label")
#     }
#
#     ## set the checkpoints path for base
#     checkpoints_dir_base = "./checkpoints/base"
#     current_time = datetime.datetime.now().strftime("%y%m%d%H%M%S")
#
#     checkpoints_dir = os.path.join(checkpoints_dir_base, current_time, current_time)
#     global_steps = tf.get_variable('global_step', trainable=False, initializer=0)
#     new_learning_rate = tf.train.exponential_decay(FLAGS.learn_rate_init, global_step=global_steps, decay_steps=1000,
#                                                    decay_rate=0.98)
#     # build models
#     model = None
#     if model_str == "gae_gan":
#         model = gaegan(placeholders, num_features, num_nodes, features_nonzero,
#                        new_learning_rate,target_list,FLAGS.alpha,FLAGS.dis_name,
#                        if_drop_edge)
#         model.build_model()
#     pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
#     norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
#     opt = 0
#     # Optimizer
#     with tf.name_scope('optimizer'):
#         if model_str == 'gae_gan':
#             opt = Optimizergaegan(preds=tf.reshape(model.x_tilde, [-1]),
#                                   labels=tf.reshape(
#                                       tf.sparse_tensor_to_dense(placeholders['adj_orig'], validate_indices=False),
#                                       [-1]),
#                                   #comm_label=placeholders["comm_label"],
#                                   model=model,
#                                   num_nodes=num_nodes,
#                                   pos_weight=pos_weight,
#                                   norm=norm,
#                                   target_list=target_list,
#                                   global_step=global_steps,
#                                   new_learning_rate=new_learning_rate,
#                                   if_drop_edge = if_drop_edge
#                                   )
#     # init the sess
#     sess_bass = tf.Session()
#     sess_bass.run(tf.global_variables_initializer())
#     #var_list = tf.global_variables()
#     #var_list = [var for var in var_list if "discriminate" in var.name]   # check if we need to do this
#     saver = tf.train.Saver(tf.global_variables(), max_to_keep=15)
#     if if_save_model:
#         os.mkdir(os.path.join(checkpoints_dir_base, current_time))
#         saver.save(sess_bass, checkpoints_dir)  # save the graph
#     feed_dict = construct_feed_dict(adj_norm, adj_label, features, placeholders)
#     feed_dict.update({placeholders['dropout']: FLAGS.dropout})
#
#     for epoch in range(FLAGS.epochs):
#         t = time.time()
#         # Construct feed dictionary
#         # run D optimizer
#         sess_bass.run(opt.D_min_op, feed_dict=feed_dict)   # here we only run the base model
#         # # Comoptpute average lossgaegan
#         #avg_acc_D = sess_bass.run(opt.D_accuracy, feed_dict)
#         if epoch % 50 == 0:
#             print("Epoch:", '%04d' % (epoch + 1),
#                   "time=", "{:.5f}".format(time.time() - t))
#             D_loss, new_learn_rate_value = sess_bass.run([opt.D_mincut_loss_test, new_learning_rate],
#                                                             feed_dict=feed_dict)
#             #new_adj = model.x_tilde_output.eval(session=sess_bass, feed_dict=feed_dict)
#             #temp_pred = new_adj.reshape(-1)
#             #temp_ori = adj_label_sparse.todense().A.reshape(-1)
#             #mutual_info = normalized_mutual_info_score(temp_pred, temp_ori)
#             print("Step %d: D: loss = %.7f , LR=%.7f" % (
#             epoch, D_loss, new_learn_rate_value))
#             ## check the target attack results
#             pred_dis_res = model.realD_tilde.eval(session=sess_bass, feed_dict=feed_dict)
#
#             # for targets in target_list:
#             #     for target in targets:
#             #         pred_target = pred_dis_res[target, :]
#             #         print("The target %d:" % (target))
#             #         print(pred_target)
#             targets = target_list[showed_target_idx]
#             # for target in targets:
#             #     pred_target = pred_dis_res[target, :]
#             #     print("The target %d:" % (target))
#             #     print(pred_target)
#             # print("#" * 30)
#         if (epoch % 200 == 1) and (if_save_model):
#             saver.save(sess_bass, checkpoints_dir, global_step=epoch, write_meta_graph=False)
#     saver.save(sess_bass, checkpoints_dir, global_step=FLAGS.epochs, write_meta_graph=False)
#     print("Optimization Finished!")
#     # new_adj = get_new_adj(feed_dict, sess_bass, model)
#     # np.save("./data/hinton/hinton_new_adj_48_0815.npy", new_adj)
#     ##### The final results ####
#     feed_dict.update({placeholders['dropout']: 0})
#     pred_dis_res = model.realD_tilde.eval(session=sess_bass, feed_dict=feed_dict)  # the prediction for the base model
#     print("*" * 30)
#     print("the final results for base model:\n")
#     # for targets in target_list:
#     #     for target in targets:
#     #         pred_target = pred_dis_res[target, :]
#     #         print("The target %d:" % (target))
#     #         print(pred_target)
#     targets = target_list[showed_target_idx]
#     for target in targets:
#         pred_target = pred_dis_res[target,:]
#         print("The target %d:" % (target))
#         print(pred_target)
#     print("*"*15)
#     print_mu(target_list, pred_dis_res, FLAGS.n_clusters)
#     print("*" * 15)
#     print_mu2(target_list, pred_dis_res, FLAGS.n_clusters)
#     print("*" * 15)
#     print("##*"*10 )
#     # new_adj = get_new_adj(feed_dict, sess_bass, model)
#     # temp_pred = new_adj.reshape(-1)
#     # temp_ori = adj_norm_sparse.todense().A.reshape(-1)
#     # mutual_info = normalized_mutual_info_score(temp_pred, temp_ori)
#     # #print(f"The mutual informations is {mutual_info}")
#     # print("The mutual informations is:")
#     # print(mutual_info)
#     return

# load the trained base model to check the results
# if_ori is if we use the adj original or the D-AD-. If the input is adj original we should convert it into normalized graph
# def trained_dis_base(adj_norm,new_adj, if_ori):
#     tf.reset_default_graph()
#     if_drop_edge = False
#     ## set the checkpoints path for base
#     checkpoints_dir_base = "./checkpoints/base"
#     checkpoints_dir = os.path.join(checkpoints_dir_base, FLAGS.trained_base_path)
#     checkpoints_dir_meta = os.path.join(checkpoints_dir_base, FLAGS.trained_base_path, FLAGS.trained_base_path + ".meta")
#     #global_steps = tf.get_variable('global_step', trainable=False, initializer=0)
#     with tf.Session() as sess:
#         #saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)
#         saver = tf.train.import_meta_graph(checkpoints_dir_meta)
#         graph = tf.get_default_graph()
#         ## set the placeholde
#         placeholders = {
#             'features': {"indices":graph.get_tensor_by_name("ph_features/indices:0"),
#                          "values" :graph.get_tensor_by_name("ph_features/values:0"),
#                          "shape": graph.get_tensor_by_name("ph_features/shape:0")
#                          },
#             'adj': {"indices":graph.get_tensor_by_name("ph_adj/indices:0"),
#                          "values" :graph.get_tensor_by_name("ph_adj/values:0"),
#                          "shape": graph.get_tensor_by_name("ph_adj/shape:0")
#                          },
#             'adj_orig': {"indices":graph.get_tensor_by_name("ph_orig/indices:0"),
#                          "values" :graph.get_tensor_by_name("ph_orig/values:0"),
#                          "shape": graph.get_tensor_by_name("ph_orig/shape:0")
#                          },
#             'dropout': graph.get_tensor_by_name("ph_dropout:0"),
#             #'comm_label': graph.get_tensor_by_name("ph_comm_label:0")
#         }
#         saver.restore(sess, tf.train.latest_checkpoint(checkpoints_dir))
#         adj_norm_trained = adj_norm
#         if not if_ori:
#             mutual_info = normalized_mutual_info_score(new_adj.reshape(-1), adj_label_sparse.todense().A.reshape(-1))
#             new_adj_nodiag = new_adj - np.diag(np.diag(new_adj))
#             adj_norm_trained, adj_norm_sparse_trained = preprocess_graph(new_adj_nodiag)
#             new_adj = sparse_to_tuple(sp.csr_matrix(new_adj))
#             print("The mutual information is:%f"%(mutual_info))
#         feed_dict = construct_feed_dict_trained(adj_norm_trained, new_adj, features, placeholders)
#         # feed_dict.update({placeholders['dropout']: FLAGS.dropout})
#         feed_dict.update({placeholders['dropout']: 0})
#         realD_tilde = graph.get_tensor_by_name("discriminate_1/Softmax_1:0")
#         pred_dis_res = realD_tilde.eval(session=sess, feed_dict=feed_dict)
#         # for targets in target_list:
#         #     for target in targets:
#         #         pred_target = pred_dis_res[target, :]
#         #         print("The target %d:" % (target))
#         #         print(pred_target)
#         targets = target_list[showed_target_idx]
#         for target in targets:
#             pred_target = pred_dis_res[target, :]
#             print("The target %d:" % (target))
#             print(pred_target)
#         print("#" * 30)
#         print("*" * 15)
#         print_mu(target_list, pred_dis_res, FLAGS.n_clusters)
#         print("*" * 15)
#         print_mu2(target_list, pred_dis_res, FLAGS.n_clusters)
#         print("*" * 15)
#         print("##*"*10)
#     return

# Train model
def train():
    if_drop_edge = True
    ## set the checkpoint path
    checkpoints_dir_base = "./checkpoints"
    current_time = datetime.datetime.now().strftime("%y%m%d%H%M%S")
    checkpoints_dir = os.path.join(checkpoints_dir_base, current_time, current_time)
    ############
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
        #'comm_label': tf.placeholder(tf.float32, name = "ph_comm_label")
    }
    # build models
    model = None
    if model_str == "gae_gan":
        model = gaegan(placeholders, num_features, num_nodes, features_nonzero, new_learning_rate, target_list, FLAGS.alpha, FLAGS.dis_name)
        model.build_model()
    pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
    opt = 0
    # Optimizer
    with tf.name_scope('optimizer'):
        if model_str == 'gae_gan':
            opt = Optimizergaegan(preds=tf.reshape(model.x_tilde, [-1]),
                                  labels=tf.reshape(
                                      tf.sparse_tensor_to_dense(placeholders['adj_orig'], validate_indices=False),
                                      [-1]),
                                  #comm_label=placeholders["comm_label"],
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
    #var_list = [var for var in var_list if "discriminate" in var.name]
    saver = tf.train.Saver(var_list, max_to_keep=10)
    if if_save_model:
        os.mkdir(os.path.join(checkpoints_dir_base, current_time))
        saver.save(sess, checkpoints_dir)  # save the graph

    if restore_trained_our:
        checkpoints_dir_our = "./checkpoints"
        checkpoints_dir_our = os.path.join(checkpoints_dir_our, FLAGS.trained_our_path)
        # checkpoints_dir_meta = os.path.join(checkpoints_dir_base, FLAGS.trained_our_path,
        #                                     FLAGS.trained_our_path + ".meta")
        #saver.restore(sess,tf.train.latest_checkpoint(checkpoints_dir_our))
        saver.restore(sess, os.path.join("./checkpoints","191215231708","191215231708-1601"))
        print("model_load_successfully")
    # else:  # if not restore the original then restore the base dis one.
    #     checkpoints_dir_base = os.path.join("./checkpoints/base", FLAGS.trained_base_path)
    #     saver.restore(sess, tf.train.latest_checkpoint(checkpoints_dir_base))

    feed_dict = construct_feed_dict(adj_norm, adj_label, features, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})
    pred_dis_res = model.vaeD_tilde.eval(session=sess, feed_dict=feed_dict)
    # targets = target_list[showed_target_idx]
    # for target in targets:
    #     pred_target = pred_dis_res[target, :]
    #     print("The target %d:" % (target))
    #     print(pred_target)
    # print("#" * 30)
    #### save new_adj without norm#############
    print("*" * 15)
    print("The modified adj mu")
    print_mu(target_list, pred_dis_res, FLAGS.n_clusters)
    print("*" * 15)
    print_mu2(target_list, pred_dis_res, FLAGS.n_clusters)
    print("*" * 15)
    print("The clean adj mu")
    clean_dis_res = model.realD_tilde.eval(session=sess, feed_dict=feed_dict)
    print_mu(target_list, clean_dis_res, FLAGS.n_clusters)
    print("*" * 15)
    print_mu2(target_list, clean_dis_res, FLAGS.n_clusters)
    print("*" * 15)
    modified_adj = get_new_adj(feed_dict,sess, model)
    modified_adj = sp.csr_matrix(modified_adj)
    # sp.save_npz("transfer_new/transfer_1216_1/qq_5000_gaegan_new.npz", modified_adj)
    # sp.save_npz("transfer_new/transfer_1216_1/qq_5000_gaegan_ori.npz", adj_orig)
    print("save the loaded adj")
    # print("before training generator")
    #####################################################
    G_loss_min = 1000
    for epoch in range(FLAGS.epochs):
        t = time.time()
        # run Encoder's optimizer
        #sess.run(opt.encoder_min_op, feed_dict=feed_dict)
        # run G optimizer  on trained model
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

        #avg_acc_D = sess.run(opt.D_accuracy, feed_dict)
        ##
        ##
        if epoch % 50 == 0:
            print("Epoch:", '%04d' % (epoch + 1),
                  "time=", "{:.5f}".format(time.time() - t))
            D_loss_clean, D_loss, G_loss,new_learn_rate_value = sess.run([opt.D_mincut_loss_clean,opt.D_mincut_loss, opt.G_comm_loss,new_learning_rate],feed_dict=feed_dict)
            #new_adj = get_new_adj(feed_dict, sess, model)
            new_adj = model.new_adj_output.eval(session = sess, feed_dict = feed_dict)
            temp_pred = new_adj.reshape(-1)
            #temp_ori = adj_norm_sparse.todense().A.reshape(-1)
            temp_ori = adj_label_sparse.todense().A.reshape(-1)
            #mutual_info = normalized_mutual_info_score(temp_pred, temp_ori)
            print("Step %d:D_clean:loss = %.7f ,  D: loss = %.7f G: loss=%.7f , LR=%.7f" % (epoch,D_loss_clean, D_loss, G_loss,new_learn_rate_value))
            ## check the D_loss_min
            if (G_loss < G_loss_min) and (epoch > int(FLAGS.epochs / 2) + 1) and (if_save_model):
                saver.save(sess, checkpoints_dir, global_step=epoch, write_meta_graph=False)
                print("min G_loss new")
            if G_loss < G_loss_min:
                G_loss_min = G_loss
        if (epoch % 200 ==1) and if_save_model:
            saver.save(sess,checkpoints_dir, global_step = epoch, write_meta_graph = False)
            print("Epoch:", '%04d' % (epoch + 1),
                  "time=", "{:.5f}".format(time.time() - t))
    if if_save_model:
        saver.save(sess, checkpoints_dir, global_step=FLAGS.epochs, write_meta_graph=False)
    print("Optimization Finished!")
    new_adj = get_new_adj(feed_dict,sess, model)
    ##### The final results ####
    feed_dict.update({placeholders['dropout']: 0})
    pred_dis_res = model.vaeD_tilde.eval(session=sess, feed_dict=feed_dict)
    print("*" * 30)
    print("the final results:\n")
    # for targets in target_list:
    #     for target in targets:
    #         pred_target = pred_dis_res[target, :]
    #         print("The target %d:" % (target))
    #         print(pred_target)
    targets = target_list[showed_target_idx]
    for target in targets:
        pred_target = pred_dis_res[target, :]
        print("The target %d:" % (target))
        print(pred_target)
    print("*" * 15)
    print("The modified adj mu")
    print_mu(target_list, pred_dis_res, FLAGS.n_clusters)
    print("*" * 15)
    print_mu2(target_list, pred_dis_res, FLAGS.n_clusters)
    print("*" * 15)
    print("The clean adj mu")
    clean_dis_res = model.realD_tilde.eval(session=sess, feed_dict=feed_dict)
    print_mu(target_list, clean_dis_res, FLAGS.n_clusters)
    print("*" * 15)
    print_mu2(target_list, clean_dis_res, FLAGS.n_clusters)
    print("*" * 15)
    new_adj = get_new_adj(feed_dict,sess, model)
    x_tilde_out = model.new_adj_output.eval(session=sess, feed_dict=feed_dict)
    temp_pred = new_adj.reshape(-1)
    temp_ori = adj_norm_sparse.todense().A.reshape(-1)
    #mutual_info = normalized_mutual_info_score(temp_pred, temp_ori)
    #################################### the KL for the discriminator
    gaegan_KL, dis_KL = sess.run([model.gaegan_KL, model.dis_KL], feed_dict = feed_dict)
    gaegan_pred, clean_pred = sess.run([model.Dis_z_gaegan, model.Dis_z_clean], feed_dict)

    print(gaegan_pred)
    print(clean_pred)
    print("**##")
    print("The modified KL is :")
    print(gaegan_KL)
    print("The clean KL is :")
    print(dis_KL)
    print("**##")
    ####################################################################
    #print(f"The mutual informations is {mutual_info}")
    # print("The mutual informations is:")
    # print(mutual_info)
    return new_adj, x_tilde_out
## delete edges between the targets and add some
# def base_line():
#     target_budget = np.random.choice(len(target_list), FLAGS.baseline_target_budget, replace = False)
#     target_budget = target_list[target_budget]
#     adj_base_sparse = adj_label_sparse
#     edge_list = []
#     for targets in target_budget:
#         for i in range(len(targets)):
#             for j in range(i+1,len(targets)):
#                 if adj_orig[targets[i],targets[j]] == 1:
#                     edge_list.append([targets[i],targets[j]])
#                 elif adj_orig[targets[j],targets[i]] == 1:
#                     edge_list.append([targets[i],targets[j]])
#     ## selected delete edges
#     edge_list = np.array(edge_list)
#     if edge_list !=[]:
#         selected_list_idx = np.random.choice(len(edge_list), min(int(FLAGS.k // 2), len(edge_list)), replace=False)
#         selected_list = edge_list[selected_list_idx]
#         for pair in selected_list:
#             adj_base_sparse[pair[0], pair[1]] = 0
#             adj_base_sparse[pair[1], pair[0]] = 0
#
#     ## add some edges on random
#     modified_num = FLAGS.baseline_target_budget
#     target_edge_list = []
#     edge_list = []
#     num_nodes = adj_orig.shape[0]
#     for targets in target_budget:
#         for i in range(modified_num):
#             for target in targets:
#                 random_node = np.random.choice(num_nodes, 1)
#                 while (adj_orig[target, random_node] ==1) or (random_node == target):
#                     random_node = np.random.choice(num_nodes,1)
#                 if [target, random_node] not in target_edge_list:
#                     target_edge_list.append([target, random_node])
#
#     ## selected add edges
#     selected_list_idx = np.random.choice(len(target_edge_list), FLAGS.k - int(FLAGS.k // 2), replace=False)
#     target_edge_list = np.array(target_edge_list)
#     selected_list = target_edge_list[selected_list_idx]
#     for pair in selected_list:
#         adj_base_sparse[pair[0], pair[1]] = 1
#         adj_base_sparse[pair[1], pair[0]] = 1
#     adj_base_new_dense= adj_base_sparse.todense().A
#     print("the clean adj")
#     trained_dis_base(adj_norm, adj_label, if_ori=True)
#     print("the baseline1 ")
#     trained_dis_base(adj_norm, adj_base_new_dense, if_ori=False)
#     #trained_dis_base(adj_norm, adj_base_sparse, if_ori=True)
#     ## save the baseline1 transfer adj
#     sp.save_npz("transfer_new/transfer_1216_1/qq_5000_base1_new.npz", adj_base_sparse)
#     sp.save_npz("transfer_new/transfer_1216_1/qq_5000_base1_ori.npz", adj_orig)
#     print("save the loaded model")
#     return
#
# def base_line_add():
#     edge_list = []
#     for targets in target_list:
#         for i in range(len(targets)):
#             for j in range(i+1,len(targets)):
#                 if adj_orig[targets[i],targets[j]] == 0:
#                     edge_list.append([targets[i],targets[j]])
#                 elif adj_orig[targets[j],targets[i]] == 0:
#                     edge_list.append([targets[i],targets[j]])
#     ## selected delete edges
#     edge_list = np.array(edge_list)
#     selected_list_idx = np.random.choice(len(edge_list), FLAGS.k, replace=False)
#     selected_list = edge_list[selected_list_idx]
#     adj_base_sparse = adj_label_sparse
#     for pair in selected_list:
#         adj_base_sparse[pair[0], pair[1]] = 1
#         adj_base_sparse[pair[1], pair[0]] = 1
#     adj_base_new_dense= adj_base_sparse.todense().A
#     print("the clean")
#     trained_dis_base(adj_norm, adj_label, if_ori=True)
#     print("The baseline1")
#     trained_dis_base(adj_norm, adj_base_new_dense, if_ori=False)
#     return
#
# #target 和随机其他点加边  the degree one
# def base_line2():
#     add_edge_list = []
#     delete_edge_list = []
#     adj_orig_dense = adj_orig.todense().A
#     num_nodes = adj_orig.shape[0]
#     Degree_nodes = np.array(adj_orig.sum(1))
#     Degree_nodes = Degree_nodes.reshape(-1)
#     target_flat = target_list.reshape(-1)
#     target_indexes = np.zeros(len(Degree_nodes))
#     target_indexes[target_flat] = 1
#     degree_targets = Degree_nodes * target_indexes
#     max_deg_indexes = np.argsort(degree_targets)[-1*int(FLAGS.k / 2):]
#     target_indexes = np.zeros(len(Degree_nodes))
#     target_indexes[target_flat] = 1
#     degree_targets = Degree_nodes * target_indexes
#     degree_targets[target_indexes == 0] = np.max(Degree_nodes)
#     degree_targets[degree_targets == 0] = np.max(Degree_nodes)  # we dont think about the node who does not have neighors
#     min_deg_indexes = np.argsort(degree_targets)[:(FLAGS.k - int(FLAGS.k / 2))]
#     ######################
#     for i in max_deg_indexes:  # add one edge for not neighbor
#         neighbors_idx = np.array(adj_orig[i,:].todense()).reshape(-1)
#         neighbor_indexes = np.ones(len(Degree_nodes))
#         neighbor_indexes[neighbors_idx == 1] = 0
#         neighbor_indexes[i] = 0
#         neighbors_degree = Degree_nodes * neighbor_indexes
#         other_idx = np.argmax(neighbors_degree)
#         add_edge_list.append([i,other_idx])
#     for i in min_deg_indexes:   # delete one edge for min neighbor
#         neighbors_idx = np.array(adj_orig[i, :].todense()).reshape(-1)
#         neighbor_indexes = np.zeros(len(Degree_nodes))
#         neighbor_indexes[neighbors_idx == 1] = 1
#         neighbors_degree = Degree_nodes * neighbor_indexes
#         neighbors_degree[neighbors_idx == 0] = np.max(Degree_nodes)
#         other_idx = np.argmin(neighbors_degree)
#         delete_edge_list.append([i, other_idx])
#     a = 1
#     adj_base_sparse = adj_label_sparse
#     for pair in add_edge_list:
#         adj_base_sparse[pair[0], pair[1]] = 1
#         adj_base_sparse[pair[1], pair[0]] = 1
#     for pair in delete_edge_list:
#         adj_base_sparse[pair[0], pair[1]] = 0
#         adj_base_sparse[pair[1], pair[0]] = 0
#     adj_base_new_dense= adj_base_sparse.todense().A
#     print("the clean adj")
#     trained_dis_base(adj_norm, adj_label, if_ori=True)
#     print("the baseline2 ")
#     trained_dis_base(adj_norm, adj_base_new_dense, if_ori=False)
#
#     ## save the baseline1 transfer adj
#     sp.save_npz("transfer_new/transfer_1216_1/qq_5000_base2_new.npz", adj_base_sparse)
#     sp.save_npz("transfer_new/transfer_1216_1/qq_5000_base2_ori.npz", adj_orig)
#     print("save the loaded model")
#     return
#
# #target 和随机其他点加边
# def base_line3():
#     target_budget = np.random.choice(len(target_list), FLAGS.baseline_target_budget, replace = False)
#     target_budget = target_list[target_budget]
#     modify_times = FLAGS.baseline_target_budget
#     adj_base_sparse = adj_label_sparse
#     target_edge_list = []
#     edge_list = []
#     num_nodes = adj_orig.shape[0]
#     for targets in target_budget:
#         for i in range(modify_times):
#             for target in targets:
#                 random_node = np.random.choice(num_nodes, 1)
#                 while (adj_orig[target, random_node] ==1) or (random_node == target):
#                     random_node = np.random.choice(num_nodes,1)
#                 if [target, random_node[0]] not in target_edge_list:
#                     target_edge_list.append([target, random_node[0]])
#
#     ## selected delete edges
#     selected_list_idx = np.random.choice(len(target_edge_list), FLAGS.k - int(FLAGS.k // 2), replace=False)
#     target_edge_list = np.array(target_edge_list)
#     selected_list = target_edge_list[selected_list_idx]
#     for pair in selected_list:
#         adj_base_sparse[pair[0], pair[1]] = 1
#         adj_base_sparse[pair[1], pair[0]] = 1
#
#     ## delete random list from target to others
#     target_edge_list = []
#     edge_list = []
#     num_nodes = adj_orig.shape[0]
#     for targets in target_budget:
#         for i in range(modify_times):
#             for target in targets:
#                 random_node = np.random.choice(num_nodes, 1)
#                 temp_idx = 0
#                 while (adj_orig[target, random_node] == 0) or (random_node == target):
#                     random_node = np.random.choice(num_nodes, 1)
#                     temp_idx += 1
#                     if temp_idx >2000:
#                         break
#                 if [target, random_node] not in target_edge_list:
#                     target_edge_list.append([target, random_node])
#
#     ## selected delete edges
#     selected_list_idx = np.random.choice(len(target_edge_list), int(FLAGS.k // 2), replace=False)
#     target_edge_list = np.array(target_edge_list)
#     selected_list = target_edge_list[selected_list_idx]
#     for pair in selected_list:
#         adj_base_sparse[pair[0], pair[1]] = 0
#         adj_base_sparse[pair[1], pair[0]] = 0
#     adj_base_new_dense= adj_base_sparse.todense().A
#     print("The clean one")
#     trained_dis_base(adj_norm, adj_label, if_ori=True)
#     print("the baseline3")
#     trained_dis_base(adj_norm, adj_base_new_dense, if_ori=False)
#
#     ## save the baseline1 transfer adj
#     sp.save_npz("transfer_new/transfer_1216_1/qq_5000_base3_new.npz", adj_base_sparse)
#     sp.save_npz("transfer_new/transfer_1216_1/qq_5000_base3_ori.npz", adj_orig)
#     print("save the loaded model")
#     return


# def test(saver,adj,features, meta_dir, checkpoints_dir):
#     adj_norm, adj_norm_sparse = preprocess_graph(adj)
#     placeholders = {
#         'features': tf.sparse_placeholder(tf.float32),
#         'adj': tf.sparse_placeholder(tf.float32),
#         'adj_orig': tf.sparse_placeholder(tf.float32),
#         'dropout': tf.placeholder_with_default(0., shape=())
#     }
#
#     num_nodes = adj.shape[0]
#     features = sparse_to_tuple(features.tocoo())
#     num_features = features[2][1]
#     features_nonzero = features[1].shape[0]
#     feed_dict = construct_feed_dict(adj_norm, adj_label, features, placeholders)
#     feed_dict.update({placeholders['dropout']: FLAGS.dropout})
#     # Create model
#     saver = tf.train.Saver(max_to_keep=10)
#     model = None
#     if model_str == "gae_gan":
#         model = gaegan(placeholders, num_features, num_nodes, features_nonzero)
#     pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
#     norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
#     global_steps = tf.get_variable(0, name="globals")
#     opt = 0
#     # Optimizer
#     with tf.name_scope('optimizer'):
#         if model_str == 'gae_gan':
#             opt = Optimizergaegan(preds=model.x_tilde,
#                                   labels=tf.reshape(
#                                       tf.sparse_tensor_to_dense(placeholders['adj_orig'], validate_indices=False),
#                                       [-1]),
#                                   model=model,
#                                   num_nodes=num_nodes,
#                                   pos_weight=pos_weight,
#                                   norm=norm,
#                                   global_step=global_steps
#                                   )
#
#         # session part
#     sess = tf.Session()
#     sess.run(tf.global_variables_initializer())
#
#     cost_val = []
#     acc_val = []
#     # load network
#     with tf.Session() as sess:
#         saver = tf.train.import_meta_graph(meta_dir)
#         saver.restore(sess, tf.train.latest_checkpoint(checkpoints_dir))
#         sess.run()
#         new_adj = get_new_adj(feed_dict)
#     return new_adj

FLAGS = flags.FLAGS
if __name__ == "__main__":
    #train_dis_base()

    new_adj, x_tilde_out = train()
    # print("The original base model")
    #trained_dis_base(adj_norm, adj_label, if_ori = True)  #
    # print("The modified model base model")
    #trained_dis_base(adj_norm, new_adj, if_ori=False)
    #print("The modified model base model using x_tilde")
    #trained_dis_base(adj_norm, x_tilde_out, if_ori=False)
    # print("finish")
    # base_line()
    # base_line2()
    # base_line3()
