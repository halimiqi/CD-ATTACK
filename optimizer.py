import tensorflow as tf
import numpy as np
flags = tf.app.flags
FLAGS = flags.FLAGS

class Optimizercdattack(object):
    def __init__(self, preds, labels, model, num_nodes, pos_weight, norm, target_list, global_step, new_learning_rate, if_drop_edge = True):
        """
        The initial functions
        :param preds: it is not used in model
        :param labels: it is not used in model
        :param model: the model built from cdattack.py
        :param num_nodes: the number of the nodes
        :param pos_weight: not used in the model
        :param norm: not used in the model
        :param target_list: the target nodes: core members
        :param global_step: the global learning steps of model
        :param new_learning_rate: teh learning rate
        :param if_drop_edge: if drop the edges when learning the model
        """
        en_preds_sub = preds
        en_labels_sub = labels
        self.opt_op = 0  # this is the minimize function
        self.cost = 0  # this is the loss
        self.accuracy = 0  # this is the accuracy
        self.G_comm_loss = 0
        self.G_comm_loss_KL = 0
        self.num_nodes = num_nodes
        self.if_drop_edge = if_drop_edge
        # this is for vae, it contains two parts of losses:
        # self.encoder_optimizer = tf.train.RMSPropOptimizer(learning_rate = new_learning_rate)
        self.generate_optimizer = tf.train.RMSPropOptimizer(learning_rate= new_learning_rate)
        self.discriminate_optimizer = tf.train.RMSPropOptimizer(learning_rate = new_learning_rate)
        # encoder_varlist = [var for var in tf.trainable_variables() if 'encoder' in var.name]
        generate_varlist = [var for var in tf.trainable_variables() if (
                    'generate' in var.name) or ('encoder' in var.name)]  # the first part is generator and the second part is discriminator
        discriminate_varlist = [var for var in tf.trainable_variables() if 'discriminate' in var.name]
        #################### the new G_comm_loss
        for targets in target_list:
            targets_indices = [[x] for x in targets]
            #self.G_target_pred = model.vaeD_tilde[targets, :]
            self.G_target_pred = tf.gather_nd(model.vaeD_tilde, targets_indices)
            ## calculate the KL divergence
            for i in range(len(targets)):
                for j in range(i + 1, len(targets)):
                    if (i == 0) and (j == 1):
                        self.G_comm_loss_KL = -1 * tf.reduce_sum(
                            (self.G_target_pred[i] * tf.log(self.G_target_pred[i] / self.G_target_pred[j])))
                    else:
                        self.G_comm_loss_KL += -1*tf.reduce_sum((self.G_target_pred[i] * tf.log(self.G_target_pred[i] / self.G_target_pred[j])))
                    # to maximize the KL is to minimize the neg KL
        ######################################################
        ######################################################
        if if_drop_edge == True:
            self.mu = 0
            ## the new G_comm_loss
            for idx, targets in enumerate(target_list):
                target_pred = tf.gather(model.vaeD_tilde, targets)
                max_index = tf.argmax(target_pred, axis=1)
                max_index = tf.cast(max_index, tf.int32)
                if idx == 0:
                    self.mu = ((len(tf.unique(max_index)) - 1) / (
                                np.max([FLAGS.n_clusters - 1, 1]) * (tf.reduce_max(tf.bincount(max_index)))))
                else:
                    self.mu += ((len(tf.unique(max_index)) - 1) / (
                                np.max([FLAGS.n_clusters - 1, 1]) * (tf.reduce_max(tf.bincount(max_index)))))
            self.mu = tf.cast(self.mu, tf.float32)
            eij = tf.gather_nd(model.x_tilde_deleted, tf.where(model.x_tilde_deleted > 0))
            eij = tf.reduce_sum(tf.log(eij))
            self.G_comm_loss = (-1)* self.mu * eij + FLAGS.G_KL_r * self.G_comm_loss_KL
            #self.G_comm_loss = (-1) * self.mu * eij   # the loss without KL
        ######################################################
        # because the generate part is only inner product , there is no variable to optimize, we should change the format and try again
            self.G_min_op = self.generate_optimizer.minimize(self.G_comm_loss, global_step=global_step,
                                                                 var_list=generate_varlist)
        ######################################## the cutminloss for discriminator
        # if the the modified model
        if if_drop_edge == True:
            A_pool = tf.matmul(
                tf.transpose(tf.matmul(model.adj_ori_dense, model.vaeD_tilde)), model.vaeD_tilde)
            num = tf.diag_part(A_pool)

            D = tf.reduce_sum(model.adj_ori_dense, axis=-1)
            D = tf.matrix_diag(D)
            D_pooled = tf.matmul(
                tf.transpose(tf.matmul(D, model.vaeD_tilde)), model.vaeD_tilde)
            den = tf.diag_part(D_pooled)
            D_mincut_loss = -(1 / FLAGS.n_clusters) * (num / den)
            D_mincut_loss = tf.reduce_sum(D_mincut_loss)
            ## the orthogonal part loss
            St_S = (FLAGS.n_clusters / self.num_nodes) * tf.matmul(tf.transpose(model.vaeD_tilde), model.vaeD_tilde)
            I_S = tf.eye(FLAGS.n_clusters)  # here is I_k
            #ortho_loss =tf.norm(St_S / tf.norm(St_S) - I_S / tf.norm(I_S))
            ortho_loss =tf.square(tf.norm(St_S - I_S))
            # S_T = tf.transpose(model.vaeD_tilde, perm=[1, 0])
            # AA_T = tf.matmul(model.vaeD_tilde, S_T) - tf.eye(FLAGS.n_clusters)
            # ortho_loss = tf.square(tf.norm(AA_T))
            ## the overall cutmin_loss
            self.D_mincut_loss = D_mincut_loss + FLAGS.mincut_r * ortho_loss

            ###### at first we need to train the discriminator with clean one
            A_pool_clean = tf.matmul(
                tf.transpose(tf.matmul(model.adj_ori_dense, model.realD_tilde)), model.realD_tilde)
            num_clean = tf.diag_part(A_pool_clean)

            D_clean = tf.reduce_sum(model.adj_ori_dense, axis=-1)
            D_clean = tf.matrix_diag(D_clean)
            D_pooled_clean = tf.matmul(
                tf.transpose(tf.matmul(D_clean, model.realD_tilde)), model.realD_tilde)
            den_clean = tf.diag_part(D_pooled_clean)
            D_mincut_loss_clean = -(1 / FLAGS.n_clusters) * (num_clean / den_clean)
            D_mincut_loss_clean = tf.reduce_sum(D_mincut_loss_clean)
            ## the orthogonal part loss
            St_S_clean = (FLAGS.n_clusters / self.num_nodes) * tf.matmul(tf.transpose(model.realD_tilde), model.realD_tilde)
            I_S_clean = tf.eye(FLAGS.n_clusters)
            # ortho_loss =tf.norm(St_S / tf.norm(St_S) - I_S / tf.norm(I_S))
            ortho_loss_clean = tf.square(tf.norm(St_S_clean - I_S_clean))
            self.D_mincut_loss_clean = D_mincut_loss_clean + FLAGS.mincut_r * ortho_loss_clean
            ########
            self.D_min_op_clean = self.discriminate_optimizer.minimize(self.D_mincut_loss_clean, global_step=global_step,
                                                                 var_list=discriminate_varlist)
        ###################################### the clean discriminator model loss ##################
        else:
            A_pool = tf.matmul(
                tf.transpose(tf.matmul(model.adj_ori_dense, model.realD_tilde)), model.realD_tilde)
            num = tf.diag_part(A_pool)

            D = tf.reduce_sum(model.adj_ori_dense, axis=-1)
            D = tf.matrix_diag(D)
            D_pooled = tf.matmul(
                tf.transpose(tf.matmul(D, model.realD_tilde)), model.realD_tilde)
            den = tf.diag_part(D_pooled)
            D_mincut_loss = -(1 / FLAGS.n_clusters)*(num / den)
            D_mincut_loss = tf.reduce_sum(D_mincut_loss)
            ## the orthogonal part loss
            St_S = (FLAGS.n_clusters / self.num_nodes) * tf.matmul(tf.transpose(model.realD_tilde), model.realD_tilde)
            I_S = tf.eye(FLAGS.n_clusters)
            # ortho_loss =tf.norm(St_S / tf.norm(St_S) - I_S / tf.norm(I_S))
            ortho_loss = tf.square(tf.norm(St_S - I_S))

            # S_T = tf.transpose(model.vaeD_tilde, perm=[1, 0])
            # AA_T = tf.matmul(model.vaeD_tilde, S_T) - tf.eye(FLAGS.n_clusters)
            # ortho_loss = tf.square(tf.norm(AA_T))
            ## the overall cutmin_loss
            self.D_mincut_loss_test = D_mincut_loss + FLAGS.mincut_r * ortho_loss
        ########
        if self.if_drop_edge == False:
            self.D_min_op = self.discriminate_optimizer.minimize(self.D_mincut_loss_test, global_step=global_step,
                                                                 var_list=discriminate_varlist)
        else:
            self.D_min_op = self.discriminate_optimizer.minimize(self.D_mincut_loss, global_step=global_step,
                                                                 var_list=discriminate_varlist)
        ## this part is not correct now
        self.correct_prediction = tf.equal(tf.cast(tf.greater_equal(tf.sigmoid(model.realD_tilde), 0.5), tf.int32),
                                           tf.cast(tf.ones_like(model.realD_tilde), tf.int32))
        self.D_accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        return

