from tensorflow.contrib.layers.python.layers import batch_norm
import numpy as np
#from tensorflow.python.layers import batch_norm
def batch_normal(input , scope="scope" , reuse=False):
    return batch_norm(input , epsilon=1e-5, decay=0.9 , scale=True, scope=scope , reuse=reuse , updates_collections=None)

def print_similarity(pred, target_label_dict):
    """
    print the similarity of the models.
    :param pred:
    :param target_label_dict:
    :return:
    """
    similar_list = []
    for target in target_label_dict:
        G_res_index = np.argmax(pred[target, :])
        G_one = pred[:, G_res_index]
        real_y = target_label_dict[target]
        G_similar = np.sum(real_y * G_one)
        G_similar = G_similar / np.sum(real_y)
        # print(f"the target:{target}. The similarity is {G_similar}")
        if target % 1000 == 0:
            print("the target:%d. The similarity is %f" % (target, G_similar))
            print("The target values are:")
            print(pred[target, :])
        similar_list.append(G_similar)
    print("check mean similar values")
    print(np.mean(similar_list))


def print_mu(target_list, pred_dis_res,n_clusters):
    """
    print the mu1 of target nodes
    :param target_list:
    :param pred_dis_res:
    :param n_clusters:
    :return:
    """
    out_list = []
    for targets in target_list:
        target_pred = pred_dis_res[targets]
        max_index =  np.argmax(target_pred, axis=1)
        out_list.append(((len(np.unique(max_index)) - 1) / (np.max([n_clusters-1, 1]) * (np.max(np.bincount(max_index))))))
    print("The mu_1 is:%f"%(np.mean(out_list)))
    return np.mean(out_list)


def print_mu2(target_list, pred_dis_res, n_clusters):
    """
    print mu2 of the target groups
    :param target_list:
    :param pred_dis_res:
    :param n_clusters:
    :return:
    """
    out_list = []
    truth_list = [[] for x in range(n_clusters)]
    overall_n = 0
    for targets in target_list:
        for target in targets:
            target_pred = pred_dis_res[target]
            max_index = np.argmax(target_pred)
            truth_list[max_index].append(target)
            overall_n += 1
    for targets in target_list:
        target_pred = pred_dis_res[targets]
        max_indexes = np.argmax(target_pred, axis = 1)
        max_indexes = np.unique(max_indexes)
        group_len = 0
        for idx in max_indexes:
            group_len += len(truth_list[idx])
        group_len -= len(targets)
        const_denom = np.max([overall_n - len(targets), 1])
        out_list.append(group_len / const_denom)
    print("The mu2 is %f" %(np.mean(out_list)))
    return np.mean(out_list)


