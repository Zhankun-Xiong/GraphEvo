import numpy as np
import scipy.sparse as sp
import scipy.io as spio
import tensorflow as tf
import gc
import random
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MinMaxScaler

import random
from datetime import datetime
import os
import requests
import pandas as pd
from sklearn.metrics import recall_score, roc_auc_score
import numpy as np
import csv
import sqlite3
import csv
import numpy as np
import random
import os
import sys
import networkx as nx
from numpy import linalg as LA
import math
from numpy.linalg import inv
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import MinMaxScaler
import copy
from numpy import linalg as LA
import csv
import array
import random
import numpy

from scipy.linalg import pinv as pinv
#import keras
# from keras import regularizers
# from keras import initializers
# from keras.models import Sequential, Model, load_model, save_model
# from keras.layers.core import Dense, Lambda, Activation
# from keras.layers import Embedding, Input, Dense, merge, Reshape, Flatten
# from keras.optimizers import Adagrad, Adam, SGD, RMSprop
# from keras.regularizers import l2
# from keras.layers import normalization
from time import time
import multiprocessing as mp
import sys
import math


flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('hidden1', 128, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 128, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('hidden3', 128, 'Number of units in hidden layer 3.')








def get_Jaccard2_Similarity(interaction_matrix):
    X = np.mat(interaction_matrix)
    E = np.ones_like(X.T)
    denominator=X * E + E.T * X.T - X * X.T
    denominator_zero_index=np.where(denominator==0)
    denominator[denominator_zero_index]=1
    result = X * X.T / denominator
    result[denominator_zero_index]=0
    result = result - np.diag(np.diag(result))
    return result

def matrix_normalize(similarity_matrix):
    similarity_matrix[np.isnan(similarity_matrix)] = 0
    if similarity_matrix.shape[0] == similarity_matrix.shape[1]:
        similarity_matrix = similarity_matrix - np.diag(np.diag(similarity_matrix))
        for i in range(200):
            D = np.diag(np.array(np.sum(similarity_matrix, axis=1)).flatten())
            xxx=np.sqrt(D)
            xxx[np.isnan(xxx)]=0
            D = np.linalg.pinv(xxx)
            similarity_matrix = D * similarity_matrix * D
    else:
        for i in range(similarity_matrix.shape[0]):
            if np.sum(similarity_matrix[i], axis=1) == 0:
                similarity_matrix[i] = similarity_matrix[i]
            else:
                similarity_matrix[i] = similarity_matrix[i] / np.sum(similarity_matrix[i], axis=1)
    return similarity_matrix
def get_metrics(real_score, predict_score):
    sorted_predict_score = np.array(sorted(list(set(np.array(predict_score).flatten()))))
    sorted_predict_score_num = len(sorted_predict_score)
    thresholds = sorted_predict_score[np.int32(sorted_predict_score_num * np.arange(1, 1000) / 1000)]
    thresholds = np.mat(thresholds)
    thresholds_num = thresholds.shape[1]

    predict_score_matrix = np.tile(predict_score, (thresholds_num, 1))
    negative_index = np.where(predict_score_matrix < thresholds.T)
    positive_index = np.where(predict_score_matrix >= thresholds.T)
    predict_score_matrix[negative_index] = 0
    predict_score_matrix[positive_index] = 1
    TP = predict_score_matrix.dot(real_score.T)
    FP = predict_score_matrix.sum(axis=1) - TP
    FN = real_score.sum() - TP
    TN = len(real_score.T) - TP - FP - FN

    fpr = FP / (FP + TN)
    tpr = TP / (TP + FN)
    ROC_dot_matrix = np.mat(sorted(np.column_stack((fpr, tpr)).tolist())).T
    ROC_dot_matrix.T[0] = [0, 0]
    ROC_dot_matrix = np.c_[ROC_dot_matrix, [1, 1]]
    x_ROC = ROC_dot_matrix[0].T
    y_ROC = ROC_dot_matrix[1].T
    auc = 0.5 * (x_ROC[1:] - x_ROC[:-1]).T * (y_ROC[:-1] + y_ROC[1:])

    recall_list = tpr
    precision_list = TP / (TP + FP)
    PR_dot_matrix = np.mat(sorted(np.column_stack((recall_list, precision_list)).tolist())).T
    PR_dot_matrix.T[0] = [0, 1]
    PR_dot_matrix = np.c_[PR_dot_matrix, [1, 0]]
    x_PR = PR_dot_matrix[0].T
    y_PR = PR_dot_matrix[1].T
    aupr = 0.5 * (x_PR[1:] - x_PR[:-1]).T * (y_PR[:-1] + y_PR[1:])

    f1_score_list = 2 * TP / (len(real_score.T) + TP - TN)
    accuracy_list = (TP + TN) / len(real_score.T)
    specificity_list = TN / (TN + FP)

    max_index = np.argmax(f1_score_list)
    f1_score = f1_score_list[max_index]
    accuracy = accuracy_list[max_index]
    specificity = specificity_list[max_index]
    recall = recall_list[max_index]
    precision = precision_list[max_index]
    return [aupr[0, 0], auc[0, 0], f1_score, accuracy, recall, specificity, precision]


def cv_model_evaluate(interaction_matrix, predict_matrix, train_matrix):
    test_index = np.where(train_matrix == 0)
    real_score = interaction_matrix[test_index]
    predict_score = predict_matrix[test_index]
    return get_metrics(real_score, predict_score)


def constructAdjNet(drug_dis_matrix):
    drug_matrix = np.matrix(np.zeros((drug_dis_matrix.shape[0], drug_dis_matrix.shape[0]), dtype=np.int8))
    dis_matrix = np.matrix(np.zeros((drug_dis_matrix.shape[1], drug_dis_matrix.shape[1]), dtype=np.int8))

    mat1 = np.hstack((drug_matrix, drug_dis_matrix))
    mat2 = np.hstack((drug_dis_matrix.T, dis_matrix))
    adj = np.vstack((mat1, mat2))
    # adj =  adj + sp.eye(adj.shape[0])
    return adj


def weight_variable_glorot(input_dim, output_dim, name=""):
    # 初始化
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = tf.random_uniform(
        [input_dim, output_dim],
        minval=-init_range,
        maxval=init_range,
        dtype=tf.float32
    )

    return tf.Variable(initial, name=name)


def weight_variable_glorot2(input_dim, name=""):
    # 初始化
    init_range = np.sqrt(3.0 / (input_dim * 2))
    initial = tf.random_uniform(
        [input_dim, input_dim],
        minval=-init_range,
        maxval=init_range,
        dtype=tf.float32
    ) + tf.eye(input_dim)
    # initial = tf.eye(input_dim)
    # + tf.eye(input_dim)
    return tf.Variable(initial, name=name)


def dropout_sparse(x, keep_prob, num_nonzero_elems):
    noise_shape = [num_nonzero_elems]
    # keep_prob设置神经元被选中的概率
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    # tf.cast 数据类型转换
    # tf.floor向下取整,ceil向上取整
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1. / keep_prob)


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    # np.vstack 垂直堆叠数组123，456
    # 堆叠成123
    # 456
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def preprocess_graph(adj):
    # 图的预处理，拉普拉斯正则化
    # coo 是一种矩阵格式
    adj_ = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    # adj_nomalized = degree_mat_inv_sqrt.dot(adj_).tocoo()
    adj_nomalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_to_tuple(adj_nomalized)


class GraphConvolution():
    """Basic graph convolution layer for undirected graph without edge labels."""

    def __init__(self, input_dim, output_dim, adj, name, dropout=0., act=tf.nn.relu):
        self.name = name
        self.vars = {}
        self.issparse = False
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(input_dim, output_dim, name='weights')

            # tf.summary.histogram(self.name + '/weights', self.vars['weights'])
        self.dropout = dropout
        self.adj = adj
        self.act = act

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            x = inputs
            x = tf.nn.dropout(x, 1 - self.dropout)
            x = tf.matmul(x, self.vars['weights'])
            x = tf.sparse_tensor_dense_matmul(self.adj, x)

            outputs = self.act(x)
            # tf.add_to_collection(self.name+'w1',self.vars['weights'])
        return outputs


class GraphConvolutionSparse():
    """Graph convolution layer for sparse inputs."""

    def __init__(self, input_dim, output_dim, adj, features_nonzero, name, dropout=0., act=tf.nn.relu):
        self.name = name
        self.vars = {}
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(input_dim, output_dim, name='weights')
            # tf.summary.histogram(self.name + '/weights', self.vars['weights'])
        self.dropout = dropout
        self.adj = adj
        self.act = act
        self.issparse = True
        self.features_nonzero = features_nonzero

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            x = inputs
            x = dropout_sparse(x, 1 - self.dropout, self.features_nonzero)
            x = tf.sparse_tensor_dense_matmul(x, self.vars['weights'])
            x = tf.sparse_tensor_dense_matmul(self.adj, x)
            outputs = self.act(x)
            # tf.add_to_collection('w3',self.vars['weights'])
        return outputs


class InnerProductDecoder():
    """Decoder model layer for link prediction."""

    def __init__(self, input_dim, name, dropout=0., act=tf.nn.sigmoid):
        self.name = name
        self.vars = {}
        self.issparse = False
        self.dropout = dropout
        self.act = act
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot2(input_dim, name='weights')
            # tf.summary.histogram(self.name + '/weights', self.vars['weights'])

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            inputs = tf.nn.dropout(inputs, 1 - self.dropout)

            U = inputs[0:469, :]
            V = inputs[469:, :]
            U = tf.matmul(U, self.vars['weights'])
            V = tf.transpose(V)
            x = tf.matmul(U, V)
            x = tf.reshape(x, [-1])
            outputs = self.act(x)
            # tf.add_to_collection('w2',self.vars['weights'])
        return outputs,self.vars['weights']


class GCNModel():

    def __init__(self, placeholders, num_features, features_nonzero, adj_nonzero, name, act=tf.nn.elu):
        self.name = name
        self.inputs = placeholders['features']
        self.input_dim = num_features
        self.features_nonzero = features_nonzero
        self.adj_nonzero = adj_nonzero
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']
        self.adjdp = placeholders['adjdp']
        self.act = act
        # atten = tf.random_uniform(
        #     [3,1],
        #     minval=0,
        #     maxval=1,
        #     dtype=tf.float32
        # )
        # self.att=tf.Variable(atten,'attention')
        self.att = tf.Variable(tf.constant([0.5, 0.33, 0.25]))
        with tf.variable_scope(self.name):
            self.build()

    def build(self):
        self.adj = dropout_sparse(self.adj, 1 - self.adjdp, self.adj_nonzero)
        self.hidden1 = GraphConvolutionSparse(
            name='gcn_sparse_layer',
            input_dim=self.input_dim,
            output_dim=FLAGS.hidden1,
            adj=self.adj,
            features_nonzero=self.features_nonzero,
            dropout=self.dropout,
            act=self.act)(self.inputs)

        self.hidden2 = GraphConvolution(
            name='gcn_dense_layer',
            input_dim=FLAGS.hidden1,
            output_dim=FLAGS.hidden2,
            adj=self.adj,
            dropout=self.dropout,
            act=self.act)(self.hidden1)

        self.emb = GraphConvolution(
            name='gcn_dense_layer2',
            input_dim=FLAGS.hidden2,
            output_dim=FLAGS.hidden3,
            adj=self.adj,
            dropout=self.dropout,
            act=self.act)(self.hidden2)

        self.embeddings = self.hidden1 * self.att[0] + self.hidden2 * self.att[1] + self.emb * self.att[2]

        self.reconstructions,self.relationemb = InnerProductDecoder(
            name='gcn_decoder',
            input_dim=FLAGS.hidden3, act=tf.nn.sigmoid)(self.embeddings)


class Optimizer():
    def __init__(self, model, preds, labels, w, lr, num):


        global_step = tf.Variable(0, trainable=False)
        # self.optimizer = tf.train.AdamOptimizer(learning_rate=
        #     cyclic_learning_rate(global_step=global_step,learning_rate=lr*0.1,
        #                  max_lr=lr, mode='exp_range',gamma=.995))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.0005)
        alpha = 0.25
        gamma = 2

        alpha_t = labels * alpha + (tf.ones_like(labels) - labels) * (1 - alpha)

        p_t = labels * preds + (tf.ones_like(labels) - labels) * (tf.ones_like(labels) - preds) + 1e-7
        focal_loss = - alpha_t * tf.pow((tf.ones_like(labels) - p_t), gamma) * tf.log(p_t)
        self.cost = tf.reduce_mean(focal_loss)

        # self.cost = norm * tf.reduce_mean(
        #     tf.nn.weighted_cross_entropy_with_logits(
        #         logits=preds_sub, targets=labels_sub, pos_weight=1))

        self.opt_op = self.optimizer.minimize(self.cost, global_step=global_step, )
        self.grads_vars = self.optimizer.compute_gradients(self.cost)


def constructXNet(drug_dis_matrix, drug_matrix, dis_matrix):
    mat1 = np.hstack((drug_matrix, drug_dis_matrix))
    mat2 = np.hstack((drug_dis_matrix.T, dis_matrix))
    return np.vstack((mat1, mat2))


def Get_embedding_Matrix(train_drug_dis_matrix, seed, epochs, dp, w, lr, drug_dis_matrix, adjdp, num):
    np.random.seed(seed)
    tf.set_random_seed(seed)
    adj = constructAdjNet(train_drug_dis_matrix)  # 没有sim就用这个吧
    # adj=constructXNet(train_drug_dis_matrix,drug_matrix,dis_matrix)
    adj = sp.csr_matrix(adj)
    num_nodes = adj.shape[0]
    num_edges = adj.sum()

    X = constructAdjNet(train_drug_dis_matrix)
    features = sparse_to_tuple(sp.csr_matrix(X))
    num_features = features[2][1]
    features_nonzero = features[1].shape[0]
    adj_orig = train_drug_dis_matrix.copy()
    adj_orig = sparse_to_tuple(sp.csr_matrix(adj_orig))

    adj_norm = preprocess_graph(adj)
    adj_nonzero = adj_norm[1].shape[0]
    placeholders = {
        'features': tf.sparse_placeholder(tf.float32),
        'adj': tf.sparse_placeholder(tf.float32),
        'adj_orig': tf.sparse_placeholder(tf.float32),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'adjdp': tf.placeholder_with_default(0., shape=())
    }
    model = GCNModel(placeholders, num_features, features_nonzero, adj_nonzero, name='yeast_gcn')
    with tf.name_scope('optimizer'):
        opt = Optimizer(
            preds=model.reconstructions,
            labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'], validate_indices=False), [-1]),
            model=model,
            w=w, lr=lr, num=num)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))  #
    sess.run(tf.global_variables_initializer())

    for epoch in range(10000):  ####
        feed_dict = dict()
        feed_dict.update({placeholders['features']: features})
        feed_dict.update({placeholders['adj']: adj_norm})
        feed_dict.update({placeholders['adj_orig']: adj_orig})
        feed_dict.update({placeholders['dropout']: dp})
        feed_dict.update({placeholders['adjdp']: adjdp})
        _, avg_cost = sess.run([opt.opt_op, opt.cost], feed_dict=feed_dict)

        if epoch % 100 == 0:
            feed_dict.update({placeholders['dropout']: 0})
            feed_dict.update({placeholders['adjdp']: 0})
            res = sess.run(model.reconstructions, feed_dict=feed_dict)  #
            metric_tmp = roc_auc_score(drug_dis_matrix.flatten(), res)  #
            print("Epoch:", '%04d' % (epoch + 1),
                  "train_loss=", "{:.5f}".format(avg_cost),
                  "score=")
            print(metric_tmp)
    print('Optimization Finished!')
    feed_dict.update({placeholders['dropout']: 0})
    feed_dict.update({placeholders['adjdp']: 0})
    res = sess.run(model.reconstructions, feed_dict=feed_dict)
    relationemb=sess.run(model.relationemb,feed_dict=feed_dict)
    print(111111111111111)
    print(relationemb.shape)
    np.save('lgcn_relation.npy',relationemb)
    embeddingss = sess.run(model.embeddings, feed_dict=feed_dict)
    print(embeddingss.shape)
    embeddingsss = embeddingss
    np.save('lgcn_em.npy', embeddingsss)
    print(sess.run(model.att, feed_dict=feed_dict))
    sess.close()
    print(res.shape)
    return res


def cross_validation_experiment(drug_dis_matrix, drug_matrix, dis_matrix, seed, epochs, dp, w, lr, adjdp, g):
    index_matrix = np.mat(np.where(np.abs(drug_dis_matrix) == 1))
    association_nam = index_matrix.shape[1]
    random_index = index_matrix.T.tolist()
    random.seed(seed)
    random.shuffle(random_index)
    k_folds = 5
    CV_size = int(association_nam / k_folds)
    temp = np.array(random_index[:association_nam - association_nam % k_folds]).reshape(k_folds, CV_size, -1).tolist()
    temp[k_folds - 1] = temp[k_folds - 1] + random_index[association_nam - association_nam % k_folds:]
    random_index = temp
    metric = np.zeros((1, 7))
    print("seed=%d, evaluating drug-disease...." % (seed))
    for k in range(k_folds):
        print("------this is %dth cross validation------" % (k + 1))
        train_matrix = np.matrix(drug_dis_matrix, copy=True)
        train_matrix[tuple(np.array(random_index[k]).T)] = 0
        drug_len = drug_dis_matrix.shape[0]
        dis_len = drug_dis_matrix.shape[1]

        drug_disease_res = Get_embedding_Matrix(train_matrix, drug_matrix, dis_matrix, seed, epochs, dp, w, lr,
                                                drug_dis_matrix, adjdp)
        predict_y_proba = drug_disease_res.reshape(drug_len, dis_len)

        metric_tmp = cv_model_evaluate(drug_dis_matrix, predict_y_proba, train_matrix)
        print(metric_tmp)

        metric += metric_tmp

        del train_matrix

        gc.collect()

    print(metric / k_folds)

    metric = np.array(metric / k_folds)

    return metric
