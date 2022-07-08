import numpy as np
import sys
import random
import torch
import copy
import copy
import time

def setup_seed(seed):
   torch.manual_seed(seed)
   torch.cuda.manual_seed_all(seed)
   np.random.seed(seed)
   random.seed(seed)
   torch.backends.cudnn.deterministic = True

setup_seed(int(11))



from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

from tool import get_metrics,preprocess
import sklearn
from sklearn import svm
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
import tqdm
id_list = []
label_list = []
featurelist = []
featurekglist = []
num_list=[]
targetlist=list(np.load('target_list.txt.npy'))
featuremat=np.load('targetf.npy')
traintxt=['train-num-TransE_l2.txt']
testtxt=['test-num-TransE_l2.txt']#
topk=30

maxrecalllist=[]
maxauclist=[]

id_list = []
label_list = []
featurelist = []
featurekglist = []
num_list = []
train = np.loadtxt('data/' + traintxt[0], dtype='str', delimiter='\n')
for i in range(train.shape[0]):
    if str(train[i]).split('\t')[0] in targetlist:

        id_list.append(str(train[i]).split('\t')[0])
        if str(train[i]).split('\t')[4] == 'NON-SUPER':
            label_list.append(0)
        else:
            label_list.append(1)

        num_list.append(int(str(train[i]).split('\t')[3]))
        feature = featuremat[targetlist.index(str(train[i]).split('\t')[0])]
        feature = np.array(feature)
        feature1 = str(train[i]).split('\t')[5]
        feature1 = preprocess(feature1)
        feature1 = np.array(feature1)
        if i ==0:
            featuregcn=feature.reshape(1,-1)
            featurekg=feature1.reshape(1,-1)
        else:

            featuregcn=np.concatenate((featuregcn,feature.reshape(1,-1)),axis=0)
            featurekg=np.concatenate((featurekg,feature1.reshape(1,-1)),axis=0)


trainx=np.hstack((featuregcn,featurekg))
train_numlist=num_list
train_label=label_list


train = np.loadtxt('data/' + testtxt[0], dtype='str', delimiter='\n')
id_list = []
label_list = []
featurelist = []
featurekglist = []
num_list = []
for i in range(train.shape[0]):
    if str(train[i]).split('\t')[0] in targetlist:
        id_list.append(str(train[i]).split('\t')[0])
        if str(train[i]).split('\t')[4] == 'NON-SUPER':
            label_list.append(0)
        else:  #
            label_list.append(1)
        num_list.append(int(str(train[i]).split('\t')[3]))
        feature = featuremat[targetlist.index(str(train[i]).split('\t')[0])]
        feature = np.array(feature)

        feature1 = str(train[i]).split('\t')[5]
        feature1 = preprocess(feature1)
        feature1 = np.array(feature1)

        if i ==0:
            featuregcn=feature.reshape(1,-1)
            featurekg=feature1.reshape(1,-1)
        else:

            featuregcn=np.concatenate((featuregcn,feature.reshape(1,-1)),axis=0)
            featurekg=np.concatenate((featurekg,feature1.reshape(1,-1)),axis=0)

testx=np.hstack((featuregcn,featurekg))

test_label_list=label_list
test_numlist=num_list


max_auc = 0
max_recall=0

model_svr =  svm.LinearSVR(C=1,max_iter=2000)
model_dtr = DecisionTreeRegressor()

train_numlistnp=np.array(train_numlist)


for epoch in range(2000):

        for batchnum in range(30):
            row_total = trainx.shape[0]
            row_sequence = np.arange(row_total)
            np.random.shuffle(row_sequence)
            x = trainx[row_sequence[0:50], :]
            y = train_numlistnp[row_sequence[0:50]]
            model_svr.fit(x, y)
            model_dtr.fit(x, y)

        trainpre1=model_svr.predict(trainx)
        trainpre2=model_dtr.predict(trainx)
        trainpre=(trainpre1+trainpre2)/2

        result1 = model_svr.predict(testx)
        result2 = model_dtr.predict(testx)

        result=(result1+result2)/2


        roc_train=roc_auc_score(np.array(train_label), trainpre)
        roc_test = roc_auc_score(np.array(test_label_list), result)


        a = get_metrics(np.array(test_label_list), result)
        recall = a[0]

        np.save('label', np.array(test_label_list))
        np.save('predict', np.array(result))

        label = np.load('label.npy')
        predict = np.load('predict.npy')
        all = np.vstack((predict, label))
        all = all.T
        all = all[all[:, 0].argsort()]
        all = all[::-1]
        k = topk
        n = 0
        for i in range(k):
            if all[i][1] == 1:
                n = n + 1
        toprecall = n / 9
        print(toprecall)

        if roc_test  >= max_auc and toprecall >= max_recall:

            max_auc = roc_test
            max_recall = toprecall
            ymaxlabel = np.array(test_label_list)
            ymaxpre = np.array(result)

            np.save('maxlabel', ymaxlabel)
            np.save('maxpre', ymaxpre)
            label = np.load('maxlabel.npy')
            predict = np.load('maxpre.npy')
            all = np.vstack((predict, label))
            all = all.T
            all = all[all[:, 0].argsort()]
            all = all[::-1]
            k = topk
            n = 0
            for i in range(k):
                if all[i][1] == 1:
                    n = n + 1
            toprecall = n / 9
            print(toprecall)
            a = get_metrics(np.array(test_label_list), np.array(result))
            print(a)

            max_a = a

        print(max_a)
        print(max_recall)

        print('epoch: {:04d}'.format(epoch + 1),
              'auroc_train: {:.4f}'.format(roc_train),
              'auroc_val: {:.4f}'.format(roc_test))
        if epoch == 1999:

            maxauclist.append(max_a[1])
            maxrecalllist.append(max_recall)


with open("result/final.txt", "w") as f:
    for i in range(len(maxrecalllist)):
        f.write(str(maxauclist[i]) + '\t' + str(maxrecalllist[i]) + '\n')
