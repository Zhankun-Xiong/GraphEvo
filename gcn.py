import csv
import numpy as np
from gcntool import Get_embedding_Matrix
import os
import random


interaction=np.loadtxt('data/target-disease.txt',dtype=str)
print(interaction.shape)
#
target_list=[]
for i in range(interaction.shape[0]):
   target_list.append(interaction[i][0])

target_list=list(set(target_list))
print(len(target_list))
np.save('target_list.txt',target_list)



disease_list=[]
for i in range(interaction.shape[0]):
    disease_list.append(interaction[i][2])

disease_list=list(set(disease_list))
np.save('disease_list.txt',disease_list)
print(len(disease_list))


target_list=list(np.load('target_list.txt.npy'))
disease_list=list(np.load('disease_list.txt.npy'))
print(len(disease_list))
mat=np.zeros((len(target_list),len(disease_list)))
for i in range(interaction.shape[0]):
    mat[target_list.index(interaction[i][0])][disease_list.index(interaction[i][2])]=1

print(mat)
print(np.sum(mat))

random.shuffle(interaction)
interaction=interaction[0:(4*interaction.shape[0])//5]
matt=np.zeros((len(target_list),len(disease_list)))
for i in range(interaction.shape[0]):
    matt[target_list.index(interaction[i][0])][disease_list.index(interaction[i][2])]=1



num = 1
epochs = 100
w = 1.4
adjdp = 0.6
dp = 0.3
lr = 0.001
gs = 2
simws = 1
yuzhi = [0.08, 0.05, 0.02, 0.01]
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

result = np.zeros((1, 7), float)
average_result = np.zeros((1, 7), float)
circle_time = 1
for i in range(circle_time):
    drug_disease_res = Get_embedding_Matrix(mat, 0, epochs, dp, w, lr, mat, adjdp,
                                            num)
    drug_len = 469
    dis_len = 2482
    predict = drug_disease_res.reshape(drug_len, dis_len)
predict = np.array(predict)
print(predict.shape)
np.save('predict.npy', predict)
k=np.load('lgcn_em.npy')
k=k[0:469]
print(k.shape)
np.save('targetf',k)
print(k.shape)