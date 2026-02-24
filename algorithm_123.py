#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @software: PyCharm
# @project : 程序2022
# @File    : 算法_123.py
# @Author  : TLX
# @Time    : 2023/2/4 16:00


import pandas as pd
import numpy as np
import math
import os
from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import classification_report,roc_curve, auc,accuracy_score
import matplotlib.pyplot as plt
from IPython.core.pylabtools import figsize
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore') # 不输出 warning

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data'))
RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../results'))
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

def read_file(filename):
    f = open(filename,"r")   # 设置文件对象
    data = f.readlines()     # 直接将文件中按行读到list
    f.close()                # 关闭文件
    return data

def Model_RF(model, skf, X, y):
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    def max_auc(model):
        y_score = model.fit(X_train, y_train).predict_proba(X_test)
        fpr, tpr, thresholds = roc_curve(y_test, y_score[:, 1])
        roc_auc = auc(fpr, tpr)
        return roc_auc

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        max_est, max_score = 1, 0
        for i in np.arange(5, 100, 10):
            model.set_params(n_estimators=i)
            roc_auc = max_auc(model)
            if roc_auc > max_score:
                max_score = roc_auc
                max_est = i
        model.set_params(n_estimators=max_est)

        max_de, max_score = 1, 0
        for i in np.arange(1, 20, 1):
            model.set_params(max_depth=i)
            roc_auc = max_auc(model)
            if roc_auc > max_score:
                max_score = roc_auc
                max_de = i
        model.set_params(max_depth=max_de)

        max_leaf, max_score = 1, 0
        for i in np.arange(1, 10, 1):
            model.set_params(min_samples_leaf=i)
            roc_auc = max_auc(model)
            if roc_auc > max_score:
                max_score = roc_auc
                max_leaf = i
        model.set_params(min_samples_leaf=max_leaf)

        max_rand, max_score = 0, 0
        for i in np.arange(0, 200, 10):
            model.set_params(random_state=i)
            roc_auc = max_auc(model)
            if roc_auc > max_score:
                max_score = roc_auc
                max_rand = i
        model.set_params(random_state=max_rand)

        y_score = model.fit(X_train, y_train).predict_proba(X_test)
        fpr, tpr, thresholds = roc_curve(y_test, y_score[:, 1])
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0

    return tprs, mean_fpr

def Model_SVM(model, skf, X, y):
    # model, skf, X, y = SVC(),skf,X2,y2
    def max_auc(model):
        y_score = model.fit(X_train, y_train).decision_function(X_test)
        fpr, tpr, thresholds = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        return roc_auc

    tprs = []
    mean_fpr = np.linspace(0, 1, 100)
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        max_ker, max_score = 'linear', 0
        ker_set = ['linear', 'poly', 'rbf', 'sigmoid']
        for ker in ker_set:
            model.set_params(kernel=ker)
            roc_auc = max_auc(model)
            if roc_auc > max_score:
                max_score = roc_auc
                max_ker = ker
        model.set_params(kernel=max_ker)

        if max_ker == 'poly':
            max_deg, max_score = 1, 0
            for deg in np.arange(1, 6, 1):
                model.set_params(degree=deg)
                roc_auc = max_auc(model)
                if roc_auc > max_score:
                    max_score = roc_auc
                    max_deg = deg
            model.set_params(degree=max_deg)

        if max_ker != 'linear':
            max_coef, max_score = 0.05, 0
            for coef in np.arange(0.05, 1.05, 0.05):
                model.set_params(coef0=coef)
                roc_auc = max_auc(model)
                if roc_auc > max_score:
                    max_score = roc_auc
                    max_coef = coef
            model.set_params(coef0=max_coef)

        max_c, max_score = 1, 0
        c_set = [0.05, 0.1, 0.5, 1, 10, 100]
        for c in c_set:
            model.set_params(C=c)
            roc_auc = max_auc(model)
            if roc_auc > max_score:
                max_score = roc_auc
                max_c = c
        model.set_params(C=max_c)

        y_score = model.fit(X_train, y_train).decision_function(X_test)
        y_score = [1 / (1 + math.exp(-s)) for s in y_score]
        fpr, tpr, thresholds = roc_curve(y_test, y_score)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0

    return tprs, mean_fpr

def Model_ElsNet(model, skf, X, y):
    # model, skf, X, y = ElasticNet(), skf, X1, y1
    def max_auc(model):
        y_score = model.fit(X_train, y_train)._decision_function(X_test)
        fpr, tpr, thresholds = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        return roc_auc

    tprs = []
    mean_fpr = np.linspace(0, 1, 100)
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        max_alpha, max_score = 0.1, 0
        for alp in np.logspace(-5, -3, 10):
            model.set_params(alpha=alp)
            roc_auc = max_auc(model)
            if roc_auc > max_score:
                max_score = roc_auc
                max_alp = alp
        model.set_params(alpha=max_alp)

        max_l1, max_score = 0.1, 0
        for l1 in np.linspace(0, 1, 11):
            model.set_params(l1_ratio=l1)
            roc_auc = max_auc(model)
            if roc_auc > max_score:
                max_score = roc_auc
                max_l1 = l1
        model.set_params(l1_ratio=max_l1)

        y_score = model.fit(X_train, y_train)._decision_function(X_test)
        y_score = [1/(1+math.exp(-s)) for s in y_score]
        fpr, tpr, thresholds = roc_curve(y_test, y_score)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0

    return tprs, mean_fpr

def Model_Logistic(model, skf, X, y):
    # model, skf, X, y = LogisticRegression(), skf, X1, y1
    def max_auc(model):
        y_score = model.fit(X_train, y_train).decision_function(X_test)
        fpr, tpr, thresholds = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        return roc_auc

    tprs = []
    mean_fpr = np.linspace(0, 1, 100)
    model.set_params(solver='saga', penalty='elasticnet')
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        l1, max_score = 0, 0
        for l1 in np.arange(0,1.1,0.1):
            model.set_params(l1_ratio = l1)
            roc_auc = max_auc(model)
            if roc_auc > max_score:
                max_score = roc_auc
                max_l1 = l1
        model.set_params(l1_ratio = max_l1)

        max_c ,max_score = 1, 0
        C_set = [0.01,0.1,0.5,1,10,50,100,1000]
        for c in C_set:
            model.set_params(C = c)
            roc_auc = max_auc(model)
            if roc_auc > max_score:
                max_score = roc_auc
                max_c = c
        model.set_params(C = max_c)

        y_score = model.fit(X_train, y_train).decision_function(X_test)
        fpr, tpr, thresholds = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        # interp:插值 把结果添加到tprs列表中
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        # 计算auc
        roc_auc = auc(fpr, tpr)

        return tprs, mean_fpr

def Find_X_y(df, class_type, P_sample, N_sample):
    # df, class_type, P_sample, N_sample = Vec_1, 'Effect_type', 'Synergistic', ['Antagonistic','Additive','Unclear','No_type']
    # df, class_type, P_sample, N_sample = Vec_1, 'Efficacy', 'Efficacious', 'Non-efficacious'

    df_syn = df.loc[df[class_type].isin([P_sample])]
    df_syn = df.loc[df['Effect_type'].isin(['Synergistic'])]
    df_ant = df.loc[df['Effect_type'].isin(['Antagonistic'])]
    df_nonsyn1 = df.loc[df[class_type].isin(N_sample)]
    if len(df_syn) < len(df_nonsyn1):
        df_nonsyn1 = df_nonsyn1.sample(n=len(df_syn)-5)

    df_nonsyn = pd.concat([df_ant,df_nonsyn1])
    df_nonsyn.loc[:, class_type] = 'not_syn'
    dff = pd.concat([df_nonsyn,df_syn])
    cols = []
    for i in dff.columns:
        if (dff[i].dtype == "float64") or (dff[i].dtype == 'int64'):
            cols.append(i)

    return dff[cols].copy(), pd.Categorical(dff[class_type]).codes

def Find_X_y_random(df, df_non,class_type ):
    "负样本为 notsyn + 随机选取"
    # df, df_non,class_type = Vec_4, Vec_4_nonsyn, 'Effect_type'
    del_dcom_inc = read_file(os.path.join(DATA_DIR, "del_dcoms.txt"))
    del_dcom_no = read_file(os.path.join(DATA_DIR, 'del_dcoms_no.txt'))
    del_dcom_dec = read_file(os.path.join(DATA_DIR, 'del_dcoms_dec.txt'))
    # del_dcom_inc.extend(del_dcom_no)
    # del_dcom = [d.replace('\n','') for d in del_dcom_inc]
    # ls_index = [l for l in df_non.index if l not in del_dcom]
    # df_non = df_non[df_non.index.isin(ls_index)]
    del_dcom_no = [d.replace('\n', '') for d in del_dcom_no]
    df_non = df_non[df_non.index.isin(del_dcom_no)]

    df_syn = df.loc[df[class_type] == 'Synergistic']
    df_nonsyn = df_non.sample(n = len(df_syn))
    df_sample = pd.concat([df_nonsyn,df_syn])

    cols = []
    for i in df_sample.columns:
        if (df_sample[i].dtype == "float64") or (df_sample[i].dtype == 'int64'):
            cols.append(i)

    return df_sample[cols].copy(), pd.Categorical(df_sample[class_type]).codes

'''vec3 只有节点特征; Vec4 节点特征 + D-D 边特征'''
def NodeVec_MetaVec(file_node,file_edge):
    Vec_3_node = read_file(file_node)
    Vec_3_node = Vec_3_node[1:]  # 去掉首行
    D_vec = pd.DataFrame()
    for i in range(len(Vec_3_node)):
        node = Vec_3_node[i].replace('\n','').split(' ')
        if 'DB' in node[0]:
            # print(i)
            D_vec[node[0]] = node[1:]
            D_vec[node[0]] = D_vec[node[0]].apply(pd.to_numeric)
            # D_vec[node[0]] = preprocessing.scale(D_vec[node[0]])
    D_vec = pd.DataFrame(D_vec.values.T, index=D_vec.columns, columns=D_vec.index) # 数据旋转
    # D_vec = D_vec.apply(pd.to_numeric) # 数据保留六位小数

    Vec_3_edge = read_file(file_edge)
    Vec_3_edge = Vec_3_edge[1:]
    DD_vec = []
    for j in range(len(Vec_3_edge)):
        edge = Vec_3_edge[j].replace('\n','').split(' ')
        if edge[0] == 'D-D':
            DD_vec = [round(float(e), 5) for e in edge[1:]]
            DD_vec = preprocessing.scale(DD_vec)
    return D_vec, DD_vec

# 构造所有的vec_3，Vec_4
def Vec3(Vec_1,D_vec):
    dcom_list = Vec_1.index
    Vec_3 = pd.DataFrame(columns=list(range(2 * len(D_vec.columns))) + ['Efficacy', 'Effect_type'])
    for dc in dcom_list:
        drugs = dc.split(',')
        if (drugs[0] in D_vec.index) and (drugs[1] in D_vec.index):
            Vec_3.loc[dc] = list(D_vec.loc[drugs[0]]) + list(D_vec.loc[drugs[1]]) + [Vec_1.loc[dc, 'Efficacy'], Vec_1.loc[dc, 'Effect_type']]
    return Vec_3

def Vec4_1(Vec_1,D_vec, DD_vec):
    "(D,D,D-D)"
    dcom_list = Vec_1.index
    Vec_4 = pd.DataFrame(columns=list(range(3 * len(D_vec.columns)))+ ['Efficacy', 'Effect_type'])
    for dc in dcom_list:
        drugs = dc.split(',')
        if (drugs[0] in D_vec.index) and (drugs[1] in D_vec.index):
            Vec_4.loc[dc] = list(D_vec.loc[drugs[0]]) + list(D_vec.loc[drugs[1]]) + DD_vec + [Vec_1.loc[dc, 'Efficacy'], Vec_1.loc[dc, 'Effect_type']]
    return Vec_4

def Vec4_2(Vec_1,D_vec, DD_vec):
    "(D + D + D-D)"
    dcom_list = Vec_1.index
    Vec_4 = pd.DataFrame(columns=list(range(len(D_vec.columns))))
    for dc in dcom_list:
        drugs = dc.split(',')
        if (drugs[0] in D_vec.index) and (drugs[1] in D_vec.index):
            Vec_4.loc[dc] = D_vec.loc[drugs[0]] + D_vec.loc[drugs[1]] + np.array(DD_vec)
            Vec_4.loc[dc,['Efficacy', 'Effect_type']] = [Vec_1.loc[dc, 'Efficacy'], Vec_1.loc[dc, 'Effect_type']]
    return Vec_4

def Vec4_3(Vec_1,D_vec, DD_vec):
    "((D + D) * D-D)"
    dcom_list = Vec_1.index
    Vec_4 = pd.DataFrame(columns=list(range(len(D_vec.columns))) +list(Vec_1.columns))
    for dc in dcom_list:
        drugs = dc.split(',')
        if (drugs[0] in D_vec.index) and (drugs[1] in D_vec.index):
            cos_dis = cosine_similarity([D_vec.loc[drugs[0]]], [D_vec.loc[drugs[1]]])
            Vec_4.loc[dc] = pd.concat([(D_vec.loc[drugs[0]] + D_vec.loc[drugs[1]]) * np.array(DD_vec), Vec_1.loc[dc]])
            Vec_4.loc[dc,'cos_dis'] = round(cos_dis[0][0],6)

    return Vec_4

def Vec3_syn(Vec_3):
    #协同药物集合
    drugs_syn = []
    Vec_3_syn = Vec_3[Vec_3['Effect_type'] == 'Synergistic']
    for dc in Vec_3_syn.index:
        drugs = dc.split(',')
        if Vec_3.loc[dc, 'Effect_type'] == 'Synergistic':
            if drugs[0] not in drugs_syn:
                drugs_syn.append(drugs[0])
            if drugs[1] not in drugs_syn:
                drugs_syn.append(drugs[1])
    return drugs_syn

def Vec3_notsyn(drugs_syn, D_vec):
    # 协同药物构造的非协同组合
    Vec_3_nonsyn = pd.DataFrame(columns=list(range(2 * len(D_vec.columns))) + ['Efficacy', 'Effect_type'])

    for i in range(len(drugs_syn)):
        for j in range(len(drugs_syn)):
            dc = ','.join([drugs_syn[i],drugs_syn[j]])
            if i<j :
                Vec_3_nonsyn.loc[dc] = list(D_vec.loc[drugs_syn[i]]) + list(D_vec.loc[drugs_syn[j]]) + ['non_eff','non_syn']
        # print(i)
    # print("Finish")
    return Vec_3_nonsyn

def Vec4_1_notsyn(drugs_syn, D_vec, DD_vec):
    "(D,D,D-D)"
    Vec_4_nonsyn = pd.DataFrame(columns=list(range(3 * len(D_vec.columns)))+ ['Efficacy', 'Effect_type'])
    for i in range(len(drugs_syn)):
        for j in range(len(drugs_syn)):
            dc = ','.join([drugs_syn[i],drugs_syn[j]])
            if i<j :
                Vec_4_nonsyn.loc[dc] = list(D_vec.loc[drugs_syn[i]]) + list(D_vec.loc[drugs_syn[j]]) + DD_vec + ['non_eff','non_syn']
        # print(i)
    # print("Finish")
    return Vec_4_nonsyn

def Vec4_2_notsyn(drugs_syn, D_vec, DD_vec):
    "(D + D + D-D)"
    Vec_4_nonsyn = pd.DataFrame(columns=list(range(len(D_vec.columns))))
    for i in range(len(drugs_syn)):
        for j in range(len(drugs_syn)):
            dc = ','.join([drugs_syn[i],drugs_syn[j]])
            if i<j :
                Vec_4_nonsyn.loc[dc] = D_vec.loc[drugs_syn[i]] + D_vec.loc[drugs_syn[j]] + np.array(DD_vec)
                Vec_4_nonsyn.loc[dc,['Efficacy', 'Effect_type']] = ['non_eff','non_syn']
        # print(i)
    # print("Finish")
    return Vec_4_nonsyn

def Vec4_3_notsyn(Vec_1_nonsyn, drugs_syn, D_vec, DD_vec):
    "(D + D + D-D)"
    dcom_list = Vec_1_nonsyn.index
    Vec_4_nonsyn = pd.DataFrame(columns=list(range(len(D_vec.columns))) + list(Vec_1_nonsyn.columns))
    for dc in dcom_list:
        drugs = dc.split(',')
        if (drugs[0] in drugs_syn) and (drugs[1] in drugs_syn):
            cos_dis = cosine_similarity([D_vec.loc[drugs[0]]],[D_vec.loc[drugs[1]]])
            Vec_4_nonsyn.loc[dc] = pd.concat([(D_vec.loc[drugs[0]] + D_vec.loc[drugs[1]]) * np.array(DD_vec), Vec_1_nonsyn.loc[dc]])
            Vec_4_nonsyn.loc[dc,'cos_dis'] = round(cos_dis[0][0],6)

    return Vec_4_nonsyn


Vec_1 = pd.read_csv(os.path.join(RESULTS_DIR, 'Vec_1.csv'), header=0, index_col=0)
Vec_1_nonsyn = pd.read_csv(os.path.join(RESULTS_DIR, 'Vec_1_nonsyn.csv'), header=0, index_col=0)
Vec_2 = pd.read_csv(os.path.join(RESULTS_DIR, 'Vec_2.csv'), header=0, index_col=0)
Vec_2_nonsyn = pd.read_csv(os.path.join(RESULTS_DIR, 'Vec_2_nonsyn.csv'), header=0, index_col=0)

# Note: node_vectors_all_l5w3d100.txt should be generated by HIN2Vec
file_node = os.path.join(DATA_DIR, 'node_vectors_all_l5w3d100.txt') 
if not os.path.exists(file_node):
    file_node = os.path.join(RESULTS_DIR, 'node_vectors_all_l5w3d100.txt')

file_edge = os.path.join(DATA_DIR, 'metapath_vectors_all_l5w3d100.txt')
D_vec, DD_vec = NodeVec_MetaVec(file_node, file_edge)
Vec_3 = Vec3(Vec_1, D_vec)
drugs_syn = Vec3_syn(Vec_3)
Vec_3_nonsyn = Vec3_notsyn(drugs_syn, D_vec)
Vec_4 = Vec4_3(Vec_1, D_vec, DD_vec)
Vec_4_nonsyn = Vec4_3_notsyn(Vec_1_nonsyn, drugs_syn, D_vec, DD_vec)


X1, y1 = Find_X_y(Vec_1, 'Effect_type', 'Synergistic', ['Antagonistic','Additive','Unclear','No_type'])
X2, y2 = Find_X_y(Vec_2, 'Effect_type', 'Synergistic', ['Antagonistic','Additive','Unclear','No_type'])
X3, y3 = Find_X_y(Vec_3, 'Effect_type', 'Synergistic', ['Antagonistic','Additive','Unclear','No_type'])
X4, y4 = Find_X_y(Vec_4, 'Effect_type', 'Synergistic', ['Antagonistic','Additive','Unclear','No_type'])

X1, y1 = Find_X_y(Vec_1, 'Effect_type', 'Synergistic', ['Antagonistic'])
X2, y2 = Find_X_y(Vec_2, 'Effect_type', 'Synergistic', ['Antagonistic'])
X3, y3 = Find_X_y(Vec_3, 'Effect_type', 'Synergistic', ['Antagonistic'])
X4, y4 = Find_X_y(Vec_4, 'Effect_type', 'Synergistic', ['Antagonistic'])

X1, y1 = Find_X_y(Vec_1, 'Efficacy', 'Efficacious', ['Non-efficacious'])
X2, y2 = Find_X_y(Vec_2, 'Efficacy', 'Efficacious',['Non-efficacious'])
X3, y3 = Find_X_y(Vec_3, 'Efficacy', 'Efficacious',['Non-efficacious'])
X4, y4 = Find_X_y(Vec_4, 'Efficacy', 'Efficacious',['Non-efficacious'])

X1, y1 = Find_X_y_random(Vec_1, Vec_1_nonsyn, 'Effect_type')
X2, y2 = Find_X_y_random(Vec_2, Vec_2_nonsyn, 'Effect_type')
X3, y3 = Find_X_y_random(Vec_3, Vec_3_nonsyn, 'Effect_type')
X4, y4 = Find_X_y_random(Vec_4, Vec_4_nonsyn, 'Effect_type')


print('相似性计算')
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=200)

# RF
rfc1, rfc2,rfc3,rfc4 = RandomForestClassifier(),RandomForestClassifier(),RandomForestClassifier(),RandomForestClassifier()
tprs_R1,mean_fpr = Model_RF(rfc1,skf,X1,y1)
tprs_R2,mean_fpr = Model_RF(rfc2,skf,X2,y2)
tprs_R3,mean_fpr = Model_RF(rfc3,skf,X3,y3)
tprs_R4,mean_fpr = Model_RF(rfc4,skf,X4,y4)
print('rfc--OK')

# SVM
svc1, svc2, svc3, svc4 = SVC(), SVC(), SVC(), SVC()
tprs_S1,mean_fpr = Model_SVM(svc1,skf,X1,y1)
tprs_S2,mean_fpr = Model_SVM(svc2,skf,X2,y2)
tprs_S3,mean_fpr = Model_SVM(svc3,skf,X3,y3)
tprs_S4,mean_fpr = Model_SVM(svc4,skf,X4,y4)
print('SVC -- OK')

# # Logistic
# Log1,Log2,Log3,Log4 = LogisticRegression(),LogisticRegression(),LogisticRegression(),LogisticRegression()
# tprs_L1,mean_fpr = LogisticRegression(Log1,skf,X1,y1)
# tprs_L2,mean_fpr = LogisticRegression(Log2,skf,X2,y2)
# tprs_L3,mean_fpr = LogisticRegression(Log3,skf,X3,y3)
# tprs_L4,mean_fpr = LogisticRegression(Log4,skf,X4,y4)
# print('LR -- OK')

# Elastic Net
enet1,enet2,enet3,enet4 = ElasticNet(max_iter=5000, tol=0.01),ElasticNet(max_iter=5000, tol=0.01),ElasticNet(max_iter=5000, tol=0.01),ElasticNet(max_iter=5000, tol=0.01)
tprs_E1,mean_fpr = Model_ElsNet(enet1,skf,X1,y1)
tprs_E2,mean_fpr = Model_ElsNet(enet2,skf,X2,y2)
tprs_E3,mean_fpr = Model_ElsNet(enet3,skf,X3,y3)
tprs_E4,mean_fpr = Model_ElsNet(enet4,skf,X4,y4)
print('ENET -- OK')

# ROC 曲线
def mean_roc(tprs,mean_fpr,lab,col,lst):
    mean_tpr=np.mean(tprs,axis=0)
    mean_tpr[-1]=1.0
    mean_auc=auc(mean_fpr,mean_tpr)#计算平均AUC值
    plt.plot(mean_fpr,mean_tpr,linestyle=lst,color=col,label=lab + ' (%0.4f)'%mean_auc,lw=2,alpha=.8)
    return

mean_roc(tprs_R1,mean_fpr,'RF_SRFVec','b','-')
mean_roc(tprs_R2,mean_fpr,'RF_LSFVec','g','-')
# mean_roc(tprs_R3,mean_fpr,'RF_3','r','-')
mean_roc(tprs_R4,mean_fpr,'RF_HINVec','r','-')
mean_roc(tprs_S1,mean_fpr,'SVM_SRFVec','b','--')
mean_roc(tprs_S2,mean_fpr,'SVM_LSFVec','g','--')
# mean_roc(tprs_S3,mean_fpr,'SVM_3','r','--')
mean_roc(tprs_S4,mean_fpr,'SVM_HINVec','r','--')
# mean_roc(tprs_L1,mean_fpr,'LR_1','b',':')
# mean_roc(tprs_L2,mean_fpr,'LR_2','g',':')
# mean_roc(tprs_L3,mean_fpr,'LR_3','r',':')
# mean_roc(tprs_L4,mean_fpr,'LR_4','y',':')
mean_roc(tprs_E1,mean_fpr,'EN_SRFVec','b',':')
mean_roc(tprs_E2,mean_fpr,'EN_LSFVec','g',':')
# mean_roc(tprs_E3,mean_fpr,'EN_3','r',':')
mean_roc(tprs_E4,mean_fpr,'EN_HINVec','r',':')

plt.plot([0,1],[0,1],linestyle=':',lw=2,color='gray')
plt.xlim([0,1.05])
plt.ylim([0,1.05])
plt.xlabel('False Positive Rate',fontsize=10)
plt.ylabel('True Positive Rate',fontsize=10)
# plt.tick_params(labelsize=23)
plt.title('ROC')
plt.legend(loc='lower right')
plt.show()
print("Over")



