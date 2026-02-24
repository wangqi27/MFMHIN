#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @software: PyCharm
# @project : 程序2022
# @File    : 算法RF_X3.py
# @Author  : TLX
# @Time    : 2022/12/27 19:30


import os
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import classification_report,roc_curve, auc,accuracy_score
import matplotlib.pyplot as plt
from IPython.core.pylabtools import figsize

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data'))
RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../results'))
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

def read_file(filename):
    f = open(filename,"r")   # 设置文件对象
    data = f.readlines()     # 直接将文件中按行读到list里hao de
    f.close()                # 关闭文件
    return data

def Model_RF(model, skf, X, y):
    # model, skf, X, y = rfc,skf,X2,y2
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    # i = 0
    # roc_auc1 = 0
    def max_auc(model):
        y_score = model.fit(X_train, y_train).predict_proba(X_test)
        y_predict = model.predict(X_test)
        fpr, tpr, thresholds = roc_curve(y_test, y_score[:, 1])
        roc_auc = auc(fpr, tpr)
        return roc_auc

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        max_est, max_score = 1, 0
        for i in np.arange(5,100,10):
            model.set_params(n_estimators = i)
            roc_auc = max_auc(model)
            if roc_auc > max_score:
                max_score = roc_auc
                max_est = i
        model.set_params(n_estimators = max_est)

        max_de, max_score = 1, 0
        for i in np.arange(1,20,1):
            model.set_params(max_depth = i)
            roc_auc = max_auc(model)
            if roc_auc > max_score:
                max_score = roc_auc
                max_de = i
        model.set_params(max_depth = max_de)

        max_leaf, max_score = 1, 0
        for i in np.arange(1,10,1):
            model.set_params(min_samples_leaf = i)
            roc_auc = max_auc(model)
            if roc_auc > max_score:
                max_score = roc_auc
                max_leaf = i
        model.set_params(min_samples_leaf = max_leaf)

        max_rand, max_score = 0, 0
        for i in np.arange(0,200,10):
            model.set_params(random_state = i)
            roc_auc = max_auc(model)
            if roc_auc > max_score:
                max_score = roc_auc
                max_rand = i
        model.set_params(random_state = max_rand)

        y_score = model.fit(X_train, y_train).predict_proba(X_test)
        y_predict = model.predict(X_test)
        fpr, tpr, thresholds = roc_curve(y_test, y_score[:, 1])
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)

    return tprs, mean_fpr

def Model_ElsNet(enet, skf, X, y):
    tprs = []
    # aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        alphas = np.logspace(-5, -3, 10)
        l1_list = np.linspace(0, 1, 11)
        train_error = []
        alp_l1 = []
        for alpha in alphas:
            for l1 in l1_list:
                enet.set_params(alpha=alpha)
                enet.set_params(l1_ratio = l1)
                enet.fit(X_train, y_train)
                train_error.append(enet.score(X_train, y_train))
                alp_l1.append([alpha, l1])
        i_optim = np.argmax(train_error)
        alp_l1_optim = alp_l1[i_optim]
        # print("Optimal regularization parameter : %s" % alp_l1_optim)
        enet.set_params(alpha=alp_l1_optim[0])
        enet.set_params(l1_ratio=alp_l1_optim[1])

        y_score = enet.fit(X_train, y_train)._decision_function(X_test)
        # y_score = enet.predict(X_test)

        fpr, tpr, thresholds = roc_curve(y_test, y_score)
        # interp:插值 把结果添加到tprs列表中
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        # 计算auc
        roc_auc = auc(fpr, tpr)
        # aucs.append(roc_auc)

    return tprs, mean_fpr

def Model_SVM(model, skf, X, y):
    # model, skf, X, y = SVC(),skf,X2,y2
    def max_auc(model):
        y_score = model.fit(X_train, y_train).decision_function(X_test)
        y_predict = model.predict(X_test)
        fpr, tpr, thresholds = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        return roc_auc

    tprs = []
    # aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        max_ker, max_score = 'linear', 0
        ker_set = ['linear', 'poly', 'rbf', 'sigmoid']
        for ker in ker_set:
            model.set_params(kernel = ker)
            roc_auc = max_auc(model)
            if roc_auc > max_score:
                max_score  = roc_auc
                max_ker = ker
        model.set_params(kernel = max_ker)

        if max_ker == 'poly':
            max_deg, max_score = 1, 0
            for deg in np.arange(1,6,1):
                model.set_params(degree = deg)
                roc_auc = max_auc(model)
                if roc_auc > max_score:
                    max_score = roc_auc
                    max_deg = deg
            model.set_params(degree = max_deg)

        if max_ker != 'linear':
            max_coef ,max_score = 0.05, 0
            for coef in np.arange(0.05,1.05,0.05):
                model.set_params(coef0 = coef)
                roc_auc = max_auc(model)
                if roc_auc > max_score:
                    max_score = roc_auc
                    max_coef = coef
            model.set_params(coef0 = max_coef)

        max_c ,max_score = 0.05, 0
        for c in np.arange(0.05,1.05,0.05):
            model.set_params(C = c)
            roc_auc = max_auc(model)
            if roc_auc > max_score:
                max_score = roc_auc
                max_c = c
        model.set_params(C = max_c)


        y_score = model.fit(X_train, y_train).decision_function(X_test)
        y_predict = model.predict(X_test)
        fpr, tpr, thresholds = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)

        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0

    return tprs, mean_fpr

def Find_X_y(df, class_type, P_sample):
    # df, class_type, P_sample, N_sample = Vec_1, 'Effect_type', 'Synergistic', 'Antagonistic'
    df_syn = df.loc[df[class_type].isin([P_sample])]
    df_nonsyn1 = df.loc[~df[class_type].isin([P_sample])]
    df_nonsyn1.loc[:,class_type] = 'not_syn'
    df_nonsyn = df_nonsyn1.sample(n = len(df_syn))
    # random_state 控制随机状态，默认为 None，表示随机数据不会重复；若为 1 表示会取得重复数据。

    dff = pd.concat([df_syn,df_nonsyn])

    cols = []
    for i in dff.columns:
        if (dff[i].dtype == "float64") or (dff[i].dtype == 'int64'):
            cols.append(i)

    return dff[cols].copy(), pd.Categorical(dff[class_type]).codes

def Find_X_y_random(df, df_non,class_type ):
    "负样本为 notsyn + 随机选取"
    # df, df_non,class_type = Vec_4_1, Vec_4_2_nonsyn, 'Effect_type'
    df_syn = df.loc[df[class_type] == 'Synergistic']
    df_nonsyn = df_non.sample(n = len(df_syn))
    df_sample = pd.concat([df_syn,df_nonsyn])

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
            D_vec[node[0]] = node[1:]
    D_vec = pd.DataFrame(D_vec.values.T, index=D_vec.columns, columns=D_vec.index) # 数据旋转
    D_vec = D_vec.apply(pd.to_numeric) # 数据保留六位小数

    Vec_3_edge = read_file(file_edge)
    Vec_3_edge = Vec_3_edge[1:]
    DD_vec = []
    for j in range(len(Vec_3_edge)):
        edge = Vec_3_edge[j].replace('\n','').split(' ')
        if edge[0] == 'D-D':
            DD_vec = [round(float(e), 5) for e in edge[1:]]
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
    Vec_4 = pd.DataFrame(columns=list(range(len(D_vec.columns))))
    for dc in dcom_list:
        drugs = dc.split(',')
        if (drugs[0] in D_vec.index) and (drugs[1] in D_vec.index):
            Vec_4.loc[dc] = (D_vec.loc[drugs[0]] + D_vec.loc[drugs[1]]) * np.array(DD_vec)
            Vec_4.loc[dc,['Efficacy', 'Effect_type']] = [Vec_1.loc[dc, 'Efficacy'], Vec_1.loc[dc, 'Effect_type']]
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

def Vec4_3_notsyn(drugs_syn, D_vec, DD_vec):
    "(D + D + D-D)"
    Vec_4_nonsyn = pd.DataFrame(columns=list(range(len(D_vec.columns))))
    for i in range(len(drugs_syn)):
        for j in range(len(drugs_syn)):
            dc = ','.join([drugs_syn[i],drugs_syn[j]])
            if i<j :
                Vec_4_nonsyn.loc[dc] =( D_vec.loc[drugs_syn[i]] + D_vec.loc[drugs_syn[j]]) * np.array(DD_vec)
                Vec_4_nonsyn.loc[dc,['Efficacy', 'Effect_type']] = ['non_eff','non_syn']
        # print(i)
    # print("Finish")
    return Vec_4_nonsyn


file_vec1 = os.path.join(RESULTS_DIR, 'Vec_1.csv')
if not os.path.exists(file_vec1):
    file_vec1 = os.path.join(DATA_DIR, 'Vec_1.csv')
Vec_1 = pd.read_csv(file_vec1,header=0,index_col=0)

def get_vectors_path(dim):
    node_file = os.path.join(DATA_DIR, f'node_vectors_all_l5w3d{dim}.txt')
    edge_file = os.path.join(DATA_DIR, f'metapath_vectors_all_l5w3d{dim}.txt')
    if not os.path.exists(node_file):
        node_file = os.path.join(RESULTS_DIR, f'node_vectors_all_l5w3d{dim}.txt')
        edge_file = os.path.join(RESULTS_DIR, f'metapath_vectors_all_l5w3d{dim}.txt')
    return node_file, edge_file

file_node1, file_edge1 = get_vectors_path(100)
D_vec1, DD_vec1 = NodeVec_MetaVec(file_node1, file_edge1)
Vec_31 = Vec3(Vec_1, D_vec1)
drugs_syn = Vec3_syn(Vec_31)
Vec_31_nonsyn = Vec3_notsyn(drugs_syn, D_vec1)
Vec_41 = Vec4_3(Vec_1, D_vec1, DD_vec1)
Vec_41_nonsyn = Vec4_3_notsyn(drugs_syn, D_vec1, DD_vec1)

file_node2, file_edge2 = get_vectors_path(200)
D_vec2, DD_vec2 = NodeVec_MetaVec(file_node2, file_edge2)
Vec_32 = Vec3(Vec_1, D_vec2)
drugs_syn = Vec3_syn(Vec_32)
Vec_32_nonsyn = Vec3_notsyn(drugs_syn, D_vec2)
Vec_42 = Vec4_3(Vec_1, D_vec2, DD_vec2)
Vec_42_nonsyn = Vec4_3_notsyn(drugs_syn, D_vec2, DD_vec2)

file_node3, file_edge3 = get_vectors_path(300)
D_vec3, DD_vec3 = NodeVec_MetaVec(file_node3, file_edge3)
Vec_33 = Vec3(Vec_1, D_vec3)
drugs_syn = Vec3_syn(Vec_33)
Vec_33_nonsyn = Vec3_notsyn(drugs_syn, D_vec3)
Vec_43 = Vec4_3(Vec_1, D_vec3, DD_vec3)
Vec_43_nonsyn = Vec4_3_notsyn(drugs_syn, D_vec3, DD_vec3)

file_node4, file_edge4 = get_vectors_path(400)
D_vec4, DD_vec4 = NodeVec_MetaVec(file_node4, file_edge4)
Vec_34 = Vec3(Vec_1, D_vec4)
drugs_syn = Vec3_syn(Vec_34)
Vec_34_nonsyn = Vec3_notsyn(drugs_syn, D_vec4)
Vec_44 = Vec4_3(Vec_1, D_vec4, DD_vec4)
Vec_44_nonsyn = Vec4_3_notsyn(drugs_syn, D_vec4, DD_vec4)

file_node5, file_edge5 = get_vectors_path(500)
D_vec5, DD_vec5 = NodeVec_MetaVec(file_node5, file_edge5)
Vec_35 = Vec3(Vec_1, D_vec5)
drugs_syn = Vec3_syn(Vec_35)
Vec_35_nonsyn = Vec3_notsyn(drugs_syn, D_vec5)
Vec_45 = Vec4_3(Vec_1, D_vec5, DD_vec5)
Vec_45_nonsyn = Vec4_3_notsyn(drugs_syn, D_vec5, DD_vec5)

file_node6, file_edge6 = get_vectors_path(1000)
D_vec6, DD_vec6 = NodeVec_MetaVec(file_node6, file_edge6)
Vec_36 = Vec3(Vec_1, D_vec6)
drugs_syn = Vec3_syn(Vec_36)
Vec_36_nonsyn = Vec3_notsyn(drugs_syn, D_vec6)
Vec_46 = Vec4_3(Vec_1, D_vec6, DD_vec6)
Vec_46_nonsyn = Vec4_3_notsyn(drugs_syn, D_vec6, DD_vec6)

# file_node7 = '/Users/tlx/PythonFile/程序2022/Data_Prep/node_vectors_200_l5n8w3p1_1.txt'
# file_edge7 = '/Users/tlx/PythonFile/程序2022/Data_Prep/metapath_vectors_200_l5n8w3p1_1.txt'
# D_vec7, DD_vec7 = NodeVec_MetaVec(file_node7, file_edge7)
# Vec_37 = Vec3(Vec_1, D_vec7)
# drugs_syn = Vec3_syn(Vec_37)
# Vec_37_nonsyn = Vec3_notsyn(drugs_syn, D_vec7)
# Vec_47 = Vec4_2(Vec_1, D_vec7, DD_vec7)
# Vec_47_nonsyn = Vec4_2_notsyn(drugs_syn, D_vec7, DD_vec7)


X31, y31 = Find_X_y(Vec_31, 'Effect_type', 'Synergistic')
X32, y32 = Find_X_y(Vec_32, 'Effect_type', 'Synergistic')
X33, y33 = Find_X_y(Vec_33, 'Effect_type', 'Synergistic')
X34, y34 = Find_X_y(Vec_34, 'Effect_type', 'Synergistic')
X35, y35 = Find_X_y(Vec_35, 'Effect_type', 'Synergistic')
X36, y36 = Find_X_y(Vec_36, 'Effect_type', 'Synergistic')
# X37, y37 = Find_X_y(Vec_37, 'Effect_type', 'Synergistic')

X41, y41 = Find_X_y(Vec_41, 'Effect_type', 'Synergistic')
X42, y42 = Find_X_y(Vec_42, 'Effect_type', 'Synergistic')
X43, y43 = Find_X_y(Vec_43, 'Effect_type', 'Synergistic')
X44, y44 = Find_X_y(Vec_44, 'Effect_type', 'Synergistic')
X45, y45 = Find_X_y(Vec_45, 'Effect_type', 'Synergistic')
X46, y46 = Find_X_y(Vec_46, 'Effect_type', 'Synergistic')
# X47, y47 = Find_X_y(Vec_47, 'Effect_type', 'Synergistic')
"over"

X31, y31 = Find_X_y_random(Vec_31, Vec_31_nonsyn, 'Effect_type')
X32, y32 = Find_X_y_random(Vec_32, Vec_32_nonsyn, 'Effect_type')
X33, y33 = Find_X_y_random(Vec_33, Vec_33_nonsyn, 'Effect_type')
X34, y34 = Find_X_y_random(Vec_34, Vec_34_nonsyn, 'Effect_type')
X35, y35 = Find_X_y_random(Vec_35, Vec_35_nonsyn, 'Effect_type')
X36, y36 = Find_X_y_random(Vec_36, Vec_36_nonsyn, 'Effect_type')
# X37, y37 = Find_X_y_random(Vec_37, Vec_37_nonsyn, 'Effect_type')

X41, y41 = Find_X_y_random(Vec_41, Vec_41_nonsyn, 'Effect_type')
X42, y42 = Find_X_y_random(Vec_42, Vec_42_nonsyn, 'Effect_type')
X43, y43 = Find_X_y_random(Vec_43, Vec_43_nonsyn, 'Effect_type')
X44, y44 = Find_X_y_random(Vec_44, Vec_44_nonsyn, 'Effect_type')
X45, y45 = Find_X_y_random(Vec_45, Vec_45_nonsyn, 'Effect_type')
X46, y46 = Find_X_y_random(Vec_46, Vec_46_nonsyn, 'Effect_type')
# X47, y47 = Find_X_y_random(Vec_47, Vec_47_nonsyn, 'Effect_type')
"over"

# 交叉验证
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=50)
# 随机森林参数设置
rfc = RandomForestClassifier()
# svc = LinearSVC(C=1.0)
# svc = SVC(kernel='rbf',random_state=0, gamma = 1,C = 1.0)
print("计算AUC")
tprs_R31,mean_fpr = Model_RF(rfc,skf,X31,y31)
tprs_R32,mean_fpr = Model_RF(rfc,skf,X32,y32)
tprs_R33,mean_fpr = Model_RF(rfc,skf,X33,y33)
tprs_R34,mean_fpr = Model_RF(rfc,skf,X34,y34)
tprs_R35,mean_fpr = Model_RF(rfc,skf,X35,y35)
tprs_R36,mean_fpr = Model_RF(rfc,skf,X36,y36)
# tprs_R37,mean_fpr = Model_RF(rfc,skf,X37,y37)

tprs_R41,mean_fpr = Model_RF(rfc,skf,X41,y41)
tprs_R42,mean_fpr = Model_RF(rfc,skf,X42,y42)
tprs_R43,mean_fpr = Model_RF(rfc,skf,X43,y43)
tprs_R44,mean_fpr = Model_RF(rfc,skf,X44,y44)
tprs_R45,mean_fpr = Model_RF(rfc,skf,X45,y45)
tprs_R46,mean_fpr = Model_RF(rfc,skf,X46,y46)
# tprs_R47,mean_fpr = Model_RF(rfc,skf,X47,y47)
"Over"

print('画图')
def mean_roc(tprs,mean_fpr,lab,col,lst):
    mean_tpr=np.mean(tprs,axis=0)
    mean_tpr[-1]=1.0
    mean_auc=auc(mean_fpr,mean_tpr)#计算平均AUC值
    plt.plot(mean_fpr,mean_tpr,linestyle=lst,color=col,label=lab + ' (%0.4f)'%mean_auc,lw=2,alpha=.8)
    return

mean_roc(tprs_R31,mean_fpr,'d_100','b','-')
mean_roc(tprs_R32,mean_fpr,'d_200','g','-')
mean_roc(tprs_R33,mean_fpr,'d_300','y','-')
mean_roc(tprs_R34,mean_fpr,'d_400','r','-')
mean_roc(tprs_R35,mean_fpr,'d_500','c','-')
mean_roc(tprs_R36,mean_fpr,'d_1000','m','-')
# mean_roc(tprs_R37,mean_fpr,'RF_37','k','-')


mean_roc(tprs_R41,mean_fpr,'d_100','b','-')
mean_roc(tprs_R42,mean_fpr,'d_200','g','-')
mean_roc(tprs_R43,mean_fpr,'d_300','y','-')
mean_roc(tprs_R44,mean_fpr,'d_400','r','-')
mean_roc(tprs_R45,mean_fpr,'d_500','c','-')
mean_roc(tprs_R46,mean_fpr,'d_1000','m','-')
'over'
# mean_roc(tprs_R47,mean_fpr,'RF_47','k','-')
# tprs_R6 = []
# for i in range(len(tprs_R5)):
#     tprs_R6.append((tprs_R1[i] + tprs_R2[i] + tprs_R3[i] + tprs_R4[i] + tprs_R5[i])/5)
# mean_roc(tprs_R6,mean_fpr,'RF_6','b',':')

plt.plot([0,1],[0,1],linestyle=':',lw=2,color='gray')
plt.xlim([0,1.05])
plt.ylim([0,1.05])
plt.xlabel('False Positive Rate',fontsize=10)
plt.ylabel('True Positive Rate',fontsize=10)
plt.title('ROC')
plt.legend(loc='lower right')
plt.show()






