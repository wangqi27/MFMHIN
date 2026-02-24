#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @software: PyCharm
# @project : 程序2022
# @File    : 算法123_预测.py
# @Author  : TLX
# @Time    : 2023/2/13 14:31


import pandas as pd
import numpy as np
import math
import os
from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.preprocessing import StandardScaler
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

def Model_RF(model, skf, X, y, X_predict):
    # model, skf, X, y, X_predict = rfc1, skf, X1,y1, X_predict1
    def max_auc(model):
        y_score = model.fit(X_train, y_train).predict_proba(X_test)
        fpr, tpr, thresholds = roc_curve(y_test, y_score[:, 1])
        roc_auc = auc(fpr, tpr)
        return roc_auc

    yy_predict_set = []
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

        yy_score = model.fit(X_train, y_train).predict_proba(X_predict)
        yy_predict = yy_score[:,1]
        yy_predict_set.append(yy_predict)

    yy_predict_f = np.sum(yy_predict_set,axis=0)/5
    # yy_result = pd.DataFrame(yy_predict_f,index=X_predict.index,columns=['score'])

    return yy_predict_f

def Model_SVM(model, skf, X, y, X_predict):
    # model, skf, X, y = SVC(),skf,X2,y2
    def max_auc(model):
        y_score = model.fit(X_train, y_train).decision_function(X_test)
        fpr, tpr, thresholds = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        return roc_auc

    yy_predict_set = []
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

        y_score = model.fit(X_train, y_train).decision_function(X_predict)
        y_score = [1 / (1 + math.exp(-s)) for s in y_score]
        yy_predict_set.append(y_score)


    yy_predict_f = np.sum(yy_predict_set,axis=0)/5
    yy_result = pd.DataFrame(yy_predict_f,index=X_predict.index,columns=['score'])

    return yy_predict_f

def Model_ElsNet(model, skf, X, y, X_predict):
    # model, skf, X, y = ElasticNet(), skf, X1, y1
    def max_auc(model):
        y_score = model.fit(X_train, y_train)._decision_function(X_test)
        fpr, tpr, thresholds = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        return roc_auc

    yy_predict_set = []
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

        y_score = model.fit(X_train, y_train)._decision_function(X_predict)
        y_score = [1 / (1 + math.exp(-s)) for s in y_score]
        yy_predict_set.append(y_score)

    yy_predict_f = np.sum(yy_predict_set,axis=0)/5
    yy_result = pd.DataFrame(yy_predict_f,index=X_predict.index,columns=['score'])

    return yy_predict_f

def Find_X_y(df, class_type, P_sample, N_sample):
    # df, class_type, P_sample, N_sample = Vec_1, 'Effect_type', 'Synergistic', ['Antagonistic','Additive','Unclear','No_type']
    # df, class_type, P_sample, N_sample = Vec_1, 'Efficacy', 'Efficacious', 'Non-efficacious'

    df_syn = df.loc[df[class_type].isin([P_sample])]
    df_syn = df.loc[df['Effect_type'].isin(['Synergistic'])]
    df_nonsyn1 = df.loc[df[class_type].isin(N_sample)]
    df_nonsyn1.loc[:,class_type] = 'not_syn'
    if len(df_syn) >= len(df_nonsyn1):
        df_nonsyn = df_nonsyn1
    else:
        df_nonsyn = df_nonsyn1.sample(n=len(df_syn))

    dff = pd.concat([df_nonsyn,df_syn])
    cols = []
    for i in dff.columns:
        if (dff[i].dtype == "float64") or (dff[i].dtype == 'int64'):
            cols.append(i)

    return dff[cols].copy(), pd.Categorical(dff[class_type]).codes

def Find_X_y_random(df, df_non,class_type ):
    "负样本为 notsyn + 随机选取"
    # df, df_non,class_type = Vec_4, Vec_4_nonsyn, 'Effect_type'
    # del_dcom_inc = read_file(os.path.join(DATA_DIR, "del_dcoms.txt"))
    # del_dcom_no = read_file(os.path.join(DATA_DIR, 'del_dcoms_no.txt'))
    # del_dcom_inc.extend(del_dcom_no)
    # del_dcom = [d.replace('\n','') for d in del_dcom_inc]
    # ls_index = [l for l in df_non.index if l not in del_dcom]
    # df_non = df_non[df_non.index.isin(ls_index)]

    save_no_risk = read_file(os.path.join(DATA_DIR, 'del_no_risk.txt'))
    save_data = [d.replace('\n', '') for d in save_no_risk]
    df_non = df_non[df_non.index.isin(save_data)]

    df_syn = df.loc[df[class_type] == 'Synergistic']
    df_nonsyn = df_non.sample(n = len(df_syn))
    df_sample = pd.concat([df_nonsyn,df_syn])

    cols = []
    for i in df_sample.columns:
        if (df_sample[i].dtype == "float64") or (df_sample[i].dtype == 'int64'):
            cols.append(i)

    return df_sample[cols].copy(), pd.Categorical(df_sample[class_type]).codes

def Find_X_predict(df_sample):
    # df_sample = Vec_4_nonsyn
    dcom_no = read_file(os.path.join(DATA_DIR, "del_no_risk.txt"))
    dcom_no = [d.replace('\n','') for d in dcom_no]
    ls_index = [l for l in df_sample.index if l not in dcom_no]
    df_sample = df_sample[df_sample.index.isin(ls_index)]


    cols = []
    for i in df_sample.columns:
        if (df_sample[i].dtype == "float64") or (df_sample[i].dtype == 'int64'):
            cols.append(i)

    return df_sample[cols].copy()

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

X_predict1 = Find_X_predict(Vec_1_nonsyn)
X_predict2 = Find_X_predict(Vec_2_nonsyn)
X_predict4 = Find_X_predict(Vec_4_nonsyn)

yy_result_RF_1_all,yy_result_RF_2_all,yy_result_RF_4_all = [],[],[]
yy_result_SVM_1_all,yy_result_SVM_2_all,yy_result_SVM_4_all = [],[],[]
yy_result_EN_1_all,yy_result_EN_2_all,yy_result_EN_4_all = [],[],[]
for i in range(5):
    # X1, y1 = Find_X_y(Vec_1, 'Effect_type', 'Synergistic', ['Antagonistic','Additive','Unclear','No_type'])
    # X2, y2 = Find_X_y(Vec_2, 'Effect_type', 'Synergistic', ['Antagonistic','Additive','Unclear','No_type'])
    # X3, y3 = Find_X_y(Vec_3, 'Effect_type', 'Synergistic', ['Antagonistic','Additive','Unclear','No_type'])
    # X4, y4 = Find_X_y(Vec_4, 'Effect_type', 'Synergistic', ['Antagonistic','Additive','Unclear','No_type'])

    #
    # X1, y1 = Find_X_y(Vec_1, 'Efficacy', 'Efficacious', ['Non-efficacious'])
    # X2, y2 = Find_X_y(Vec_2, 'Efficacy', 'Efficacious',['Non-efficacious'])
    # X3, y3 = Find_X_y(Vec_3, 'Efficacy', 'Efficacious',['Non-efficacious'])
    # X4, y4 = Find_X_y(Vec_4, 'Efficacy', 'Efficacious',['Non-efficacious'])
    #
    X1, y1 = Find_X_y_random(Vec_1, Vec_1_nonsyn, 'Effect_type')
    X2, y2 = Find_X_y_random(Vec_2, Vec_2_nonsyn, 'Effect_type')
    X3, y3 = Find_X_y_random(Vec_3, Vec_3_nonsyn, 'Effect_type')
    X4, y4 = Find_X_y_random(Vec_4, Vec_4_nonsyn, 'Effect_type')


    print('相似性计算')
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=200)

    # RF
    rfc1, rfc2,rfc3,rfc4 = RandomForestClassifier(),RandomForestClassifier(),RandomForestClassifier(),RandomForestClassifier()
    yy_result_RF_1 = Model_RF(rfc1, skf, X1, y1, X_predict1)
    yy_result_RF_2 = Model_RF(rfc2, skf, X2, y2, X_predict2)
    yy_result_RF_4 = Model_RF(rfc4, skf, X4, y4, X_predict4)
    yy_result_RF_1_all.append(yy_result_RF_1)
    yy_result_RF_2_all.append(yy_result_RF_2)
    yy_result_RF_4_all.append(yy_result_RF_4)
    print('rfc--OK')

    # SVM
    svc1, svc2, svc3, svc4 = SVC(), SVC(), SVC(), SVC()
    yy_result_SVM_1 = Model_SVM(svc1,skf,X1,y1, X_predict1)
    yy_result_SVM_2 = Model_SVM(svc2,skf,X2,y2, X_predict2)
    yy_result_SVM_4 = Model_SVM(svc4,skf,X4,y4, X_predict4)
    yy_result_SVM_1_all.append(yy_result_SVM_1)
    yy_result_SVM_2_all.append(yy_result_SVM_2)
    yy_result_SVM_4_all.append(yy_result_SVM_4)
    print('SVC -- OK')

    # Elastic Net
    enet1,enet2,enet3,enet4 = ElasticNet(max_iter=5000, tol=0.01),ElasticNet(max_iter=5000, tol=0.01),ElasticNet(max_iter=5000, tol=0.01),ElasticNet(max_iter=5000, tol=0.01)
    yy_result_EN_1 = Model_ElsNet(enet1,skf,X1,y1, X_predict1)
    yy_result_EN_2 = Model_ElsNet(enet2,skf,X2,y2, X_predict2)
    yy_result_EN_4 = Model_ElsNet(enet4,skf,X4,y4, X_predict4)
    yy_result_EN_1_all.append(yy_result_EN_1)
    yy_result_EN_2_all.append(yy_result_EN_2)
    yy_result_EN_4_all.append(yy_result_EN_4)
    print('ENET -- OK')


def Find_DrugName():
    # yy_result = yy_result_SVM_4
    file = os.path.join(DATA_DIR, "DB_Name.txt")
    drug_name = read_file(file)
    list_id = []
    list_name = []
    for line in drug_name:
        element = line.replace('\n','').split(',')
        list_id.append(str(element[0]))
        list_name.append(str(element[2]))
    return list_id, list_name

def yy_result_drugname_predict(yy_result,list_id, list_name,label_name):
    yy_predict = pd.read_csv(os.path.join(DATA_DIR, "yy_result.txt"),sep='\t')
    drugids = []
    for ii in range(len(yy_predict)):
        dc0 = set([yy_predict["Column1"][ii], yy_predict["Column2"][ii]])
        drugids.append(dc0)

    for d in range(len(yy_result.index)):
        dc = yy_result.index[d]
        DComs = dc.split(',')
        drug1 = DComs[0]
        drug2 = DComs[1]
        i = list_id.index(drug1)
        j = list_id.index(drug2)
        drug1_name = list_name[i]
        drug2_name = list_name[j]
        yy_result.loc[dc,'drug1'] = list_name[i]
        yy_result.loc[dc,'drug2'] = list_name[j]
        yy_result.loc[dc,'label'] = label_name

        d1 = yy_predict["Column1"].index

        for jj in range(len(drugids)):
            if (drug1 in drugids[jj]) and (drug2 in drugids[jj]):
                l = jj
                break
        yy_result.loc[dc, 'interaction'] = yy_predict["interactions"][jj]

    return yy_result

yy_result_EN_1_mean	= pd.DataFrame(np.sum(yy_result_EN_1_all,axis=0)/5,index=X_predict1.index,columns=['score'])
yy_result_EN_2_mean	= pd.DataFrame(np.sum(yy_result_EN_2_all,axis=0)/5,index=X_predict2.index,columns=['score'])
yy_result_EN_4_mean	= pd.DataFrame(np.sum(yy_result_EN_4_all,axis=0)/5,index=X_predict4.index,columns=['score'])
yy_result_RF_1_mean	= pd.DataFrame(np.sum(yy_result_RF_1_all,axis=0)/5,index=X_predict1.index,columns=['score'])
yy_result_RF_2_mean	= pd.DataFrame(np.sum(yy_result_RF_2_all,axis=0)/5,index=X_predict2.index,columns=['score'])
yy_result_RF_4_mean	= pd.DataFrame(np.sum(yy_result_RF_4_all,axis=0)/5,index=X_predict4.index,columns=['score'])
yy_result_SVM_1_mean= pd.DataFrame(np.sum(yy_result_SVM_1_all,axis=0)/5,index=X_predict1.index,columns=['score'])
yy_result_SVM_2_mean= pd.DataFrame(np.sum(yy_result_SVM_2_all,axis=0)/5,index=X_predict2.index,columns=['score'])
yy_result_SVM_4_mean= pd.DataFrame(np.sum(yy_result_SVM_4_all,axis=0)/5,index=X_predict4.index,columns=['score'])

list_id, list_name = Find_DrugName()
# yy_result_EN_1_mean = yy_result_EN_1.sort_values(by=['score'],ascending=False)
yy_result_EN_1_mean = yy_result_drugname_predict(yy_result_EN_1_mean,list_id,list_name,'EN_1')
# yy_result_EN_1.to_csv('/Users/tlx/PythonFile/程序2022/Data_Prep/yy_result_EN_1.txt')
# yy_result_EN_2 = yy_result_EN_2.sort_values(by=['score'],ascending=False)
yy_result_EN_2_mean = yy_result_drugname_predict(yy_result_EN_2_mean,list_id,list_name,'EN_2')
# yy_result_EN_2.to_csv('/Users/tlx/PythonFile/程序2022/Data_Prep/yy_result_EN_2.txt')
# yy_result_EN_4 = yy_result_EN_4.sort_values(by=['score'],ascending=False)
yy_result_EN_4_mean = yy_result_drugname_predict(yy_result_EN_4_mean,list_id,list_name,'EN_4')
# yy_result_EN_4.to_csv('/Users/tlx/PythonFile/程序2022/Data_Prep/yy_result_EN_4.txt')

# yy_result_RF_1 = yy_result_RF_1.sort_values(by=['score'],ascending=False)
yy_result_RF_1_mean = yy_result_drugname_predict(yy_result_RF_1_mean,list_id,list_name,'RF_1')
# yy_result_RF_1.to_csv('/Users/tlx/PythonFile/程序2022/Data_Prep/yy_result_RF_1.txt')
# yy_result_RF_2 = yy_result_RF_2.sort_values(by=['score'],ascending=False)
yy_result_RF_2_mean = yy_result_drugname_predict(yy_result_RF_2_mean,list_id,list_name,'RF_2')
# yy_result_RF_2.to_csv('/Users/tlx/PythonFile/程序2022/Data_Prep/yy_result_RF_2.txt')
# yy_result_RF_4 = yy_result_RF_4.sort_values(by=['score'],ascending=False)
yy_result_RF_4_mean = yy_result_drugname_predict(yy_result_RF_4_mean,list_id,list_name,'RF_4')
# yy_result_RF_4.to_csv('/Users/tlx/PythonFile/程序2022/Data_Prep/yy_result_RF_4.txt')

# yy_result_SVM_1 = yy_result_SVM_1.sort_values(by=['score'],ascending=False)
yy_result_SVM_1_mean = yy_result_drugname_predict(yy_result_SVM_1_mean,list_id,list_name,'SVM_1')
# yy_result_SVM_1.to_csv('/Users/tlx/PythonFile/程序2022/Data_Prep/yy_result_SVM_1.txt')
# yy_result_SVM_2 = yy_result_SVM_2.sort_values(by=['score'],ascending=False)
yy_result_SVM_2_mean = yy_result_drugname_predict(yy_result_SVM_2_mean,list_id,list_name,'SVM_2')
# yy_result_SVM_2.to_csv('/Users/tlx/PythonFile/程序2022/Data_Prep/yy_result_SVM_2.txt')
# yy_result_SVM_4 = yy_result_SVM_4.sort_values(by=['score'],ascending=False)
yy_result_SVM_4_mean = yy_result_drugname_predict(yy_result_SVM_4_mean,list_id,list_name,'SVM_4')
# yy_result_SVM_4.to_csv('/Users/tlx/PythonFile/程序2022/Data_Prep/yy_result_SVM_4.txt')


yy_result_all = pd.concat([yy_result_EN_1_mean,yy_result_EN_2_mean,yy_result_EN_4_mean,yy_result_RF_1_mean,yy_result_RF_2_mean,yy_result_RF_4_mean,yy_result_SVM_1_mean,yy_result_SVM_2_mean,yy_result_SVM_4_mean])

yy_result_all.to_csv('/Users/tlx/PythonFile/程序2022/Data_Prep/预测结果_increase_23_111.txt')

yy_result_all = pd.read_csv('/Users/tlx/PythonFile/程序2022/Data_Prep/预测结果_increase_23_334.txt', header=0, index_col=0)
inter_label = read_file('/Users/tlx/PythonFile/程序2022/Data_Prep/interaction_label.txt',)
interactions = []
labels = []
for i in range(len(inter_label)):
    il = inter_label[i].replace('\n','').split('\t')
    interactions.append(il[0])
    labels.append(il[1])

yy_result_mean_1 = pd.DataFrame(columns=['score', 'drug1', 'drug2', 'label', 'interaction','inter_label'])
yy_result_mean_2 = pd.DataFrame(columns=['score', 'drug1', 'drug2', 'label', 'interaction','inter_label'])
yy_result_mean_4 = pd.DataFrame(columns=['score', 'drug1', 'drug2', 'label', 'interaction','inter_label'])
yy_result_mean_vote_1 = pd.DataFrame(columns=['score', 'drug1', 'drug2', 'label', 'interaction','inter_label'])
yy_result_mean_vote_2 = pd.DataFrame(columns=['score', 'drug1', 'drug2', 'label', 'interaction','inter_label'])
yy_result_mean_vote_4 = pd.DataFrame(columns=['score', 'drug1', 'drug2', 'label', 'interaction','inter_label'])
for l in set(yy_result_all.index):
    yy_l = yy_result_all.loc[l]
    yy_1 = yy_l.loc[yy_l['label'].isin(['EN_1','SVM_1','RF_1'])]
    yy_2 = yy_l.loc[yy_l['label'].isin(['EN_2', 'SVM_2', 'RF_2'])]
    yy_4 = yy_l.loc[yy_l['label'].isin(['EN_4', 'SVM_4', 'RF_4'])]
    j1 = interactions.index(yy_1['interaction'][0])
    j2 = interactions.index(yy_2['interaction'][0])
    j4 = interactions.index(yy_4['interaction'][0])

    yy_result_mean_1.loc[l] = [np.mean(yy_1.score), yy_1['drug1'][0], yy_1['drug2'][0],'X1',yy_1['interaction'][0],labels[j1]]
    yy_result_mean_2.loc[l] = [np.mean(yy_2.score), yy_2['drug1'][0], yy_2['drug2'][0], 'X2', yy_2['interaction'][0],labels[j2]]
    yy_result_mean_4.loc[l] = [np.mean(yy_4.score), yy_4['drug1'][0], yy_4['drug2'][0], 'X4', yy_4['interaction'][0],labels[j4]]
    vote_1 = [1 if l>=0.5 else 0 for l in yy_1.score]
    vote_2 = [1 if l >= 0.5 else 0 for l in yy_2.score]
    vote_4 = [1 if l >= 0.5 else 0 for l in yy_4.score]
    yy_result_mean_vote_1.loc[l] = [np.sum(vote_1), yy_1['drug1'][0], yy_1['drug2'][0],'X1',yy_1['interaction'][0],labels[j1]]
    yy_result_mean_vote_2.loc[l] = [np.sum(vote_2), yy_2['drug1'][0], yy_2['drug2'][0], 'X2', yy_2['interaction'][0],labels[j2]]
    yy_result_mean_vote_4.loc[l] = [np.sum(vote_4), yy_4['drug1'][0], yy_4['drug2'][0], 'X4', yy_4['interaction'][0],labels[j4]]

yy_result_mean = pd.concat([yy_result_mean_1,yy_result_mean_2,yy_result_mean_4])
yy_result_mean.to_csv('/Users/tlx/PythonFile/程序2022/Data_Prep/预测结果_mean_23_334.txt')
yy_result_mean_vote = pd.concat([yy_result_mean_vote_1,yy_result_mean_vote_2,yy_result_mean_vote_4])
yy_result_mean_vote.to_csv('/Users/tlx/PythonFile/程序2022/Data_Prep/预测结果_mean_vote_23_334.txt')

'over'





































