#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @software: PyCharm
# @project : 程序2022
# @File    : 构造特征向量3_HIN.py
# @Author  : TLX
# @Time    : 2022/11/21 16:10
'''
Step1: 生成 HIN 网络文件
Step2: 运行 HIN2Vec 算法
Step3: 构造 组合药物 特征向量3
'''

import numpy as np
import pandas as pd

import os

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data'))
RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../results'))
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

file_dcom = os.path.join(DATA_DIR, 'DCDB_134.txt')
data_dcom = pd.read_csv(file_dcom)
data_dcom.columns

'''寻找DB_ID'''
db_id = pd.read_csv(os.path.join(DATA_DIR, 'DB_Name.txt'), sep=',')
db_id['name'] = db_id['name'].str.lower()
db_id.columns

def DB_id(comp):
    '''提取药物名称对应有DB_ID'''
    a = db_id[(db_id.name == comp)].index.tolist()[0]
    return db_id.at[a,'drugbank_id']

# '''所有药物的DB_ID'''
# drug_name = set(data_dcom['COMPONENT_1']).union(set(data_dcom['COMPONENT_2']))
# drug_id = [DB_id(n) for n in drug_name]

'''生成网络文件'''

def Elect_Dcoms_type(effect_type):
    '''按照 effect_type 选取需要的药物组合'''
    ''' 'Additive','Antagonistic','Potentiative','Reductive','Synergistic','Unclear',nan  '''
    data = data_dcom[data_dcom.EFFECT_TYPE == effect_type]
    return data

def Elect_Dcoms_eff(efficacy):
    '''按照 efficacy 选取需要的药物组合'''
    ''' 'Efficacious', 'Need further study', 'Non-efficacious'  '''
    data = data_dcom[data_dcom.EFFICACY == effect_type]
    return data

def file_D_D(data):
    '''生成 D—D '''
    # data = Vec_1
    DD_list = pd.DataFrame(columns = Net_clo)
    for i in data.index:
        id1 = DB_id(data.at[i,'COMPONENT_1'])
        id2 = DB_id(data.at[i,'COMPONENT_2'])
        DD_list.loc[','.join([id1, id2])] = [id1, 'D', id2, 'D', 'D-D']
        DD_list.loc[','.join([id2, id1])] = [id2, 'D', id1, 'D', 'D-D']
        # DD_list.to_csv(os.path.join(RESULTS_DIR, 'DD_list.txt'), index=False, sep=' ', header= False)
        # 没有行索引，没有列索引，空格间隔
    return DD_list

def file_D_CTP(ctp_data, ctp_str, CTP):
    '''生成 D-C, D-T, D-P'''
    # drugs = set(df_dcom['COMPONENT_1']).union(set(df_dcom['COMPONENT_2']))
    # drugs_id = [DB_id(n) for n in drugs]

    D_CTP_list = pd.DataFrame(columns=Net_clo)
    for i in ctp_data.index:
        id = ctp_data.drugbank_id[i]
        ctp = ctp_data[ctp_str][i]
        D_CTP_list.loc[','.join([id, ctp])] = [id, 'D', ctp, CTP, 'D-' + CTP]
        D_CTP_list.loc[','.join([ctp, id])] = [ctp, CTP, id, 'D', CTP + '-D']
        print(i)

    return D_CTP_list

def file_C_C():
    cate = set(data_cate['category'])
    cate = list(cate)
    CC_list = pd.DataFrame(columns=Net_clo)
    for i in range(len(cate)):
        for j in range(len(cate)):
            if i < j:
                sim = len(set(cate[i].split(' ')).intersection(set(cate[j].split(' ')))) / len(set(cate[i].split(' ')).union(set(cate[j].split(' '))))
                if sim >= 0.5:
                    CC_list.loc[','.join([cate[i], cate[j]])] = [cate[i], 'C', cate[j], 'C', 'C-C']
                    CC_list.loc[','.join([cate[j], cate[i]])] = [cate[j], 'C', cate[i], 'C', 'C-C']

    return CC_list

def file_T_T():

    tar_int = pd.read_excel(os.path.join(DATA_DIR, 'PPI.xlsx'))
    tar_int.columns

    tar_list = set(data_tar['Gene_ID'])
    # tar_list1 = set(data_tar['id'])
    tar_list = [int(tt) for tt in tar_list]
    tar_list = sorted(tar_list)

    TT_list = pd.DataFrame(columns=Net_clo)
    # i = 0
    for t in tar_list:
        # t = tar_list[0]
        tar_int_A = tar_int[tar_int['Protein_A_Entrez_ID'] == t]
        for ta in set(tar_int_A['Protein_B_Entrez_ID']):
            if ta in tar_list:
                # print(ta)
                TT_list.loc[','.join([str(t), str(ta)])] = [str(t), 'T', str(ta), 'T', 'T-T']
                TT_list.loc[','.join([str(ta), str(t)])] = [str(ta), 'T', str(t), 'T', 'T-T']

        tar_int_B = tar_int[tar_int['Protein_B_Entrez_ID'] == t]
        for tb in set(tar_int_B['Protein_A_Entrez_ID']):
            if tb in tar_list:
                TT_list.loc[','.join([str(t), str(tb)])] = [str(t), 'T', str(tb), 'T', 'T-T']
                TT_list.loc[','.join([str(tb), str(t)])] = [str(tb), 'T', str(t), 'T', 'T-T']
    return TT_list

def file_P_P():
    # data_path['pathway_enzgmes']
    data_path_s = data_path.drop_duplicates(subset=['pathway_smpdb_id'],keep='first', inplace=False) #去除重复项 -- 874
    data_path_s = data_path_s[data_path_s['pathway_enzgmes'] != '无'] # 删除 ‘无'
    data_path_s = data_path_s.set_index('pathway_smpdb_id', drop=False) # 设置新的index，并保留原来的列

    PP_list = pd.DataFrame(columns=Net_clo)
    path_list = data_path_s.index
    for i in range(len(path_list)):
        for j in range(len(path_list)):
            # i,j = 1,0
            if i < j:
                p_ij = set(data_path_s.at[path_list[i],'pathway_enzgmes'].split(';')).intersection(set(data_path_s.at[path_list[j],'pathway_enzgmes'].split(';')))
                if len(p_ij) > 0:
                    PP_list.loc[','.join([path_list[i], path_list[j]])] = [path_list[i], 'P', path_list[j], 'P', 'P-P']
                    PP_list.loc[','.join([path_list[j], path_list[i]])] = [path_list[j], 'P', path_list[i], 'P', 'P-P']
        # print(i)
    return PP_list

def file_T_P():
    data_path_s = data_path.drop_duplicates(subset=['pathway_smpdb_id'],keep='first', inplace=False) #去除重复项 -- 874
    data_path_s = data_path_s[data_path_s['pathway_enzgmes'] != '无'] # 删除 ‘无'
    data_path_s = data_path_s.set_index('pathway_smpdb_id', drop=False)

    data_tar_s = data_tar.drop_duplicates(subset=['polypeptide_id'], keep='first',inplace=False) # 2707 (2688,Gene_ID)
    data_tar_s = data_tar_s.set_index('polypeptide_id', drop=False)

    path_list = data_path_s.index
    tar_list = data_tar_s.index
    TP_list = pd.DataFrame(columns=Net_clo)
    # i = 0
    for p in path_list:
        p_tars = data_path_s.at[p, 'pathway_enzgmes'].split(';')
        for t in p_tars:
            if t in tar_list :
                tar_id = data_tar_s.at[t, 'Gene_ID']
                # print(t,tar_id)
                TP_list.loc[','.join([p, str(tar_id)])] = [p, 'P', str(tar_id), 'T', 'P-T']
                TP_list.loc[','.join([str(tar_id), p])] = [str(tar_id), 'T', p, 'P', 'T-P']
        # print(i)
        # i += 1
    return TP_list

def Find_all_drugs(df):
    drugs = []
    for dc in df.index:
        dcom = dc.split(',')
        if dcom[0] not in drugs:
            drugs.append(dcom[0])
        if dcom[1] not in drugs:
            drugs.append(dcom[1])
    return drugs

'''组合药物样本信息 --- 根据Vec_1'''
Vec_1 = pd.read_csv(os.path.join(RESULTS_DIR, 'Vec_1.csv'), header=0, index_col=0)
"涉及的所有药物"
drugs = Find_all_drugs(Vec_1)
df_syn = Elect_Dcoms_type('Synergistic')


'''读取 ctp 信息'''
data_cate = pd.read_csv(os.path.join(DATA_DIR, 'DB_Category.txt'))
data_cate['category'] = data_cate['category'].str.lower()
# data_cate.columns
data_tar = pd.read_csv(os.path.join(DATA_DIR, 'DB_Target.txt')) # 27234
data_tar = data_tar[data_tar.type == 'target'] # 19018
data_tar = data_tar[data_tar.Gene_ID > 0] # 14178
# data_tar.columns
data_path = pd.read_csv(os.path.join(DATA_DIR, 'DB_Pathway.txt'))
# data_path.columns


Net_clo = ['source_node', 'source_class','dest_node','dest_class', 'edge_class']
Net_txt = pd.DataFrame(columns = Net_clo)
DD_list = file_D_D(df_syn)
print('DD')
DC_list = file_D_CTP(data_cate, 'category', 'C')
print('DC')
data_tar['Gene_ID'] = data_tar['Gene_ID'].apply(str)
DT_list = file_D_CTP( data_tar, 'Gene_ID', 'T')
print('DT')
DP_list = file_D_CTP(data_path, 'pathway_smpdb_id', 'P')
print('DP')
CC_list = file_C_C()
print('CC')
TT_list = file_T_T()
print('TT')
PP_list = file_P_P()
print('PP')
TP_list = file_T_P()
print('TP')

Net_txt_all = pd.concat([DD_list,DC_list,DT_list,DP_list,CC_list,TT_list,PP_list,TP_list], ignore_index=True)
print(Net_txt_all.shape)
Net_txt_all.to_csv(os.path.join(RESULTS_DIR, 'HIN_all.txt'),header=0,index=False,sep='\t')
print('Finish-all')

Net_txt_DC = pd.concat([DD_list,DC_list,CC_list], ignore_index=True)
print(Net_txt_DC.shape)
Net_txt_DC.to_csv(os.path.join(RESULTS_DIR, 'HIN_DC.txt'),header=0,index=False,sep='\t')
print('Finish-DC')

Net_txt_DT = pd.concat([DD_list,DT_list,TT_list], ignore_index=True)
print(Net_txt_DT.shape)
Net_txt_DT.to_csv(os.path.join(RESULTS_DIR, 'HIN_DT.txt'),header=0,index=False,sep='\t')
print('Finish-DT')

Net_txt_DP = pd.concat([DD_list,DP_list,PP_list], ignore_index=True)
print(Net_txt_DP.shape)
Net_txt_DP.to_csv(os.path.join(RESULTS_DIR, 'HIN_DP.txt'),header=0,index=False,sep='\t')
print('Finish-DP')

Net_txt_DTP = pd.concat([DD_list,DT_list,DP_list,TT_list,PP_list,TP_list], ignore_index=True)
print(Net_txt_DTP.shape)
Net_txt_DTP.to_csv(os.path.join(RESULTS_DIR, 'HIN_DTP.txt'),header=0,index=False,sep='\t')
print('Finish-DTP')

