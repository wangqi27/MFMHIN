#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/11/19 17:40
# @Author  : TLX
# @File    : 构造特征向量2.py --- 向量相加

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

'''所有药物的DB_ID'''
drug_name = set(data_dcom['COMPONENT_1']).union(set(data_dcom['COMPONENT_2']))
drug_id = [DB_id(n) for n in drug_name]

'''读取 ctp 信息'''
data_cate = pd.read_csv(os.path.join(DATA_DIR, 'DB_Category.txt'))
data_cate.columns
data_cate['category'] = data_cate['category'].str.lower()
data_tar = pd.read_csv(os.path.join(DATA_DIR, 'DB_Target.txt'))
data_tar = data_tar[data_tar.type == 'target']
data_tar.columns
data_path = pd.read_csv(os.path.join(DATA_DIR, 'DB_Pathway.txt'))
data_path.columns

vec2_columns = list(set(data_cate['category'])) + list(set(data_tar['id'])) + list(set(data_path['pathway_smpdb_id'])) # 9878
vec2_drug = pd.DataFrame(np.zeros((128,9878)), columns=vec2_columns, index=drug_id)

'''单个药物的 one-hot 向量'''
def Drug_One_hot(id):
    df_cate = data_cate[data_cate.drugbank_id == id]
    df_tar = data_tar[data_tar.drugbank_id == id]
    df_path = data_path[data_path.drugbank_id == id]
    id_vec = list(set(df_cate['category'])) + list(set(df_tar['id'])) + list(set(df_path['pathway_smpdb_id']))
    return  id_vec

'''构建所有药物的 one-hot 向量'''
for id in drug_id:
    id_col = Drug_One_hot(id)
    vec2_drug.loc[id, id_col] = 1

'''构建药物组合 特征向量2 data_vec2 '''

dcom_len = data_dcom.shape[0]
data_vec2 = pd.DataFrame(columns=vec2_columns)
for i in range(dcom_len):
    id1 = DB_id(data_dcom['COMPONENT_1'][i])
    id2 = DB_id(data_dcom['COMPONENT_2'][i])
    data_vec2.loc[','.join([id1, id2])] = vec2_drug.loc[id1] + vec2_drug.loc[id2]
    data_vec2.loc[','.join([id1, id2]), 'Efficacy'] = data_dcom.at[i,'EFFICACY']
    if str(data_dcom.at[i, 'EFFECT_TYPE']) == 'nan':
        data_vec2.loc[','.join([id1, id2]), 'Effect_type'] = 'No_type'
    else:
        data_vec2.loc[','.join([id1, id2]), 'Effect_type'] = data_dcom.at[i, 'EFFECT_TYPE']
print('Finish')

data_vec2.to_csv(os.path.join(RESULTS_DIR, 'Vec_2.csv'))
print('Write in file_vec2')

print('构造 35 药物的其他组合')
df_syn = data_vec2[data_vec2['Effect_type'] == 'Synergistic']
drugs_syn = []
for dc in df_syn.index:
    dc = dc.split(',')
    if dc[0] not in drugs_syn:
        drugs_syn.append(dc[0])
    if dc[1] not in drugs_syn:
        drugs_syn.append(dc[1])


data_vec2_nonsyn = pd.DataFrame(columns=vec2_columns)
for i in range(len(drugs_syn)):
    for j in range(len(drugs_syn)):
        if i < j :
            dcom1 = ','.join([drugs_syn[i], drugs_syn[j]])
            dcom2 = ','.join([drugs_syn[j], drugs_syn[i]])
            if (dcom1 not in df_syn.index) and (dcom2 not in df_syn.index):
                data_vec2_nonsyn.loc[dcom1] = vec2_drug.loc[drugs_syn[i]] + vec2_drug.loc[drugs_syn[j]]
                data_vec2_nonsyn.loc[dcom1, 'Efficacy'] = 'No_eff'
                data_vec2_nonsyn.loc[dcom1, 'Effect_type'] = 'Not_syn'
    print(i)

print('Finish')

data_vec2_nonsyn.to_csv(os.path.join(RESULTS_DIR, 'Vec_2_nonsyn.csv'))
print('Write in file_vec2_nonsyn')























