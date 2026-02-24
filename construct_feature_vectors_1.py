#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/11/10 17:06
# @Author  : TLX
# @File    : construct_feature_vectors_1.py

import os
import numpy as np
import pandas as pd

# Define data directory relative to this script
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

'''化学结构相似性'''
Sim_Chem = pd.read_csv(os.path.join(DATA_DIR, "Sim_Chem.txt"), index_col=0)
def Extract_Sim_Chem(id1,id2):
    return round(Sim_Chem.at[id1,id2],2)

'''Jaccard相似性'''
def Jaccard(s1,s2):
    value = len(set(s1).intersection(set(s2))) / len(set(s1).union(set(s2)))
    return round(value,2)



'''ATC相似性'''
data_atc = pd.read_csv(os.path.join(DATA_DIR, "DB_ATC.txt"))
data_atc.head()
def Extract_Sim_ATC(id1,id2):
    '''ATC编码的前三级别'''
    data1 = data_atc[data_atc.drugbank_id == id1]
    data2 = data_atc[data_atc.drugbank_id == id2]
    def atc_set123(data_codes):
        atc1, atc2, atc3 = [], [], []
        for a in data_codes:
            if a[0] not in atc1:
                atc1.append(a[0])
            if a[1:3] not in atc2:
                atc2.append(a[1:3])
            if a[3] not in atc3:
                atc3.append(a[3])
        return atc1,atc2,atc3
    atc1_1, atc1_2, atc1_3 = atc_set123(data1['atc-codes'])
    atc2_1, atc2_2, atc2_3 = atc_set123(data2['atc-codes'])
    Sim_atc_1 = Jaccard(atc1_1, atc2_1)
    Sim_atc_2 = Jaccard(atc1_2, atc2_2)
    Sim_atc_3 = Jaccard(atc1_3, atc2_3)

    return Sim_atc_1, Sim_atc_2, Sim_atc_3

'''Category相似性'''
data_cate = pd.read_csv(os.path.join(DATA_DIR, 'DB_Category.txt'))
data_cate.head()
data_cate.columns
data_cate['category'] = data_cate['category'].str.lower()
def Extract_Sim_Cate(id1,id2):
    '''data_cate.category'''
    data1 = data_cate[data_cate.drugbank_id == id1]
    data2 = data_cate[data_cate.drugbank_id == id2]
    Sim_Cate = Jaccard(data1['category'],data2['category'])
    return Sim_Cate

'''target相似性'''
data_tar = pd.read_csv(os.path.join(DATA_DIR, 'DB_Target.txt'))
data_tar.head()
data_tar.columns
data_tar = data_tar[data_tar.type == 'target'] # type : 'carriers', 'enzymes', 'target', 'transporters'
def Extract_Sim_Tar(id1,id2):
    '''data_tar.id'''
    data1 = data_tar[data_tar.drugbank_id == id1]
    data2 = data_tar[data_tar.drugbank_id == id2]
    Sim_Tar = Jaccard(data1['id'],data2['id'])
    return Sim_Tar

'''pathway相似性'''
data_path = pd.read_csv(os.path.join(DATA_DIR, 'DB_Pathway.txt'))
data_path.head()
data_path.columns
def Extract_Sim_Path(id1,id2):
    '''data_path.pathway_smpdb_id'''
    data1 = data_path[data_path.drugbank_id == id1]
    data2 = data_path[data_path.drugbank_id == id2]
    Sim_Path = Jaccard(data1['pathway_smpdb_id'],data2['pathway_smpdb_id'])
    return Sim_Path

'''构造药物组合 特征向量1 data_vec1'''
dcom_len = data_dcom.shape[0]
data_vec1 = pd.DataFrame(columns=['Sim_Chem','Sim_Atc1','Sim_Atc2','Sim_Atc3','Sim_Cate','Sim_Tar','Sim_Path','Efficacy','Effect_type'])

for i in range(dcom_len):
    id1 = DB_id(data_dcom.at[i,'COMPONENT_1'])
    id2 = DB_id(data_dcom.at[i,'COMPONENT_2'])
    dcom = ','.join([id1,id2])
    value_chem = Extract_Sim_Chem(id1, id2)
    value_atc_1, value_atc_2, value_atc_3 = Extract_Sim_ATC(id1, id2)
    value_cate = Extract_Sim_Cate(id1, id2)
    value_tar = Extract_Sim_Tar(id1, id2)
    value_path = Extract_Sim_Path(id1, id2)
    efficacy = data_dcom.at[i,'EFFICACY']
    if str(data_dcom.at[i,'EFFECT_TYPE']) != 'nan':
        effect_type = data_dcom.at[i,'EFFECT_TYPE']
    else:
        effect_type = 'No_type'
    data_vec1.loc[dcom] = [value_tar,value_atc_1,value_atc_2,value_atc_3,value_cate,value_tar,value_path,efficacy,effect_type]
    # vec_df = pd.Series(vec,index=data_vec1.columns,name=i)
    # data_vec1 = data_vec1.append(vec_df,ignore_index=True)
print('Finish')
data_vec1.to_csv(os.path.join(RESULTS_DIR, 'Vec_1.csv'))
print('write_in_file')

# data_vec1 = pd.read_csv(os.path.join(RESULTS_DIR, 'Vec_1.csv'),header=0, index_col=0)
df_syn = data_vec1[data_vec1['Effect_type'] == 'Synergistic']
drugs_syn = []
for dc in df_syn.index:
    dc = dc.split(',')
    if dc[0] not in drugs_syn:
        drugs_syn.append(dc[0])
    if dc[1] not in drugs_syn:
        drugs_syn.append(dc[1])

print('构造 35 药物的其他组合')
data_vec1_nonsyn = pd.DataFrame(columns=['Sim_Chem','Sim_Atc1','Sim_Atc2','Sim_Atc3','Sim_Cate','Sim_Tar','Sim_Path','Efficacy','Effect_type'])
for i in range(len(drugs_syn)):
    for j in range(len(drugs_syn)):
        if i < j:
            dcom1 = ','.join([drugs_syn[i], drugs_syn[j]])
            dcom2 = ','.join([drugs_syn[j], drugs_syn[i]])
            if (dcom1 not in df_syn.index) and (dcom2 not in df_syn.index):
                value_chem = Extract_Sim_Chem(drugs_syn[i], drugs_syn[j])
                value_atc_1, value_atc_2, value_atc_3 = Extract_Sim_ATC(drugs_syn[i], drugs_syn[j])
                value_cate = Extract_Sim_Cate(drugs_syn[i], drugs_syn[j])
                value_tar = Extract_Sim_Tar(drugs_syn[i], drugs_syn[j])
                value_path = Extract_Sim_Path(drugs_syn[i], drugs_syn[j])
                efficacy = 'No_eff'
                effect_type = 'Not_syn'
                data_vec1_nonsyn.loc[dcom1] = [value_tar, value_atc_1, value_atc_2, value_atc_3, value_cate, value_tar, value_path, efficacy, effect_type]
    print(i)
print('Finish')

data_vec1_nonsyn.to_csv(os.path.join(RESULTS_DIR, 'Vec_1_nonsyn.csv'))
print('write_in_file_nonsyn')
