#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/10/9 17:17
# @Author  : TLX
# @File    : 提取药物信息.py.py
# @software: PyCharm
# ============================================
# @Introduction:
# Step1: 药物组合 --- > 单个药物
#        用到的文件：Data_Prep/DCDB_945.txt
#        生成文件： Data_Prep/DCDB_DCC.txt
# ============================================

import numpy as np
import pandas as pd
import os

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data'))
RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../results'))
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

filename = os.path.join(DATA_DIR, 'DCDB_945.txt')
dcdb = pd.read_csv(filename,sep='\t')
# 去空格，全部小写
def qukongge1(hang):
    return hang[' COMPONENT 1'].strip()
def qukongge2(hang):
    return hang[' COMPONENT 2'].strip()
dcdb[' COMPONENT 1'] =  dcdb.apply(qukongge1,axis=1)
dcdb[' COMPONENT 2'] =  dcdb.apply(qukongge2,axis=1)
dcdb[' COMPONENT 1'] = dcdb[' COMPONENT 1'].str.lower()
dcdb[' COMPONENT 2'] = dcdb[' COMPONENT 2'].str.lower()
# 统计信息查询
# dcdb.columns
# dcdb["EFFICACY"].value_counts()
# dcdb["EFFECT_TYPE"].value_counts()

# 提出dcdb 中所有的药物 --- 815
dcdb_drugs = set(dcdb[' COMPONENT 1']).union(set(dcdb[' COMPONENT 2']))
dcdb_drugs = [ drug for drug in dcdb_drugs ] # set 转list，并去除开头空格
#----写入药物文件 DCDB_drug_815.txt ----
dcdb_drugs_0 = pd.DataFrame(dcdb_drugs)
dcdb_drugs_0.to_csv(os.path.join(RESULTS_DIR, 'DCDB_drug_815.txt'), index=False)

'''提取存在DB_ID的药物  815 --> 604 ️'''
# 药物名作为index， dcdb_drugs_1: 在DrugBank中的药物Dataframe 593
db_id = pd.read_csv(os.path.join(DATA_DIR, 'DB_Name.txt'), sep=',')
db_id['name'] = db_id['name'].str.lower()
dcdb_drugs_1 = db_id[db_id['name'].isin(dcdb_drugs)]

'''提取存在 全部特征 的药物  604 --> 182 ️'''

db_smile = pd.read_csv(os.path.join(DATA_DIR, 'DB_SMILES.csv'),sep=',')
db_atc = pd.read_csv(os.path.join(DATA_DIR, 'DB_ATC.txt'), sep=',')
db_target = pd.read_csv(os.path.join(DATA_DIR, 'DB_Target.txt'), sep=',')
db_pathway = pd.read_csv(os.path.join(DATA_DIR, 'DB_Pathway.txt'),sep=',')
db_category = pd.read_csv(os.path.join(DATA_DIR, 'DB_Category.txt'), sep=',')

dcdb_drugs_2 = dcdb_drugs_1 # 604
dcdb_drugs_2 = dcdb_drugs_2[dcdb_drugs_2['drugbank_id'].isin(db_smile['drugbank_id'])] # 550
dcdb_drugs_2 = dcdb_drugs_2[dcdb_drugs_2['drugbank_id'].isin(db_atc['drugbank_id'])] # 494
dcdb_drugs_2 = dcdb_drugs_2[dcdb_drugs_2['drugbank_id'].isin(db_category['drugbank_id'])]  # 494
dcdb_drugs_2 = dcdb_drugs_2[dcdb_drugs_2['drugbank_id'].isin(db_target['drugbank_id'])]  # 482
dcdb_drugs_2 = dcdb_drugs_2[dcdb_drugs_2['drugbank_id'].isin(db_pathway['drugbank_id'])]  # 182


dcdb_drugs_2.to_csv(os.path.join(RESULTS_DIR, 'DCDB_drug_182.txt'), index= False, header= False)

'''存在DB_ID, 并可搜索到所有特征的组合药物 945 ---> 134 (药物数 128 )'''
dcdb1 = dcdb # 945
dcdb1 = dcdb1[dcdb1[' COMPONENT 1'].isin(dcdb_drugs_2['name'])] # 306
dcdb1 = dcdb1[dcdb1[' COMPONENT 2'].isin(dcdb_drugs_2['name'])] # 134
dcdb1.to_csv(os.path.join(RESULTS_DIR, 'DCDB_134.txt'), index= False)
dcdb1["EFFICACY"].value_counts()
dcdb1["EFFECT_TYPE"].value_counts()

dd = set(dcdb1[' COMPONENT 2']).union(set(dcdb1[' COMPONENT 1'])) # 128
size(dd)



'''统计各个特征下 药物名称'''

drug_smile = dcdb_drugs_1[dcdb_drugs_1['drugbank_id'].isin(db_smile['drugbank_id'])] # 550
drug_atc = dcdb_drugs_1[dcdb_drugs_1['drugbank_id'].isin(db_atc['drugbank_id'])] # 541
drug_category = dcdb_drugs_1[dcdb_drugs_1['drugbank_id'].isin(db_category['drugbank_id'])]  # 599
drug_target = dcdb_drugs_1[dcdb_drugs_1['drugbank_id'].isin(db_target['drugbank_id'])]  # 570
drug_pathway = dcdb_drugs_1[dcdb_drugs_1['drugbank_id'].isin(db_pathway['drugbank_id'])]  # 191

arr = np.zeros((604,5))
Venn_data = pd.DataFrame(arr,columns=['Smile','Atc_code','Category','Target','Pathway'],index=dcdb_drugs_1['drugbank_id'])

for drugid in Venn_data.index:
    if drugid in db_smile['drugbank_id'].values:
        Venn_data.loc[drugid,'Smile'] = 1.0
    if drugid in db_atc['drugbank_id'].values:
        Venn_data.loc[drugid,'Atc_code'] = 1.0
    if drugid in db_category['drugbank_id'].values:
        Venn_data.loc[drugid,'Category'] = 1.0
    if drugid in db_target['drugbank_id'].values:
        Venn_data.loc[drugid,'Target'] = 1.0
    if drugid in db_pathway['drugbank_id'].values:
        Venn_data.loc[drugid,'Pathway'] = 1.0

Venn_data.to_csv(os.path.join(RESULTS_DIR, 'Venn_604.txt'))





























