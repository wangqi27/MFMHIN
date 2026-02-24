#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @software: PyCharm
# @project : 程序2022
# @File    : 文件处理.py
# @Author  : TLX
# @Time    : 2022/11/23 15:10

import pandas as pd
import numpy as np
import os

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data'))

'''DB_Target 添加Gene_ID (PubMed 下载)'''
'''在原文件上更改 '''
data_tar = pd.read_csv(os.path.join(DATA_DIR, 'DB_Target.txt'))
homo_info = pd.read_csv(os.path.join(DATA_DIR, 'Homo_sapiens.txt'),sep = '\t')
# data_tar1 = data_tar.fillna(' ')
# data_tar = data_tar.dropna(axis=0)
# data_tar1 = data_tar['gene_name'].dropna(axis=0)
# tar1 = data_tar[pd.notnull(data_tar['gene_name'])]
# tar2 = data_tar[~pd.isnull(data_tar['gene_name'])]
# tar3 = data_tar.query('gene_name == gene_name')
# tar4 = data_tar.replace(np.nan,'None')
data_tar1 = data_tar.replace(np.nan,'None')
for i in range(data_tar1.shape[0]):
    if data_tar1.at[i,'gene_name'] == None:
        data_tar1.at[i, 'Gene_ID'] = None
    else:
        id_list = homo_info[(homo_info.Symbol == data_tar1.at[i,'gene_name'])].index.tolist()
        if len(id_list) == 0:
            data_tar1.at[i, 'Gene_ID'] = None
        else:
            id0 = id_list[0]
            data_tar1.at[i,'Gene_ID'] = homo_info.at[id0,'GeneID']
print(i)

data_tar1['Gene_ID'] = data_tar1['Gene_ID'].replace(np.nan, 0)
data_tar1['Gene_ID'] = pd.to_numeric(data_tar1['Gene_ID'],downcast='signed')
data_tar1.to_csv(os.path.join(DATA_DIR, 'DB_Target.txt'),index=False)

# data_tar = data_tar.drop('Unnamed: 0',axis=1) # 删除第一行（误存的index）
# data_tar['Gene_ID'] = data_tar['Gene_ID'].apply(str) # 将数据类型（int64） 改为 str
# data_tar.to_csv(os.path.join(DATA_DIR, 'DB_Target.txt'),index=False) # 写入文件，不保存 index

''' int ---> str'''
data_tar['Gene_ID'] = data_tar['Gene_ID'].apply(str)
