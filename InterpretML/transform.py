# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

def init_transform(df,file_path):
    with open(file_path,"r",encoding='utf-8') as f:
        name_list = f.readlines()
    raw_list = []
    for i in range(len(name_list)):
        raw_list.append(name_list[i].split(":")[0])
        name_list[i] = name_list[i].split(":")[1][:-1]
    cnt = 0
    for name in name_list:
        if name != '':
            print("name: ",name)
            print(raw_list[cnt])
            df.rename(columns={raw_list[cnt]:name},inplace=True)
        cnt += 1
from interpret.blackbox import LimeTabular