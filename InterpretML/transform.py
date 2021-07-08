# -*- coding: utf-8 -*-
from interpret.blackbox import LimeTabular
import pandas as pd
import numpy as np

class lime_transformed(LimeTabular):
    def __init__(self,
        predict_fn,
        data,
        sampler=None,
        feature_names=None,
        feature_types=None,
        explain_kwargs={},
        n_jobs=1,
        **kwargs):
        LimeTabular.__init__(self,
        predict_fn,
        data,
        sampler=None,
        feature_names=None,
        feature_types=None,
        explain_kwargs={},
        n_jobs=1,
        **kwargs)
    def init_transform(self,file_path):
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
                print(self.feature_names.index(raw_list[cnt]))
                self.feature_names[self.feature_names.index(raw_list[cnt])] = name
            cnt += 1
        #print(self.feature_names)