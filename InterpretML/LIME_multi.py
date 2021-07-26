from multiprocessing import  Process
from lime.lime_tabular import  LimeTabularExplainer
from multiprocessing import Manager
'''
class LimeTabular_Multi(LimeTabularExplainer):
    def __init__(self,predict_fn,n_jobs = 1,**kwargs):
        self.predict_fn = predict_fn
        self.n_jobs = n_jobs
        self.object_list = [None for i in range(n_jobs)]
        for i in range(n_jobs):
            self.object_list[i] = LimeTabularExplainer.__init__(**kwargs)
    def explain_by_each(self,dataset,ith):
        explainer = self.object_list[ith]
        index_list = []
        for i in range(dataset.shape[0]):
            if i%self.n_jobs == ith:
                index_list.append(i)
        dataset_part = dataset.iloc[index_list,:]
        result = []
        for i in index_list:
            result.append(result.)
    def explain_dataset(self,dataset,labels,**kwargs): #dataset是一个pd.DataFrame
        row_num = dataset.shape[0]
        result = [None for i in range(row_num)]
        for i in range(row_num):
            result[i] = self.object_list[i%self.n_jobs].explain_instance(dataset.iloc[i,:],self.predict_fn)
        return result
'''
import sys
sys.path.append("..")
from interpret import set_visualize_provider
from interpret.provider import InlineProvider
set_visualize_provider(InlineProvider())
from evaluate import IoU_value
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from interpret import show
from interpret.blackbox import LimeTabular
from interpret.blackbox import ShapKernel
import pandas as pd
import numpy as np
import xgboost
from interpret.glassbox import ExplainableBoostingClassifier
plt.rcParams['font.sans-serif'] = ['KaiTi'] # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题
pd.options.display.max_columns = None
class LimeTabular_Multi(LimeTabularExplainer):
    def __init__(
        self,
        training_data,
        mode = "classification",
        training_labels = None,
        feature_names = None,
        categorical_features = None,
        categorical_names = None,
        kernel_width = None,
        kernel = None,
        verbose = False,
        class_names = None,
        feature_selection = 'auto',
        discretize_continuous = True,
        discretizer = 'quartile',
        sample_around_instance = False,
        random_state = None,
        training_data_stats = None):
        LimeTabularExplainer.__init__(self,
                 training_data,
                 mode="classification",
                 training_labels=None,
                 feature_names=None,
                 categorical_features=None,
                 categorical_names=None,
                 kernel_width=None,
                 kernel=None,
                 verbose=False,
                 class_names=None,
                 feature_selection='auto',
                 discretize_continuous=True,
                 discretizer='quartile',
                 sample_around_instance=False,
                 random_state=None,
                 training_data_stats=None)
        self.result =[]
        self.index = 0
    def get_result(self,index,**kwargs):
        self.result[index] = self.explain_instance(**kwargs)
    def explain_multi(self,n_jobs,predict_fn):
        self.result = [None for _ in range(self.dataset.shape[0])]
        print("use "+ str(n_jobs)+" process to explain " +str(len(self.result))+" row")
        process_list = []
        if len(process_list) == 0:
            for i in range(n_jobs):
                p = Process(target=self.get_result, args=(self.dataset.iloc[self.index,:],predict_fn))  # 实例化进程对象
                p.start()
                process_list.append(p)
                self.index = self.index + 1
                if self.index == self.dataset.shape[0]:
                    print("处理完成")
                    break
        else:
            for i in process_list:
                i = Process(target=self.get_result, args=(self.dataset.iloc[self.index,:],predict_fn))  # 实例化进程对象
                i.start()
                self.index = self.index + 1
        for process in process_list:
            process.join()
    def explain_dataset(self,dataset,n_jobs,predict_fn):
        self.dataset = dataset.copy()
        while(self.index <self.dataset.shape[0]):
            self.explain_multi(n_jobs=n_jobs,predict_fn=predict_fn)


df = pd.read_csv("../train/true_dataset/lending club/lending_club_processed.csv",index_col=0)
df.drop(columns="target",inplace=True)
target_list = ["loan_amnt","int_rate","emp_length","grade","home_ownership","total_bc_limit","installment"]
def compare(target_list,list_10):
    set_result = set(target_list) & set(list_10)
    return len(set_result)
def normalize(x):
    x_max = x.max()
    x_min = x.min()
    if x.max() == x.min():
        return x
    else:
        result = (x-x_min)/(x_max-x_min)
        return result
df_norm = df
df_norm["target"] = 0.01 * df_norm["loan_amnt"] + 10 * df_norm["int_rate"]+ 5* df_norm["emp_length"]+2*df_norm["grade"]+10*df_norm["home_ownership"]+0.5*df_norm["installment"]+ 0.005*df_norm["total_bc_limit"]#-2*df_norm["mths_since_last_delinq"]#
df_norm["target"] = (normalize(df_norm["target"])>0.4).astype(int)
#df_norm["target"].value_counts()
X,y = df.iloc[:,:-1],df.iloc[:,-1]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)
xgb_clf = xgboost.XGBClassifier(use_label_encoder=False,eval_metric=['logloss','auc','error'])
xgb_clf.fit(X_train,y_train)
#lime_explainer = LimeTabular_Multi(X_train.values,feature_names=X_train.columns,discretize_continuous=True,discretizer="quartile",verbose=True,mode = "classification")
#lime_explainer.explain_dataset(X_train,10,xgb_clf.predict_proba)

#exp_list = [None for _ in range(X_test.shape[0])]
#exp_list = [None for _ in range(15)]
process_list = []
n_jobs = 10
def explain(index_num,X_train,X_test,predict_fn,exp_list):
    limeexplainer = LimeTabularExplainer(X_train.values, feature_names=X_train.columns, discretize_continuous=True,
                                         discretizer="quartile", verbose=True, mode="classification")
    exp = limeexplainer.explain_instance(X_test.iloc[index_num], predict_fn, num_features=10)
    exp_list[index_num] = exp
    #print(exp_list[index_num].local_exp[1])
    print(exp_list)
    #print("index_num: "+str(index_num))
if __name__ == '__main__':
    n_jobs = input("进程数： ")
    index = 0
    manager = Manager()
    exp_list = manager.list()
    for i in range(X_test.shape[0]):
        exp_list.append(None)
    #while(index < X_test.shape[0]):
    while(True):
        if len(process_list) ==0:
            for i in range(n_jobs):
                p = Process(target=explain,args=(index,X_train,X_test,xgb_clf.predict_proba,exp_list))
                p.start()
                index += 1
                process_list.append(p)
                #if index == X_test.shape[0]:
                if index ==X_test.shape[0]:
                    break
        else:
            for i in range(len(process_list)):
                process_list[i] = Process(target=explain,args=(index,X_train,X_test,xgb_clf.predict_proba,exp_list))
                process_list[i].start()
                index += 1
                #if index == X_test.shape[0]:
                if index ==X_test.shape[0]:
                    break
        for i in process_list:
            i.join()
        print("已处理 " + str(index) + " 行数据")
        if index == X_test.shape[0]:
            break
    target_list = ["loan_amnt", "int_rate", "emp_length", "grade", "home_ownership", "total_bc_limit", "installment"]
    #print(exp_list[0].local_exp[1])

    def compare(target_list, list_10):
        set_result = set(target_list) & set(list_10)
        return len(set_result)
    def extract_from_exp(exp, columns):
        exp_local = exp.local_exp[1]
        result_list = []
        for i in range(len(exp_local)):
            result_list.append(columns[exp_local[i][0]])
        return result_list
    count = 0
    for i in range(X_test.shape[0]):
    #for i in range(15):
        exp = exp_list[i]
        result_list = extract_from_exp(exp, X_train.columns)
        count += compare(target_list, result_list)
    count /= X_test.shape[0]
    print(count)
