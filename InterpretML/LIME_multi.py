from multiprocessing import  Process
from lime.lime_tabular import  LimeTabularExplainer

class LimeTabular_Multi(LimeTabularExplainer):
    def __init__(self,n_jobs = 1,**kwargs):
        self.n_jobs = n_jobs
        self.object_list = [None for i in range(n_jobs)]
        for i in range(n_jobs):
            self.object_list [i] = LimeTabularExplainer.__init__(**kwargs)
    def explain_dataset(self,dataset,labels,**kwargs): #dataset是一个pd.DataFrame
        row_num = dataset.shape[0]
        result = [None for i in range(row_num)]
        for i in range(row_num):
            result[i] = self.object_list[i%self.n_jobs].explain_instance(dataset.iloc[i,:],labels.iloc[i,:],**kwargs)
        return result