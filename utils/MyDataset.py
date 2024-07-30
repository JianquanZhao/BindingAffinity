from  torch.utils.data import Dataset
import torch
import dgl

class MyDataset(Dataset):
    def __init__(self, featurelist, labellsit, namelist):
        self.featurelist = featurelist
        self.labellist = labellsit
        self.namelist = namelist
    
    def __getitem__(self,index):
        return  [self.featurelist[index], self.labellist[index],self.namelist[index]]

    def __len__(self): 
        return len(self.featurelist)

def collate_common(samples):
    allfeatures=list(map(list, zip(*samples))) #把一批图 zip成 列表对象
    allfeatures[0] = dgl.batch(allfeatures[0])
    allfeatures[1] = torch.tensor(allfeatures[1],dtype=torch.float32)
    return allfeatures
    

def gcn_pickfold(featurelist,train_index, test_index):
    train=[]
    test=[]
    for index in train_index:
        train.append(featurelist[index])
    for index in test_index:
        test.append(featurelist[index])
    return  train,test
    
class MyGCNDataset(Dataset):
    def __init__(self, graph_list):
        self.graph_list = graph_list

    def __len__(self):
        return len(self.graph_list)

    def __getitem__(self, idx):
        return self.graph_list[idx]