import torch.nn as nn
import torch
import math
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class FSGNN(nn.Module):
    def __init__(self,nfeat,nlayers,nhidden,nclass,dropout):
        super(FSGNN,self).__init__()
        #self.fc2 = nn.Linear(nhidden*nlayers,nclass)
        self.fc2 = nn.Linear(nfeat+nhidden,nclass)
        self.dropout = dropout
        self.act_fn = nn.ReLU()
        self.nlayers=nlayers
        #self.fc_features=nn.Linear(nfeat,int(nhidden))
        self.fc0=nn.Linear((nfeat+3*nclass),int(nhidden))
        self.fc1 = nn.ModuleList([nn.Linear((nhidden+3*nclass),int(nhidden)) for _ in range(nlayers-1)])
        self.att = nn.Parameter(torch.ones(nlayers))
        self.sm = nn.Softmax(dim=0)
        #self.sm_alpha = nn.Softmax(dim=1)

        self.alphas=nn.Parameter(torch.ones(nlayers,5))

    def forward(self,adj, features,y, layer_norm):
        initial_features=features.clone().detach()# $check this
        layer_list=[]#None#[]
        max_features=None
        #mask = self.sm(self.att)
        pow_adj=adj
        for i in range(self.nlayers):
            mask_alpha=self.sm(self.alphas[i])

            for j in range(i):
              pow_adj=torch.sparse.mm(adj,pow_adj)
            hop_features=torch.sparse.mm(pow_adj,features)#.to(device)
            dist_mat=torch.where(pow_adj.to_dense() > 0, 1., 0.).to_sparse()
            y_out=torch.sparse.mm(dist_mat,y)#.to(device)
            y_in=torch.sparse.mm(torch.transpose(dist_mat.to_dense(), 0, 1).to_sparse(),y)#.to(device)
            y_all=torch.add(y_in.to_dense(),y_out.to_dense())#.to_sparse()#.to(device)
            #for ind in range(4):
            hop_features=torch.mul(mask_alpha[0],hop_features)
            y_out=torch.mul(mask_alpha[1],y_out)
            y_in=torch.mul(mask_alpha[2],y_in)
            y_all=torch.mul(mask_alpha[3],y_all)
            #print("y_all:::",y_all)
            '''list_feat=list()
            list_feat.append(hop_features)
            list_feat.append(y_out)
            list_feat.append(y_in)
            list_feat.append(y_all)'''
            final_feat=torch.cat((torch.cat((torch.cat((hop_features,y_out),dim=1),y_in),dim=1), y_all.to_dense()),dim=1)
            #final_feat=torch.cat((final_feat,y_all.to_dense()),dim=1)
            #final_feat=torch.cat((torch.cat((torch.cat((hop_features,y_out),dim=1),y_in),dim=1),y_all),dim=1)
            '''print("hop_features:::", hop_features.shape)
            print("y_out:::", y_out.shape)
            print("y_in:::", y_in.shape)
            print("y_all:::", y_all.shape)
            del y_out
            del y_in
            del y_all
            del hop_features       '''
            mat=final_feat#torch.cat(final_feat, dim=1)#.to(device)
            #print("mat....",mat.shape)
            '''if layer_norm == True:
                mat = F.normalize(mat,p=2,dim=1) '''            
            if i==0:
              tmp_out = self.fc0(mat)
            else:
              tmp_out = self.fc1[i-1](mat)
            if layer_norm == True:
                tmp_out = F.normalize(tmp_out,p=2,dim=1) 
            
            features=  tmp_out 
            wtd_out=tmp_out#torch.mul(mask[i],tmp_out)
            if i==0:
              max_features=wtd_out

            else:
              max_features=torch.maximum(max_features, wtd_out)

        
        '''list_out = list()
        for ind, mat in enumerate(list_mat):
            tmp_out = self.fc1[ind](mat)
            if layer_norm == True:
                tmp_out = F.normalize(tmp_out,p=2,dim=1)
            tmp_out = torch.mul(mask[ind],tmp_out)

            list_out.append(tmp_out)'''
        layer_list.append(initial_features)
        layer_list.append(max_features)
        final_mat = torch.cat(layer_list, dim=1)
        out = self.act_fn(final_mat)
        out = F.dropout(out,self.dropout,training=self.training)
        out = self.fc2(out)
        #print("self.alphas::",self.alphas)
        #print("self.att::",self.att)

        return F.log_softmax(out, dim=1)



if __name__ == '__main__':
    pass






