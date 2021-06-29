

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

class SelfAttention(nn.Module):
    def __init__(self, apperture, ignore_itself=False, input_size=1024, output_size=1024,dropout=0.5): #apperture -1 to ignore
        super(SelfAttention, self).__init__()
        self.apperture = apperture
        self.ignore_itself = ignore_itself
        self.m = input_size
        self.output_size = output_size
        self.K = nn.Linear(in_features=self.m, out_features=self.output_size, bias=False)
        self.Q = nn.Linear(in_features=self.m, out_features=self.output_size, bias=False)
        self.V = nn.Linear(in_features=self.m, out_features=self.output_size, bias=False)
        self.output_linear = nn.Linear(in_features=self.output_size, out_features=self.m, bias=False)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        n = x.shape[0]
        K = self.K(x)  
        Q = self.Q(x)  
        V = self.V(x)
        Q *= 0.06
        logits = torch.matmul(Q, K.transpose(1,0))
        if self.ignore_itself:
            logits[torch.eye(n).byte()] = -float("Inf")
        if self.apperture > 0:
            onesmask = torch.ones(n, n)
            trimask = torch.tril(onesmask, -self.apperture) + torch.triu(onesmask, self.apperture)
            logits[trimask == 1] = -float("Inf")
        att_weights_ = nn.functional.softmax(logits, dim=-1)
        weights = self.dropout(att_weights_)
        y = torch.matmul(V.transpose(1,0), weights).transpose(1,0)
        y = self.output_linear(y)
        return y, att_weights_
		
class MSVA_Gen_auto(nn.Module):  # MSVA auto variation train and learn for different shapes of N input features
    def __init__(self,feat_input,cmb):
        super(MSVA_Gen_auto, self).__init__()
        self.cmb = cmb
        self.feat_input =   feat_input
        self.m = self.feat_input.feature_size # cnn features size
        self.hidden_size = self.m 
        self.apperture= self.feat_input.apperture
        self.att_1_3_size =self.feat_input.feature_size_1_3
        self.att1_3 = SelfAttention(apperture=self.apperture,input_size=self.att_1_3_size, output_size=self.att_1_3_size,dropout=self.feat_input.att_dropout1)
        self.ka1_3 = nn.Linear(in_features=self.att_1_3_size , out_features=self.feat_input.L1_out)
        self.kb = nn.Linear(in_features=self.ka1_3.out_features, out_features=self.feat_input.L2_out)
        self.kc = nn.Linear(in_features=self.kb.out_features, out_features=self.feat_input.L3_out)
        self.kd = nn.Linear(in_features=self.kc.out_features, out_features=self.feat_input.pred_out)
        self.sig = nn.Sigmoid()
        self.relu = nn.ReLU()
#        self.dropout= nn.Dropout(self.feat_input["dropout1"])
        self.dropout= nn.Dropout(self.feat_input.dropout1)
        self.softmax = nn.Softmax(dim=0)
        self.layer_norm_y_1_3 = LayerNorm(self.att_1_3_size)
        self.layer_norm_y_4 = LayerNorm(self.att_1_3_size)
        self.layer_norm_kc = LayerNorm(self.kc.out_features)
        self.layer_norm_kd = LayerNorm(self.kd.out_features)
    def forward(self,x_list, seq_len):
        y_out_ls = []
        att_weights_ = []
        for i in range(len(x_list)):
            x = x_list[i].view(-1, x_list[i].shape[2])
            y, att_weights = self.att1_3(x)
            att_weights_  = att_weights
            y = y + x
            y = self.dropout(y)    
            y = self.layer_norm_y_1_3(y)
            y_out_ls.append(y)
        y_out_ls_filter = []
        for i in range(0,len(y_out_ls)):
            if(self.cmb[i]):
                y_out_ls_filter.append(y_out_ls[i])
        y_out = y_out_ls_filter[0]
        for i in range(1,len(y_out_ls)):
            y_out = y_out + y_out_ls_filter[i] 
        # Frame level importance score regression
        y = y_out
        y = self.ka1_3(y)# Two layer NN
        y = self.kb(y)
        y = self.kc(y) 
        y = self.relu(y)
        y = self.dropout(y)
        y = self.layer_norm_kc(y)
        y = self.kd(y)
        y = self.sig(y)
        y = y.view(1, -1)
        return y, att_weights_
		
class MSVA_auto(nn.Module):  # MSVA auto variation train and learn for different shapes of N input features
    def __init__(self,feat_input,cmb):
        super(MSVA_auto, self).__init__()
        self.cmb = cmb
        self.feat_input =   feat_input
        self.m = self.feat_input["feature_size"] # cnn features size
        self.hidden_size = self.m 
        self.apperture= self.feat_input["apperture"]
        self.att_1_3_size =self.feat_input["feature_size_1_3"]
        self.att1_3 = SelfAttention(apperture=self.apperture,input_size=self.att_1_3_size, output_size=self.att_1_3_size,dropout=self.feat_input["att_dropout1"])
        
        self.ka1_3 = nn.Linear(in_features=self.att_1_3_size , out_features=self.feat_input["L1_out"])
        self.kb = nn.Linear(in_features=self.ka1_3.out_features, out_features=self.feat_input["L2_out"])
        self.kc = nn.Linear(in_features=self.kb.out_features, out_features=self.feat_input["L3_out"])
        self.kd = nn.Linear(in_features=self.kc.out_features, out_features=self.feat_input["pred_out"])
        self.sig = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.dropout= nn.Dropout(self.feat_input["dropout1"])
        self.softmax = nn.Softmax(dim=0)
        self.layer_norm_y_1_3 = LayerNorm(self.att_1_3_size)
        self.layer_norm_y_4 = LayerNorm(self.att_1_3_size)
        self.layer_norm_kc = LayerNorm(self.kc.out_features)
        self.layer_norm_kd = LayerNorm(self.kd.out_features)
    def forward(self, x1,x2,x3, seq_len):
        m = x1.shape[2] # Feature size
        # Place the video frames to the batch dimension to allow for batch arithm. operations.
        # Assumes input batch size = 1.
        x1 = x1.view(-1, x1.shape[2] )
        x2 = x2.view(-1, x2.shape[2] )
        x3 = x3.view(-1, x3.shape[2] )
        y1, att_weights_1 = self.att1_3(x1)
        y2, att_weights_2 = self.att1_3(x2)
        y3, att_weights_3 = self.att1_3(x3)
        att_weights_ = att_weights_1
        y1 = y1 + x1
        y2 = y2 + x2
        y3 = y3 + x3
        y1 = self.dropout(y1)
        y2 = self.dropout(y2)
        y3 = self.dropout(y3)
        y1 = self.layer_norm_y_1_3(y1)
        y2 = self.layer_norm_y_1_3(y2)
        y3 = self.layer_norm_y_1_3(y3)
        # ------------------------------------------ before regressor intermediate fusion
        if(self.cmb[0] and  self.cmb[1] and self.cmb[2]):
            y= y1 + y2 + y3
        if(self.cmb[0]==0 and  self.cmb[1] and self.cmb[2]):
            y=  y2 + y3
        if(self.cmb[0] and  self.cmb[1]==0 and self.cmb[2]):
            y=  y1 + y3
        if(self.cmb[0] and  self.cmb[1] and self.cmb[2]==0):
            y=  y1 + y2
        # Frame level importance score regression
        y = self.ka1_3(y)# Two layer NN
        y = self.kb(y)
        y = self.kc(y) 
        y = self.relu(y)
        y = self.dropout(y)
        y = self.layer_norm_kc(y)
        y = self.kd(y)
        y = self.sig(y)
        y = y.view(1, -1)
        return y, att_weights_
    

class MSVABiLstm(nn.Module):
    def __init__(self, apperture,input_feature=1024,hidden_size=1024):
        super(MSVABiLstm, self).__init__()
        self.m = input_feature 
        self.hidden_size =hidden_size
        self.apperture=apperture

        self.att = SelfAttention(apperture=self.apperture,input_size=self.m, output_size=self.m)
        self.ka = nn.Linear(in_features=self.m, out_features=self.hidden_size )
        self.kb = nn.Linear(in_features=self.ka.out_features, out_features=self.hidden_size )
        self.kc = nn.Linear(in_features=self.kb.out_features, out_features=self.hidden_size )
        self.kd = nn.Linear(in_features=self.kc.out_features, out_features=1)
        self.lstm = nn.LSTM(input_size=self.m,
                            hidden_size=self.hidden_size ,
                            num_layers=self.lstm_layer , 
                            dropout = self.lstm_drop,
                            bidirectional=False)
        self.sig = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.drop50 = nn.Dropout(0.5)
        self.drop20 = nn.Dropout(0.2)
        self.softmax = nn.Softmax(dim=0)
        self.layer_norm_y = LayerNorm(self.m)
        self.layer_norm_ka = LayerNorm(self.ka.out_features)
        self.layer_norm_kd = LayerNorm(self.kd.out_features)
    def forward(self, x1,x2,x3, seq_len):
        m = x1.shape[2] # Feature size
        x1 = x1.view(-1, m)
        x2 = x2.view(-1, m)
        x3 = x3.view(-1, m)
        y1, att_weights_1 = self.att(x1)
        y2, att_weights_2 = self.att(x2)
        y3, att_weights_3 = self.att(x3)
        att_weights_ = att_weights_1
        y1 = y1 + x1
        y2 = y2 + x2
        y3 = y3 + x3
        y1 = self.drop50(y1)
        y2 = self.drop50(y2)
        y3 = self.drop50(y3)
        y1 = self.layer_norm_y(y1)
        y2 = self.layer_norm_y(y2)
        y3 = self.layer_norm_y(y3)
        # ------------------------------------------ before regressor intermediate fusion
        y= y1 + y2 + y3
        y=torch.unsqueeze(y, 1)
        lstm_out, (h_n, c_n) = self.lstm(y)
        y = self.ka(lstm_out)
        y = self.relu(y)
        y = self.drop50(y)
        y = self.layer_norm_ka(y)
        y = self.kd(y)
        y = self.sig(y)
        y = y.view(1, -1)
        return y, att_weights_

class MSVA_Gen(nn.Module):
    def __init__(self, apperture,input_feature=1024,hidden_size=1024):
        super(MSVA_Gen, self).__init__()
        self.m = input_feature 
        self.hidden_size =hidden_size
        self.apperture=apperture

        self.att = SelfAttention(apperture=self.apperture,input_size=self.m, output_size=self.m)
        self.ka = nn.Linear(in_features=self.m, out_features=1024)
        self.kb = nn.Linear(in_features=self.ka.out_features, out_features=1024)
        self.kc = nn.Linear(in_features=self.kb.out_features, out_features=1024)
        self.kd = nn.Linear(in_features=self.kc.out_features, out_features=1)
        self.sig = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.softmax = nn.Softmax(dim=0)
        self.layer_norm_y = LayerNorm(self.m)
        self.layer_norm_ka = LayerNorm(self.ka.out_features)
        self.layer_norm_kd = LayerNorm(self.kd.out_features)
    def forward(self,x_list, seq_len):
        y_out_ls = []
        att_weights_ = []
        for i in range(len(x_list)):
            x = x_list[i].view(-1, x_list[i].shape[2])
            y, att_weights = self.att(x)
            att_weights_  = att_weights
            y = y + x
            y = self.dropout(y)    
            y = self.layer_norm_y(y)
            y_out_ls.append(y)
        y_out = y_out_ls[0]
        for i in range(1,len(y_out_ls)):
            y_out = y_out + y_out_ls[i]    
        y = self.ka(y_out)
        y = self.relu(y)
        y = self.drop50(y)
        y = self.layer_norm_ka(y)
        y = self.kd(y)
        y = self.sig(y)
        y = y.view(1, -1)
        return y, att_weights_
		

if __name__ == "__main__":
    pass