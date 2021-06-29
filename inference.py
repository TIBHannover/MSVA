import sys
import torch
from src.msva_models import  MSVA_Gen_auto



cmb = [1,1,1]
feat_input = {"feature_size":365,"L1_out":365,"L2_out":365,"L3_out":512,"pred_out":1,"apperture":250,"dropout1":0.5,"att_dropout1":0.5,"feature_size_1_3":1024,"feature_size_4":365}
model_filename = "model_weights\\summe_random_non_overlap_splits_1_0.5349.tar.pth"
model = MSVA_Gen_auto(feat_input,cmb)
model.load_state_dict(torch.load(model_filename, map_location=lambda storage, loc: storage))

#seq1 =  '' # Feature source 1
#seq2 =  '' # Feature source 1
#seq3 =  '' # Feature source 1
#y, _ = model([seq1,seq2,seq3],seq_len) 
#print('Argument are:', str(sys.argv))