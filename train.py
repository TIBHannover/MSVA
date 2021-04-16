
import numpy as np
import glob
import random
import argparse
import h5py
import json
import torch
import torch.nn.init as init
from torchvision import transforms

import sys
from src.sys_utils import *
from src.vsum_tools import  *
from src.msva_models import  *
import os
import cv2
import pickle
from scipy.stats import kendalltau, spearmanr
from scipy.stats import rankdata


class obj(object):
    def __init__(self, d):
        for a, b in d.items():
            if isinstance(b, (list, tuple)):
               setattr(self, a, [obj(x) if isinstance(x, dict) else x for x in b])
            else:
               setattr(self, a, obj(b) if isinstance(b, dict) else b)
               
def weights_init(m):
    classname = m.__class__.__name__
    if classname == 'Linear':
        init.xavier_uniform_(m.weight, gain=np.sqrt(2.0))
        if m.bias is not None:
            init.constant_(m.bias, 0.1)
def parse_arguments(args):
    master_arg = "-params"
    try:
        if(master_arg in args and args.index(master_arg)+1<len(args)):
            json_file  = args[args.index(master_arg)+1]
#            json_params = json.loads(open(json_file,"r").read())
            json_params = eval(open(json_file,"r").read())
            return json_params
    except:
        print("Something went wrong when loading the parameters, Kindly check input carefully!!!")

def parse_splits_filename(splits_filename):
    spath, sfname = os.path.split(splits_filename)
    sfname, _ = os.path.splitext(sfname)
    dataset_name = sfname.split('_')[0]  # Get dataset name e.g. tvsum
    dataset_type = sfname.split('_')[1]  # augmentation type e.g. aug
    if dataset_type == 'splits':
        dataset_type = ''
    with open(splits_filename, 'r') as sf:
        splits = json.load(sf)
    return dataset_name, dataset_type, splits

def lookup_weights_splits_file(path, dataset_name, dataset_type, split_id):
    dataset_type_str = '' if dataset_type == '' else dataset_type + '_'
#    weights_filename = path + '/models/{}_{}splits_{}_*.tar.pth'.format(dataset_name, dataset_type_str, split_id)
    weights_filename = os.path.join(path,"models",dataset_name+"_"+dataset_type_str+""+"splits_"+str(split_id)+"_*.tar.pth")
    weights_filename = glob.glob(weights_filename)
    if len(weights_filename) == 0:
        print("Couldn't find model weights: ", weights_filename)
        return ''
    # Get the first weights file in the dir
    weights_filename = weights_filename[0]
#    splits_file = path + '/splits/{}_{}splits.json'.format(dataset_name, dataset_type_str)
    splits_file = os.path.join(path, "splits"+dataset_name+"_"+dataset_type_str+"splits.json")
    return weights_filename, splits_file

class TrainingObj:
    def __init__(self, params_obj):
        self.params_obj = params_obj
        self.model = None
        self.log_file = None
        self.verbose = params_obj.verbose
        self.dataset_name=""
        self.feat_input=[]
    def fix_keys(self, keys, dataset_name = None):
        """
        :param keys:
        :return:
        """
        if len(self.object_features) == 1:
            dataset_name = next(iter(self.object_features))
        keys_out = []
        for key in keys:
            t = key.split('/')
            if len(t) != 2:
#                assert dataset_name is not None, "ERROR dataset name in some keys is missing but there are multiple dataset {} to choose from".format(len(self.object_features))
                assert dataset_name is not None, "ERROR dataset name in some keys is missing but there are multiple dataset {} to choose from"+str(len(self.object_features))

                key_name = dataset_name+'/'+key
                keys_out.append(key_name)
            else:
                keys_out.append(key)
        return keys_out

    def load_object_features(self, object_features = None):
        if object_features is None:
            object_features = self.params_obj.object_features
        object_features_dict = {}
        for dataset in object_features:
            _, base_filename = os.path.split(dataset)
            base_filename, _ = os.path.splitext(base_filename)
            print("Loading:", dataset)
            object_features_dict[base_filename] = h5py.File(dataset, 'r')

        self.object_features = object_features_dict
        return object_features_dict

    def load_split_file(self, splits_file):
        self.dataset_name, self.dataset_type, self.splits = parse_splits_filename(splits_file)
        n_folds = len(self.splits)
        self.split_file = splits_file
        print("Loading splits from: ",splits_file)
        return n_folds
    def get_dataset_by_name(self, dataset_name):
        for d in self.params_obj.object_features:
            if dataset_name in d:
                return [d]
        return None
    def select_split(self, split_id):
        print("Selecting split: ",split_id)
        self.split_id = split_id
        n_folds = len(self.splits)
#        assert self.split_id < n_folds, "split_id (got {}) exceeds {}".format(self.split_id, n_folds)
        assert self.split_id < n_folds, "split_id (got {}) exceeds {}"+str(self.split_id)+"_"+str(n_folds)
        split = self.splits[self.split_id]
        self.train_keys = split['train_keys']
        self.test_keys = split['test_keys']
        dataset_filename = self.get_dataset_by_name(self.dataset_name)[0]
        _,dataset_filename = os.path.split(dataset_filename)
        dataset_filename,_ = os.path.splitext(dataset_filename)
        self.train_keys = self.fix_keys(self.train_keys, dataset_filename)
        self.test_keys = self.fix_keys(self.test_keys, dataset_filename)
        return

    def load_model(self, model_filename):
        self.model.load_state_dict(torch.load(model_filename, map_location=lambda storage, loc: storage))
        return

    def initialize(self,feat_input,cmb, cuda_device=None):
        rnd_seed = 12345
#        rnd_seed = random.randint(1,123457)  # for the random seed and have different weights each time
        print("randomSeed: ",rnd_seed)
        random.seed(rnd_seed)
        np.random.seed(rnd_seed)
        torch.manual_seed(rnd_seed)
        self.feat_input = feat_input
        self.model = MSVA_Gen_auto(self.feat_input,cmb) # imported drom msva_models , checck other variations to explore the effect of other models inputs and training techniques
        self.model.eval()
        self.model.apply(weights_init)
        cuda_device = cuda_device or self.params_obj.cuda_device
        if self.params_obj.use_cuda:
            print("Setting CUDA device: ",cuda_device)
            torch.cuda.set_device(cuda_device)
            torch.cuda.manual_seed(rnd_seed)
        if self.params_obj.use_cuda:
            self.model.cuda()
        return

    def get_data(self, key):
        key_parts = key.split('/')
        assert len(key_parts) == 2, "ERROR. Wrong key name: "+key
        dataset, key = key_parts
        return self.object_features[dataset][key]

    def lookup_weights_file(self, data_path):
        dataset_type_str = '' if self.dataset_type == '' else self.dataset_type + '_'
#        weights_filename = data_path + '/models/{}_{}splits_{}_*.tar.pth'.format(self.dataset_name, dataset_type_str, self.split_id)
#        weights_filename = data_path + '/models/{}_{}splits_{}_*.tar.pth'.format(self.dataset_name, dataset_type_str, self.split_id)
        weights_filename = os.path.join(data_path,"models",self.dataset_name+"_"+dataset_type_str+""+"splits_"+str(self.split_id)+"_*.tar.pth")
        weights_filename = glob.glob(weights_filename)
        if len(weights_filename) == 0:
            print("Couldn't find model weights: ", weights_filename)
            return ''

        weights_filename = weights_filename[0]
#        splits_file = data_path + '/splits/{}_{}splits.json'.format(self.dataset_name, dataset_type_str)
        splits_file = os.path.join(data_path, "splits"+self.dataset_name+"_"+dataset_type_str+"splits.json")
        return weights_filename, splits_file

    def early_fusion(self,features,method,stack):  # early fusion variation for multiple features
        final_stack=[]
        if(stack=="v"):
            final_stack=np.vstack(features)
        if(stack=="h"):
            final_stack=np.hstack(features)
        final_stack_rsp=final_stack.reshape((features[0].shape[0],len(features),features[0].shape[1]))
        if(method=="min"):
            return final_stack_rsp.min(axis=1)
        if(method=="max"):
            return final_stack_rsp.max(axis=1)
        if(method=="mean"):
            return final_stack_rsp.mean(axis=1)
    def getIdxInNames(self,files_feature,key):
        idx=int(key.split("_")[-1])-1
        return idx
    def getIdxInNamesSumme(self,files_feature,key):
        for i in range(len(files_feature)):
            if(bool(np.array(files_feature[i].split("/")[-1][:-4])==key)):
                return i
    def train(self,cmb, output_dir='results'):
        print("Initializing model and optimizer for Feature Combination: ",cmb)
        self.model.train()
        criterion = nn.MSELoss()
        if self.params_obj.use_cuda:
            criterion = criterion.cuda()
        parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = torch.optim.Adam(parameters, lr=self.params_obj.lr[0], weight_decay=self.params_obj.weight_decay)
        max_val_fscore = 0
        maxkt=0
        maxsp=0
        max_val_fscore_epoch = 0
        train_keys = self.train_keys[:]
        lr = self.params_obj.lr[0]
        sameCount=0
        max_val_fscoreLs=[]
        sameAccStopThres=50
        train_val_loss_score=[]
        mx_video_scores=[]
        mx_test_preds=[]
        print("Starting training...")
        for epoch in range(self.params_obj.epochs_max):
            print("Epoch ",epoch," out of total: ",self.params_obj.epochs_max)
            self.model.train()
            avg_loss = []
            random.shuffle(train_keys)
            path_feature_rgb = os.path.join(self.params_obj.kinetic_features,self.dataset_name,"RGB","features")
            path_feature_flow = os.path.join(self.params_obj.kinetic_features,self.dataset_name,"FLOW","features")
            base_add_targets = os.path.join(self.params_obj.kinetic_features,self.dataset_name,"targets")
            files_feature_rgb  = glob.glob(os.path.join(path_feature_rgb,"*.npy"))
            files_feature_flow  = glob.glob(os.path.join(path_feature_flow,"*.npy"))
            files_feature_rgb.sort()
            files_feature_flow.sort()
            for i, key in enumerate(train_keys):
                dataset = self.get_data(key)
                seq1 = dataset['features'][...]
                if(self.dataset_name=="tvsum"):
                    idx=self.getIdxInNames(files_feature_rgb,key)
                if(self.dataset_name=="summe"):
                    idx=self.getIdxInNamesSumme(files_feature_rgb,dataset['video_name'][...].astype(str))
                seq2 = np.load(files_feature_rgb[idx])
                seq3 = np.load(files_feature_flow[idx])
                maxShape=np.max([seq1.shape[0],seq2.shape[0],seq3.shape[0]])
                minShape=np.min([seq1.shape[0],seq2.shape[0],seq3.shape[0]])
                fileName = files_feature_rgb[idx].split(os.path.sep)[-1].split('.')[0]
#                visFeaturesSize = self.feat_input["feature_size"]
                visFeaturesSize = self.feat_input.feature_size
                target = np.load(os.path.join(base_add_targets,fileName+".npy"))
                if(self.params_obj.sample_technique=="up"):
                    seq1 = cv2.resize(seq1, (seq1.shape[1],maxShape), interpolation = cv2.INTER_AREA)
                    seq2 = cv2.resize(seq2, (seq2.shape[1],maxShape), interpolation = cv2.INTER_AREA)
                    seq3 = cv2.resize(seq3, (seq3.shape[1],maxShape), interpolation = cv2.INTER_AREA)
                    target =  np.load(os.path.join(base_add_targets,fileName+".npy"))
                if(self.params_obj.sample_technique=="sub"):
                    seq1 = cv2.resize(seq1, (seq1.shape[1],minShape), interpolation = cv2.INTER_AREA)
                    seq2 = cv2.resize(seq2, (seq2.shape[1],minShape), interpolation = cv2.INTER_AREA)
                    seq3 = cv2.resize(seq3, (seq3.shape[1],minShape), interpolation = cv2.INTER_AREA)
                    target = dataset['gtscore'][...]
                features=[]
                if(cmb[0]):
                    features.append(seq1)
                if(cmb[1]):
                    features.append(seq2)
                if(cmb[2]):
                    features.append(seq3)
                if(self.params_obj.fusion_technique=="early"):
                    seq=self.early_fusion(features,method=self.params_obj.method,stack=self.params_obj.stack)
                    seq_len = seq.shape[1]
                    y, _ = self.model(seq,seq_len)
                else:
                    seq1 = torch.from_numpy(seq1).unsqueeze(0)
                    seq2 = torch.from_numpy(seq2).unsqueeze(0)
                    seq3= torch.from_numpy(seq3).unsqueeze(0)
                    target = torch.from_numpy(target).unsqueeze(0)
                    # Normalize frame scores
                    target -= target.min()
                    target=np.true_divide(target, target.max())
                    if self.params_obj.use_cuda:
                        target = target.float().cuda()
                        seq1 = seq1.float().cuda()
                        seq2 = seq2.float().cuda()
                        seq3 = seq3.float().cuda()
                    seq_len = seq1.shape[1]
                    if(self.params_obj.fusion_technique=="inter"):
                        y, _ = self.model([seq1,seq2,seq3],seq_len) # for three source of feature Xo, Xr, Xf
                    elif(self.params_obj.fusion_technique=="late"):
                        y1, _ = self.model(seq1,seq_len)
                        y2, _ = self.model(seq2,seq_len)
                        y3, _ = self.model(seq3,seq_len)
                        y = y1 + y2 + y3
                loss_att = 0
                loss = criterion(y, target)
                loss = loss + loss_att
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                avg_loss.append([float(loss), float(loss_att)])
            val_fscore, video_scores,kt,sp,test_loss,test_preds  = self.eval_function(self.test_keys,files_feature_rgb,files_feature_flow,cmb)
            if max_val_fscore < val_fscore:
                max_val_fscore = val_fscore
                mx_video_scores=video_scores
                mx_test_preds=test_preds
                max_val_fscore_epoch = epoch
                maxkt=kt
                maxsp=sp
            avg_loss = np.array(avg_loss)
            print("   Train loss: ",np.mean(avg_loss[:, 0]))
            print("   Test F-score avg/max: ",val_fscore,"/",max_val_fscore)
            train_val_loss_score.append([loss,np.mean(avg_loss[:, 0]),val_fscore,test_loss, video_scores,kt,sp])
            max_val_fscoreLs.append(max_val_fscore)
            if(len(max_val_fscoreLs)>2 and max_val_fscoreLs[-2]==max_val_fscoreLs[-1]):
                sameCount+=1
            else:
                sameCount=0
            if self.verbose:
                video_scores = [["No", "Video", "F-score"]] + video_scores
            # Save model weights
            path, filename = os.path.split(self.split_file)
            base_filename, _ = os.path.splitext(filename)
            path = os.path.join(output_dir, 'models_temp', base_filename+'_'+str(self.split_id))
            os.makedirs(path, exist_ok=True)
            filename = str(epoch)+'_'+str(round(val_fscore*100,3))+'.pth.tar'
            torch.save(self.model.state_dict(), os.path.join(path, filename))
            if(sameCount>=sameAccStopThres):
                break
        return max_val_fscore, max_val_fscore_epoch, maxkt, maxsp,mx_video_scores,train_val_loss_score,mx_test_preds

    def eval_function(self, keys,files_feature_rgb,files_feature_flow,cmb, results_filename=None):
        self.model.eval()
        summary = {}
        preds = {}
        att_vecs = {}
        with torch.no_grad():
            for i, key in enumerate(keys):
                dataset = self.get_data(key)
                seq1 = dataset['features'][...]
                if(self.dataset_name=="tvsum"):
                    idx=self.getIdxInNames(files_feature_rgb,key)
                if(self.dataset_name=="summe"):
                    idx=self.getIdxInNamesSumme(files_feature_rgb,dataset['video_name'][...].astype(str))
                seq2 = np.load(files_feature_rgb[idx])
                seq3 = np.load(files_feature_flow[idx])
                maxShape=np.max([seq1.shape[0],seq2.shape[0],seq3.shape[0]])
                minShape=np.min([seq1.shape[0],seq2.shape[0],seq3.shape[0]])
                fileName = files_feature_rgb[idx].split(os.path.sep)[-1].split('.')[0]
#                visFeaturesSize = self.feat_input["feature_size"]
                visFeaturesSize = self.feat_input.feature_size
                if(self.params_obj.sample_technique=="up"):
                    seq1 = cv2.resize(seq1, (seq1.shape[1],maxShape), interpolation = cv2.INTER_AREA)
                    seq2 = cv2.resize(seq2, (seq2.shape[1],maxShape), interpolation = cv2.INTER_AREA)
                    seq3 = cv2.resize(seq3, (seq3.shape[1],maxShape), interpolation = cv2.INTER_AREA)
                if(self.params_obj.sample_technique=="sub"):
                    seq1 = cv2.resize(seq1, (seq1.shape[1],minShape), interpolation = cv2.INTER_AREA)
                    seq2 = cv2.resize(seq2, (seq2.shape[1],minShape), interpolation = cv2.INTER_AREA)
                    seq3 = cv2.resize(seq3, (seq3.shape[1],minShape), interpolation = cv2.INTER_AREA)
                    target = dataset['gtscore'][...]
                features=[]
                if(cmb[0]):
                    features.append(seq1)
                if(cmb[1]):
                    features.append(seq2)
                if(cmb[2]):
                    features.append(seq3)
                if(self.params_obj.fusion_technique=="early"):
                    seq=self.early_fusion(features,method=self.params_obj.method,stack=self.params_obj.stack)
                    seq = torch.from_numpy(seq).unsqueeze(0)
                    if self.params_obj.use_cuda:
                        seq = seq.float().cuda()
                    y, att_vec = self.model(seq,seq.shape[1])
                else:
                    seq1 = torch.from_numpy(seq1).unsqueeze(0)
                    seq2 = torch.from_numpy(seq2).unsqueeze(0)
                    seq3 = torch.from_numpy(seq3).unsqueeze(0)
                    if self.params_obj.use_cuda:
                        seq1 = seq1.float().cuda()
                        seq2 = seq2.float().cuda()
                        seq3 = seq3.float().cuda()
                    criterion = nn.MSELoss()
                    if self.params_obj.use_cuda:
                    	criterion = criterion.cuda()
                    if(self.params_obj.fusion_technique=="inter"):
                        y, att_vec = self.model([seq1,seq2,seq3], seq1.shape[1])
                    else:
                        y1, att_vec1 = self.model(seq1,seq1.shape[1])
                        y2, att_vec2 = self.model(seq2,seq2.shape[1])
                        y3, att_vec3 = self.model(seq3,seq3.shape[1])
                        y = y1 + y2 + y3
                        att_vec = att_vec1 + att_vec2 + att_vec3
                summary[key] = y[0].detach().cpu().numpy()
                att_vecs[key] = att_vec.detach().cpu().numpy()
                preds[key] = y[0].detach().cpu().numpy()
                target = torch.from_numpy(target).unsqueeze(0)
                target -= target.min()
                target=np.true_divide(target, target.max())
                if self.params_obj.use_cuda:
                    target = target.float().cuda()
                test_loss = criterion(y, target)
        f_score, video_scores,kt,sp = self.eval_summary(summary, keys, metric=self.dataset_name,
                    results_filename=results_filename, att_vecs=att_vecs)
        return f_score, video_scores,kt,sp,test_loss ,preds
    
    def eval_summary(self, machine_summary_activations, test_keys, results_filename=None, metric='tvsum', att_vecs=None):
        eval_metric = 'avg' if metric == 'tvsum' else 'max'
        if results_filename is not None:
            h5_res = h5py.File(results_filename, 'w')
        fms = []
        kts = []
        sps = []
        video_scores = []
        for key_idx, key in enumerate(test_keys):
            d = self.get_data(key)
            probs = machine_summary_activations[key]
            if 'change_points' not in d:
                print("ERROR: No change points in dataset/video ",key)
            cps = d['change_points'][...]
            num_frames = d['n_frames'][()]
            nfps = d['n_frame_per_seg'][...].tolist()
            positions = d['picks'][...]
            user_summary = d['user_summary'][...]
            machine_summary = generate_summary(probs, cps, num_frames, nfps, positions)
            fm, _, _ = evaluate_summary(machine_summary, user_summary, eval_metric)
            fms.append(fm)
            y_pred2=machine_summary
            y_true2=user_summary.mean(axis=0)
            pS=spearmanr(y_pred2,y_true2)[0]
            kT=kendalltau(rankdata(-np.array(y_true2)), rankdata(-np.array(y_pred2)))[0]
            kts.append(kT)
            sps.append(pS)
            # Reporting & logging
#            video_scores.append([key_idx + 1, key, "{:.1%}".format(fm)])
            video_scores.append([key_idx + 1, key, str(round(fm, 2))])
            if results_filename:
                gt = d['gtscore'][...]
                h5_res.create_dataset(key + '/score', data=probs)
                h5_res.create_dataset(key + '/machine_summary', data=machine_summary)
                h5_res.create_dataset(key + '/gtscore', data=gt)
                h5_res.create_dataset(key + '/fm', data=fm)
                h5_res.create_dataset(key + '/picks', data=positions)
                video_name = key.split('/')[1]
                if 'video_name' in d:
                    video_name = d['video_name'][...]
                h5_res.create_dataset(key + '/video_name', data=video_name)
                if att_vecs is not None:
                    h5_res.create_dataset(key + '/att', data=att_vecs[key])
        mean_fm = np.mean(fms)
        kt_fm = np.mean(kts)
        sp_fm = np.mean(sps)
        # Reporting & logging
        if results_filename is not None:
            h5_res.close()
        return mean_fm, video_scores,kt_fm , sp_fm

#==============================================================================================
def eval_split(params_obj, splits_filename,aperture, data_dir='output'):
    print("\n")
    trainObj = TrainingObj(params_obj)
    trainObj.initialize(aperture)
    trainObj.load_object_features()
    trainObj.load_split_file(splits_filename)
    val_fscores = []
    for split_id in range(len(trainObj.splits)):
        trainObj.select_split(split_id)
        weights_filename, _ = trainObj.lookup_weights_file(data_dir)
        print("Loading model:", weights_filename)
        trainObj.load_model(weights_filename)
        val_fscore, video_scores = trainObj.eval(trainObj.test_keys)
        val_fscores.append(val_fscore)
        val_fscore_avg = np.mean(val_fscores)
        if params_obj.verbose:
            video_scores = [["No.", "Video", "F-score"]] + video_scores
        print("Avg F-score: ", val_fscore)
        print("")
    print("Total AVG F-score: ", val_fscore_avg)
    del trainObj
    return val_fscore_avg

def get_dataset_by_name(params_obj, dataset_name):
        for d in  params_obj.object_features:
            if dataset_name in d:
                return [d]
        return None
    
    
def train(params_obj):
    os.makedirs(params_obj.output_dir, exist_ok=True)
    os.makedirs(os.path.join(params_obj.output_dir, 'splits'), exist_ok=True)
    os.makedirs(os.path.join(params_obj.output_dir, 'code'), exist_ok=True)
    os.makedirs(os.path.join(params_obj.output_dir, 'models'), exist_ok=True)
    os.system('cp -f splits/*.json  ' + params_obj.output_dir + '/splits/')
    os.system('cp *.py ' + params_obj.output_dir + '/code/')
    for aperture in params_obj.apertures:
        for cmb in params_obj.combis:
            model_anchor=",".join(np.array(cmb).astype(str))+"_"+str(aperture)+"_"+params_obj.method+"_"+params_obj.sample_technique+"_"+params_obj.name_anchor+"_"+params_obj.stack+"_"
            # Create a file to collect results from all splits
            f = open(params_obj.output_dir + os.path.sep +model_anchor+'_results.txt', 'wt')
            for split_filename in params_obj.splits:
                f2 = open(params_obj.output_dir+ os.path.sep +model_anchor+'_results_'+split_filename.split(os.path.sep)[-1][:-5]+'.txt', 'wt')
                dataset_name, dataset_type, splits = parse_splits_filename(split_filename)
                # For no augmentation use only a dataset corresponding to the split file
                object_features = None
                if dataset_type == '':
                    object_features =   get_dataset_by_name(params_obj,dataset_name)
                if object_features is None:
                    object_features = params_obj.object_features
                f_avg = 0
                kt_avg = 0
                sp_avg = 0
                n_folds = len(splits)
                for split_id in range(n_folds):
                    trainObj = TrainingObj(params_obj)
                    trainObj.initialize(params_obj.feat_input,cmb)
                    trainObj.dataset_name=dataset_name
                    trainObj.load_object_features(object_features=object_features)
                    trainObj.load_split_file(splits_file=split_filename)
                    trainObj.select_split(split_id=split_id)
                    fscore, fscore_epoch,kt,sp,video_scores,train_val_loss_score,mx_test_preds = trainObj.train(cmb=cmb,output_dir=params_obj.output_dir)
                    del trainObj
                    f_avg += fscore
                    kt_avg += kt
                    sp_avg += sp
                    f.write(split_filename + ', ' + str(split_id) + ', ' + str(fscore) + ', ' + str(fscore_epoch) + ', ' + str(kt) + ', ' + str(sp) +','+'|'.join(np.array(video_scores)[:,0].astype(str))+','+'|'.join(np.array(video_scores)[:,1].astype(str))+','+'|'.join(np.array(video_scores)[:,2].astype(str))+ '\n')
                    f2.write(split_filename + ', ' + str(split_id) + ', ' + str(fscore) + ', ' + str(fscore_epoch) + ', ' + str(kt) + ', ' + str(sp)+','+'|'.join(np.array(video_scores)[:,0].astype(str))+','+'|'.join(np.array(video_scores)[:,1].astype(str))+','+'|'.join(np.array(video_scores)[:,2].astype(str)) + '\n')
                    # Save model with the highest F score
                    _, log_file = os.path.split(split_filename)
                    log_dir, _ = os.path.splitext(log_file)
                    log_dir += '_' + str(split_id)
                    log_file = os.path.join(params_obj.output_dir, 'models', log_dir) + '_' + str(fscore) + '.tar.pth'
                    os.makedirs(os.path.join(params_obj.output_dir, 'models', ), exist_ok=True)
                    os.system('mv ' + params_obj.output_dir + '/models_temp/' + log_dir + '/' + str(fscore_epoch) + '_*.pth.tar ' + log_file)
                    os.system('rm -rf ' + params_obj.output_dir + '/models_temp/' + log_dir)
#                    print("Split: {0:}   Best F-score: {1:0.5f}   Model: {2:}".format(split_filename, fscore, log_file))
                    print("Split: ",split_filename,"  Best F-score: ",fscore," Model: ", log_file)
                    pickle.dump( np.array(train_val_loss_score), open(params_obj.output_dir + '/'+model_anchor+"_"+split_filename.split("/")[-1][:-5]+"_"+str(split_id)+"_train_val_loss_score.p", "wb" ) )
                    pickle.dump( mx_test_preds, open(params_obj.output_dir + '/'+model_anchor+"_"+split_filename.split("/")[-1][:-5]+"_"+str(split_id)+"_test_preds.p", "wb" ) )
                # Write average F-score for all splits to the results.txt file
                f_avg /= n_folds
                kt_avg /= n_folds
                sp_avg /= n_folds
                f.write(split_filename + ', ' + str('avg') + ', ' + str(f_avg)+ ', ' + str(kt_avg)+ ', ' + str(sp_avg) + '\n')
                f.flush()
                f2.flush()
                f2.close()
            f.close()

if __name__ == "__main__":
    print("Parameters:")
    print("----------------------------------------------------------------------")
    json_params = parse_arguments(sys.argv)
    params_obj = obj(json_params)
    print(json_params)
    print("----------------------------------------------------------------------")
    train(params_obj)
    sys.exit(0)



