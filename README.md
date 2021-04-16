# MSVA (Multi Source Visual Attention)
MSVA is a deep learning model for supervised video summarization. In this research, we address this research gap and investigate how different feature types, i.e., static and motion features, can be integrated in a model architecture for video summarization.

![msva model](media/msva_model.PNG)

## Get started (Requirements and Setup)
Python version >= 3.6

``` bash
# clone the repository
git clone git@github.com:VideoAnalysis/MSVA.git
cd MSVA
conda create -n msva python=3.6
conda activate msva  
pip install -r requirements.txt
```

## Dataset
Extracted features for the datasets can be downloaded as,
``` bash
wget -O datasets.tar https://zenodo.org/record/4682137/files/msva_video_summarization.tar
tar -xvf datasets.tar
```

## Training and Crossfold Validation
``` bash
# for training, crossfold validation according to default parameters "parameters.json".
python train.py -params parameters.json
```

## Experimental Configuration
Update parameters.json for desired experimental parameters.s.
``` bash
{"verbose":False,
"train":True,
"use_cuda":True,
"cuda_device":0,
"max_summary_length":0.15,
"l2_req":0.00001,
"lr_epochs":[0],
"lr":[0.00005],
"epochs_max":300,
"train_batch_size":5,
"root":'',
"fusion_technique":'inter',
"method":'mean',
"sample_technique":'sub',
"stack":'v',
"name_anchor":"inter_add_aperture_250",
"output_dir" : "./results/",
"base_dir" : "",
"apertures":[250],
"local":False,
"combis":[[1,1,1]],
"feat_input":[1024],
"object_features":['datasets/object_features/eccv16_dataset_summe_google_pool5.h5',
			   'datasets/object_features/eccv16_dataset_tvsum_google_pool5.h5'],
"kinetic_features":"./datasets/kinetic_features/",
"splits":['splits/tvsum_splits.json',
 'splits/summe_splits.json',
 'splits/summe_random_non_overlap_splits.json',
 'splits/tvsum_random_non_overlap_splits.json'],
"feat_input":{"feature_size":365,"L1_out":365,"L2_out":365,"L3_out":512,"pred_out":1,"apperture":250,"dropout1":0.5,"att_dropout1":0.5,"feature_size_1_3":1024,"feature_size_4":365}}
```

## Citation
```
@article{ghauri2021MSVA, 
   title={SUPERVISED VIDEO SUMMARIZATION VIA MULTIPLE FEATURE SETS WITH PARALLEL ATTENTION},
   author={Ghauri, Junaid Ahmed and Hakimov, Sherzod and Ewerth, Ralph}, 
   Conference={IEEE International Conference on Multimedia and Expo (ICME)}, 
   year={2021} 
}
```

For orignal source of these datasets including videos, Follow:
“[SumMe, Creating Summaries from User Videos, ECCV 2014](https://gyglim.github.io/me/vsum/index.html)”
“[TVSum , TVSum: Summarizing web videos using titles, 2015](https://github.com/yalesong/tvsum)”
