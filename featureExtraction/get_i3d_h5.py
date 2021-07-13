import os
import h5py
import numpy as np
import json

feat_npy_root = 'feats_i3d_rgb_npy/'   # need to be replaced
feat_files = os.listdir(feat_npy_root)
feat_files = [item for item in feat_files if item.endswith('.npy')]

feat_dict = {}

print('Start ...')
count = 0
for item in feat_files:
    vid = item.split('.')[0]
    print(vid)
    filepath = os.path.join(feat_npy_root, item)
    feat = np.load(filepath)
    
    feat_dict[vid] = feat

    count += 1
    if count%1000 == 0:
        print('Processed %d files.'%count)

print('Processed %d files.'%count)


print('Writing file ...')

fid = h5py.File('charades_i3d_rgb_stride_1s.hdf5', 'w')

for vid in feat_dict.keys():
    if vid in fid:
        print('WARNING: group name exists.')
        continue

    fid.create_group(vid).create_dataset('i3d_rgb_features', data=feat_dict[vid])

print('Done.')


