"""Modified from SparseConvNet data preparation: https://github.com/facebookres
earch/SparseConvNet/blob/master/examples/ScanNet/prepare_data.py."""

import argparse
import glob
import json
import multiprocessing as mp
import os, sys
import numpy as np
import plyfile
import scannet_util
import torch

# Map relevant classes to {0,1,...,19}, and ignored classes to -100
# remapper = np.ones(150) * (-100)
# for i, x in enumerate([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]):
#     remapper[x] = i

remapper = {'bathtub':0, 'bed':1, 'bench':2, 'bookshelf':3, 'bottle':4, 'chair':5,
            'cup':6,'curtain':7,'desk':8, 'door':9, 'dresser':10, 'keyboard':11,
            'lamp':12, 'laptop':13, 'monitor':14, 'nightstand':15, 'plant':16, 'sofa':17, 'stool':18, 'table':19, 'toilet':20, 'wardrobe':21, 'backgroud':22}
g_label_names = ['bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'chair', 'cup', 'curtain', 'desk', 'door', 'dresser',
             'keyboard', 'lamp', 'laptop', 'monitor',
             'night_stand', 'plant', 'sofa', 'stool', 'table', 'toilet', 'wardrobe','unannotated']
parser = argparse.ArgumentParser()
parser.add_argument('--data_split', help='data split (train / val / test)', default='train')
opt = parser.parse_args()

split = opt.data_split
print('data split: {}'.format(split))
files = sorted(glob.glob('/data1/szh/softgroup/' + split + '/*_vh_clean_2.ply'))
if opt.data_split != 'test':
    files2 = sorted(glob.glob('/data1/szh/softgroup/' + split + '/*_vh_clean_2.labels.ply'))
    files3 = sorted(glob.glob('/data1/szh/softgroup/' + split + '/*_vh_clean_2.0.010000.segs.json'))
    files4 = sorted(glob.glob('/data1/szh/softgroup/' + split + '/*[0-9].aggregation.json'))
    assert len(files) == len(files2)
    assert len(files) == len(files3)
    assert len(files) == len(files4), '{} {}'.format(len(files), len(files4))


def f_test(fn):
    print(fn)

    f = plyfile.PlyData().read(fn)
    points = np.array([list(x) for x in f.elements[0]])
    coords = np.ascontiguousarray(points[:, :3] - points[:, :3].mean(0))
    colors = np.ascontiguousarray(points[:, 3:6]) / 127.5 - 1

    torch.save((coords, colors), fn[:-15] + '_inst_nostuff.pth')
    print('Saving to ' + fn[:-15] + '_inst_nostuff.pth')



def f(fn):
    fn2 = fn[:-3] + 'labels.ply'
    fn3 = fn[:-15] + '_vh_clean_2.0.010000.segs.json'
    fn4 = fn[:-15] + '.aggregation.json'
    print(fn)

    f = plyfile.PlyData().read(fn)
    points = np.array([list(x) for x in f.elements[0]])#_vh_clean_2.ply read point's xyzRGB
    coords = np.ascontiguousarray(points[:, :3] - points[:, :3].mean(0))
    colors = np.ascontiguousarray(points[:, 3:6]) / 127.5 - 1


    label_map = scannet_util.g_raw2scannetv2
    object_id_to_segs, label_to_segs = scannet_util.read_aggregation(fn4)
    seg_to_verts, num_verts = scannet_util.read_segmentation(fn3)
    # label_ids = np.zeros(shape=(num_verts), dtype=np.uint32)# 0: unannotated
    label_ids = np.array([None] * num_verts)  # 0: unannotated
    object_id_to_label_id = {}
    for label, segs in label_to_segs.items():
        label_id = label_map[label]
        for seg in segs:
            verts = seg_to_verts[seg]
            label_ids[verts] = label_id
    # for i in range(label_ids.shape[0]):
    #     print(label_ids[i])
    instance_ids = np.zeros(shape=(num_verts), dtype=np.uint32)  # 0: unannotated
    num_instances = len(np.unique(list(object_id_to_segs.keys())))
    for object_id, segs in object_id_to_segs.items():
        for seg in segs:
            verts = seg_to_verts[seg]
            instance_ids[verts] = object_id
            if object_id not in object_id_to_label_id:
                object_id_to_label_id[object_id] = label_ids[verts][0]
    instance_labels = instance_ids.astype(np.float64)
    sem_labels =  np.ones(num_verts) * (-100)  # -100: unannotated
    for i in range(points.shape[0]):
        if label_ids[i] in g_label_names :
            sem_labels[i] = remapper[label_ids[i]]

    f2 = plyfile.PlyData().read(fn2)
    # sem_labels = remapper[np.array(f2.elements[0]['label'])]
    #
    # with open(fn3) as jsondata:
    #     d = json.load(jsondata)
    #     seg = d['segIndices']
    # segid_to_pointid = {}
    # for i in range(len(seg)):
    #     if seg[i] not in segid_to_pointid:
    #         segid_to_pointid[seg[i]] = []
    #     segid_to_pointid[seg[i]].append(i)
    #
    # instance_segids = []
    # labels = []
    # with open(fn4) as jsondata:
    #     d = json.load(jsondata)
    #     for x in d['segGroups']:
    #         if scannet_util.g_raw2scannetv2[x['label']] != 'wall' and scannet_util.g_raw2scannetv2[
    #                 x['label']] != 'floor':
    #             instance_segids.append(x['segments'])
    #             labels.append(x['label'])
    #             assert (x['label'] in scannet_util.g_raw2scannetv2.keys())
    # if (fn == 'val/scene0217_00_vh_clean_2.ply'
    #         and instance_segids[0] == instance_segids[int(len(instance_segids) / 2)]):
    #     instance_segids = instance_segids[:int(len(instance_segids) / 2)]
    # check = []
    # for i in range(len(instance_segids)):
    #     check += instance_segids[i]
    # assert len(np.unique(check)) == len(check)
    #
    # instance_labels = np.ones(sem_labels.shape[0]) * -100
    # for i in range(len(instance_segids)):
    #     segids = instance_segids[i]
    #     pointids = []
    #     for segid in segids:
    #         pointids += segid_to_pointid[segid]
    #     instance_labels[pointids] = i
    #     assert (len(np.unique(sem_labels[pointids])) == 1)

    torch.save((coords, colors, sem_labels, instance_labels), fn[:-15] + '_inst_nostuff.pth')
    print('Saving to ' + fn[:-15] + '_inst_nostuff.pth')


# for fn in files:
#     f(fn)

p = mp.Pool(processes=mp.cpu_count())
if opt.data_split == 'test':
    p.map(f_test, files)
else:
    p.map(f, files)
p.close()
p.join()
