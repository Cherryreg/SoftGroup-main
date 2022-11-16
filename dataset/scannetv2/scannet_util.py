import os
import sys
import json
import csv
#
# g_label_names = [
#     'unannotated', 'wall', 'floor', 'chair', 'table', 'desk', 'bed', 'bookshelf', 'sofa', 'sink',
#     'bathtub', 'toilet', 'curtain', 'counter', 'door', 'window', 'shower curtain', 'refridgerator',
#     'picture', 'cabinet', 'otherfurniture']
g_label_names = ['unannotated', 'wall', 'floor', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'chair', 'cup', 'curtain', 'desk', 'door', 'dresser',
             'keyboard', 'lamp', 'laptop', 'monitor',
             'night_stand', 'plant', 'sofa', 'stool', 'table', 'toilet', 'wardrobe']
def represents_int(s):
    ''' if string s represents an int. '''
    try:
        int(s)
        return True
    except ValueError:
        return False
def read_aggregation(filename):
    assert os.path.isfile(filename)
    object_id_to_segs = {}
    label_to_segs = {}
    with open(filename) as f:
        data = json.load(f)
        num_objects = len(data['segGroups'])
        for i in range(num_objects):
            object_id = data['segGroups'][i]['objectId'] + 1 # instance ids should be 1-indexed
            label = data['segGroups'][i]['label']
            segs = data['segGroups'][i]['segments']
            object_id_to_segs[object_id] = segs
            if label in label_to_segs:
                label_to_segs[label].extend(segs)
            else:
                label_to_segs[label] = segs
    return object_id_to_segs, label_to_segs

def read_segmentation(filename):
    assert os.path.isfile(filename)
    seg_to_verts = {}
    with open(filename) as f:
        data = json.load(f)
        num_verts = len(data['segIndices'])
        for i in range(num_verts):
            seg_id = data['segIndices'][i]
            if seg_id in seg_to_verts:
                seg_to_verts[seg_id].append(i)
            else:
                seg_to_verts[seg_id] = [i]
    return seg_to_verts, num_verts
def get_raw2scannetv2_label_map():
    # lines = [line.rstrip() for line in open('/data/lxr/dataset/scannet/scannetv2-labels.combined.tsv')]
    # lines_0 = lines[0].split('\t')
    # print(lines_0)
    # print(len(lines))
    # lines = lines[1:]
    # raw2scannet = {}
    # for i in range(len(lines)):
    #     label_classes_set = set(g_label_names)
    #     elements = lines[i].split('\t')
    #     raw_name = elements[1]
    #     if (elements[1] != elements[2]):
    #         print('{}: {} {}'.format(i, elements[1], elements[2]))
    #     nyu40_name = elements[7]
    #     modelnet40_name = elements[9]
    #     if nyu40_name not in label_classes_set:
    #         raw2scannet[raw_name] = 'unannotated'
    #     else:
    #         raw2scannet[raw_name] = nyu40_name
    filename = '/data/lxr/dataset/scannet/scannetv2-labels.combined.tsv'
    label_from = 'raw_category'
    label_to = 'ModelNet40'
    assert os.path.isfile(filename)
    mapping = dict()
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')
        for row in reader:
            if label_to == 'nyu40id':
                mapping[row[label_from]] = int(row[label_to])
            if label_to == 'ModelNet40':
                mapping[row[label_from]] = (row[label_to])
    if represents_int(list(mapping.keys())[0]):
        mapping = {int(k): v for k, v in mapping.items()}
    return mapping


g_raw2scannetv2 = get_raw2scannetv2_label_map()
