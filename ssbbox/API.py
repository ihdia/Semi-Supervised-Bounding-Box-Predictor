import os

import numpy
import requests
import torch
import json
import cv2
import json
import numpy as np
import math
import argparse
import torch
from .edgeonlyy import Model
from . import ConcaveHull as ch
from torch.utils.data import DataLoader
# import operator
import torch.nn as nn
from collections import OrderedDict

device = torch.device("cpu")
# if torch.cuda.is_available():
#     print("!!!!!!got cuda!!!!!!!")
# else:
#     print("!!!!!!!!!!!!no cuda!!!!!!!!!!!!")

def uniformsample_batch(batch, num):
    final = []
    for i in batch:
        i1 = np.asarray(i).astype(int)
        a = uniformsample(i1, num)
        a = torch.from_numpy(a)
        a = a.long()
        final.append(a)
    return final

def uniformsample(pgtnp_px2, newpnum):
    pnum, cnum = pgtnp_px2.shape
    assert cnum == 2

    idxnext_p = (np.arange(pnum, dtype=np.int32) + 1) % pnum
    pgtnext_px2 = pgtnp_px2[idxnext_p]
    edgelen_p = np.sqrt(np.sum((pgtnext_px2 - pgtnp_px2) ** 2, axis=1))
    edgeidxsort_p = np.argsort(edgelen_p)

    # two cases
    # we need to remove gt points
    # we simply remove shortest paths
    if pnum > newpnum:
        edgeidxkeep_k = edgeidxsort_p[pnum - newpnum:]
        edgeidxsort_k = np.sort(edgeidxkeep_k)
        pgtnp_kx2 = pgtnp_px2[edgeidxsort_k]
        assert pgtnp_kx2.shape[0] == newpnum
        return pgtnp_kx2
    # we need to add gt points
    # we simply add it uniformly
    else:
        edgenum = np.round(edgelen_p * newpnum / np.sum(edgelen_p)).astype(np.int32)
        for i in range(pnum):
            if edgenum[i] == 0:
                edgenum[i] = 1

        # after round, it may has 1 or 2 mismatch
        edgenumsum = np.sum(edgenum)
        if edgenumsum != newpnum:

            if edgenumsum > newpnum:

                id = -1
                passnum = edgenumsum - newpnum
                while passnum > 0:
                    edgeid = edgeidxsort_p[id]
                    if edgenum[edgeid] > passnum:
                        edgenum[edgeid] -= passnum
                        passnum -= passnum
                    else:
                        passnum -= edgenum[edgeid] - 1
                        edgenum[edgeid] -= edgenum[edgeid] - 1
                        id -= 1
            else:
                id = -1
                edgeid = edgeidxsort_p[id]
                edgenum[edgeid] += newpnum - edgenumsum

        assert np.sum(edgenum) == newpnum

        psample = []
        for i in range(pnum):
            pb_1x2 = pgtnp_px2[i:i + 1]
            pe_1x2 = pgtnext_px2[i:i + 1]

            pnewnum = edgenum[i]
            wnp_kx1 = np.arange(edgenum[i], dtype=np.float32).reshape(-1, 1) / edgenum[i];

            pmids = pb_1x2 * (1 - wnp_kx1) + pe_1x2 * wnp_kx1
            psample.append(pmids)

        psamplenp = np.concatenate(psample, axis=0)
        return psamplenp

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--img', type=str)
    parser.add_argument('--bbox', type=str, default=None)

    args = parser.parse_args()

    return args

def get_hull(edge_logits):
    test = edge_logits
    test_0 = test[:, :]
    test_1 = test[:, :]

    


    # for i in range(len(test_0)):
    #     for j in range(len(test_0[0])):
    #         if test_0[i][j] > 0.7:
    #             test_1[i][j] = 1
    #         else:
    #             test_1[i][j] = 0
    points_pred = []

    for i in range(len(test_1)):
        for j in range(len(test_1[0])):
            if test_1[i][j] > 0:
                # points_pred.append([i + 1, j + 1])
                points_pred.append([i, j])

    points_pred = np.asarray(points_pred)

    hull = ch.concaveHull(points_pred, 3)
    return hull

def convert_hull_to_cv(hull, w, h):

    original_hull = []

    # w = bbox[2] + 20
    # h = bbox[3]

    for i in hull:
        original_hull.append([int((i[1]-5) * w / 80), int((i[0]-2) * h / 20)])
    return original_hull

# model_path = './checkpoints_cgcn/Final.pth'
model = Model(256, 960, 3).to(device)
# if torch.cuda.device_count() >= 1:
#     print("Let's use", torch.cuda.device_count(), "GPUs!")
#     # dim = 0 [20, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
#     model = nn.DataParallel(model)
model_path = os.environ.get('SSBBOX_MODEL_PATH')
if not model_path:
    (file_dir, fname) = os.path.split(__file__)
    model_path = os.path.normpath(os.path.join(file_dir, '..', 'Best2_12.pth'))

state_dict = torch.load(model_path, map_location=torch.device('cpu'))
new_state_dict = OrderedDict()
for k, v in state_dict["gcn_state_dict"].items():
    name = k[7:] # remove `module.`
    # print(k,v)
    new_state_dict[name] = v
# load params
model.load_state_dict(new_state_dict)
model.to(device)


def get_edge(img,bbox):

    # img = cv2.imread(i_url)

    x0 = max(int(bbox[0]),0)
    y0 = max(int(bbox[1]),0)
    w = max(int(bbox[2]),0)
    h = max(int(bbox[3]),0)

    img = img[y0:y0+h,x0:x0+w]

    # img= cv2.copyMakeBorder(img,0,0,10,10,cv2.BORDER_REPLICATE)
    img = cv2.resize(img, (256, 960))
    img = torch.from_numpy(img)


    # img = torch.cat(img)
    img = img.view(-1, 256, 960, 3)
    img = torch.transpose(img, 1, 3)
    img = torch.transpose(img, 2, 3)
    img = img.float()

    edge_logits, tg2, class_prob = model(img.to(device))
    edge_logits = torch.sigmoid(edge_logits)

    edge_logits = edge_logits[0,0,:,:].cpu().detach().numpy()
    # print(len(edge_logits))
    arrs1 = np.zeros((24, 90), np.uint8)

    for j in range(len(edge_logits)):
        for k in range((len(edge_logits[j]))):
            j1 = math.floor(j)
            k1 = math.floor(k)
            if edge_logits[j][k]>0.55:
                arrs1[j1+2][k1+5]= 255


    borders = np.zeros((24, 90), np.uint8)
    kernel5 = np.ones((3,3),np.uint8)

    arrs1 = cv2.morphologyEx(arrs1, cv2.MORPH_CLOSE, kernel5)
    
    im2, contours, hierarchy = cv2.findContours(arrs1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(borders, contours, -1, 255, 1)

    arrs1 = np.zeros((24, 90), np.float32)
    for j in range(len(borders)):
        for k in range((len(borders[j]))):
            j1 = math.floor(j)
            k1 = math.floor(k)
            if borders[j][k] > 180:
                arrs1[j1][k1]= 1.0

    arrs1 = torch.from_numpy(arrs1)
    hull = get_hull(arrs1)

    hull = np.asarray(hull)
    hull = hull.tolist()


    original_hull = convert_hull_to_cv(hull, w, h)

    total_points = 110
    # original_hull = uniformsample(np.asarray(original_hull), total_points).astype(int)
    original_hull = np.asarray(original_hull).astype(int)
    original_hull[:,0] = (original_hull[:,0] + x0)
    original_hull[:,1] = (original_hull[:,1] + y0)

    original_hull = original_hull.tolist()


    # all_points_x = simplify_coords_vw(all_points_x, 30.0)
    # all_points_y = simplify_coords_vw(all_points_y, 30.0)

    return original_hull

def get_edges(i_url, bboxs):

    i_url = str(i_url)

    img_content = requests.get(i_url).content
    np_arr = numpy.fromstring(img_content, numpy.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # img = cv2.imread(i_url)

    #  print(img)

    polys = []
    for bbox in bboxs:
        poly = get_edge(img, bbox)
        polys.append(poly)
    return polys

# if __name__ == '__main__':
#     args = get_args()
#     img = args.img
#     bbox1 = args.bbox
#     # bbox1 = bbox1.split(",")
#     bbox1 = map(float, bbox1.strip('[]').split(','))
#     # x, y = get_edge("./Andros332.jpg", [41,240,909,168])
#     hullpp = get_edges(img, bbox1)
#     print(hullpp)
#     # print(y)
