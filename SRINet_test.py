import torch
import torch.nn.functional as F
import sys
sys.path.append('./models')
import numpy as np
import os, argparse
import cv2
from models.SRINet_model import SRINet
from data import Test_dataset
import torch.utils.data as data
import time


parser = argparse.ArgumentParser()
# parser.add_argument('--testsize', type=int, default=352, help='testing size')
# parser.add_argument('--gpu_id', type=str, default='0', help='select gpu id')
parser.add_argument('--test_path',type=str,default='/home/dataset/Light-Field_dataset/test_data/',help='test dataset path')
opt = parser.parse_args()

dataset_path = opt.test_path

#set device for test
# if opt.gpu_id=='0':
#     os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#     print('USE GPU 0')
# elif opt.gpu_id=='1':
#     os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#     print('USE GPU 1')

image_root = dataset_path + '/test_images/'
gt_root = dataset_path +  '/test_masks/'
depth_root=dataset_path +'/test_focal/'
test_dataset=Test_dataset(image_root, gt_root,depth_root, 256)
batch= 1
data_loader= data.DataLoader(dataset=test_dataset,
    batch_size=batch,
    shuffle=False,
    num_workers=4,
    pin_memory=True)
#load the model
# model = SRINet()
model_fuse= SRINet()
CE = torch.nn.BCEWithLogitsLoss()

checkplist=[]
with torch.no_grad():
    modelpath= './SRINet_cpts/SRINet.pth'
    model_fuse.load_state_dict(torch.load(modelpath))

    model_fuse.cuda()
    model_fuse.eval()
    save_path = './test_maps/DUTLF/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    loss_e= 0
    t_sum= 0
    batch_sum=0

    for i, (image, gt, depth, name) in enumerate(data_loader):

        print('batch {:4d}'.format(i))
        image = image.cuda()
        depth = depth.cuda()
        gt= gt.cuda()
        bz= image.shape[0]

        t_start= time.time()
        s2_fuse, _, _, _, _, _, _, = model_fuse(image, depth)
        t_inter= time.time()- t_start
        t_sum= t_sum+ t_inter
        batch_sum= batch_sum+ 1


        res= s2_fuse

        res = res.data.cpu().numpy()

        for j in range(bz):
            res2= res[j]
            res2= res2.squeeze()
            name2= name[j]
            res2 = (res2 - res2.min()) / (res2.max() - res2.min() + 1e-8)
            cv2.imwrite(save_path+ name2, res2*255)


######HFUT
dataset_path= '/home/dataset/HFUT-Test-155/'
image_root = dataset_path + '/test_images/'
gt_root = dataset_path +  '/test_masks/'
depth_root=dataset_path +'/test_focal/'
test_dataset=Test_dataset(image_root, gt_root,depth_root, 256)
batch= 1
data_loader= data.DataLoader(dataset=test_dataset,
    batch_size=batch,
    shuffle=False,
    num_workers=4,
    pin_memory=True)

model_fuse= SRINet()
CE = torch.nn.BCEWithLogitsLoss()

checkplist=[]

with torch.no_grad():
    modelpath= './SRINet_cpts/SRINet.pth'
    model_fuse.load_state_dict(torch.load(modelpath))

    model_fuse.cuda()

    model_fuse.eval()
    save_path = './test_maps/HFUT/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    loss_e= 0

    for i, (image, gt, depth, name) in enumerate(data_loader):

        print('batch {:4d}'.format(i))
        image = image.cuda()
        depth = depth.cuda()
        gt= gt.cuda()
        bz= image.shape[0]

        s2_fuse, _, _, _, _, _, _, = model_fuse(image, depth)

        res= s2_fuse

        res = res.data.cpu().numpy()

        for j in range(bz):
            res2= res[j]
            res2= res2.squeeze()
            name2= name[j]

            res2 = (res2 - res2.min()) / (res2.max() - res2.min() + 1e-8)

            cv2.imwrite(save_path+ name2, res2*255)


