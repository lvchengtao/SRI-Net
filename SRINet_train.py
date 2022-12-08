import os
import torch
import torch.nn.functional as F
import sys
import torch.optim as optim
sys.path.append('./models')
import numpy as np
from datetime import datetime
from torchvision.utils import make_grid
from models.SRINet_model import SRINet
from data import get_loader,Test_dataset
from utils import clip_gradient, adjust_lr
#from tensorboardX import SummaryWriter
import logging
import torch.backends.cudnn as cudnn
# from options import opt
from torch.optim.lr_scheduler import MultiStepLR

cudnn.benchmark = True

#build the model
model = SRINet()

model.cuda()

params = model.parameters()

opt={'lr':1e-4,
    'epoch':50,
    'batchsize':16,
    'clip':0,
    'decay_rate':0.1,
    'decay_epoch':10,
    'load':0,

}


optimizer = optim.RMSprop(params, opt['lr'], alpha=0.9)

milestones=[10,]
scheduler_focal = MultiStepLR(optimizer, milestones, gamma=0.1, last_epoch=-1)

#set the path
image_root = '/home/dataset/Light-Field_dataset/train_data/train_images/'
gt_root = '/home/dataset/Light-Field_dataset/train_data/train_masks/'
depth_root= '/home/dataset/Light-Field_dataset/train_data/train_focal/'

save_path='./SRINet_cpts/'

if not os.path.exists(save_path):
    os.makedirs(save_path)

#load data
print('load data...')
train_loader = get_loader(image_root, gt_root, depth_root, batchsize= opt['batchsize'], trainsize= 256)

total_step = len(train_loader)

logging.basicConfig(filename=save_path+'log.log',format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]', level = logging.INFO,filemode='a',datefmt='%Y-%m-%d %I:%M:%S %p')
logging.info("SRINet-Train")
logging.info("Config")
logging.info('epoch:{};lr:{};batchsize:{};trainsize:{};clip:{};decay_rate:{};load:{};save_path:{};decay_epoch:{}'.format(opt['epoch'],opt['lr'],opt['batchsize'], 256, opt['clip'],opt['decay_rate'],opt['load'],save_path,opt['decay_epoch']))

#set loss function
CE = torch.nn.BCELoss(size_average=True)
CE_2= torch.nn.BCEWithLogitsLoss(reduction='none')
CE_3= torch.nn.CrossEntropyLoss()

step=0
best_mae=1
best_epoch=0

#train function
def train(train_loader, model, optimizer, epoch, save_path):
    global step
    model.train()
    loss_all=0
    epoch_step=0
    loss_100=0
    min_v_m_s=0
    try:
        for i, (images, gts, depths) in enumerate(train_loader, start=1):
            optimizer.zero_grad()
            images = images.cuda()
            gts = gts.cuda()
            depths=depths.cuda()

            s0,s1,s2,s3,s4,s5,s6= model(images, depths)

            loss0= CE(s0, gts)
            loss1= CE(s1, gts)
            loss2 = CE(s2, gts)
            loss3 = CE(s3, gts)
            loss4 = CE(s4, gts)
            loss5 = CE(s5, gts)
            loss6 = CE(s6, gts)
            
            loss = loss0+loss1+loss2+loss3+loss4+loss5+loss6

            loss.backward()
            optimizer.step()

            step+=1
            epoch_step+=1
            loss_all+=loss.data
            loss_100 += loss.data

            if i % 100 == 0: 
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss: {:.4f}, Loss_100: {:.4f}, Loss2_fuse: {:.4f}, Loss2_de1: {:.4f},'.
                    format(datetime.now(), epoch, opt['epoch'], i, total_step, loss.data, loss_100 / 100.0, loss0.item(), loss1.item()))
                logging.info('#TRAIN#:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss: {:.4f} Loss_100: {:.4f}'.
                    format( epoch, opt['epoch'], i, total_step, loss.data, loss_100 / 100.0))

                loss_100 = 0
        
        loss_all/=epoch_step
        logging.info('#TRAIN#:Epoch [{:03d}/{:03d}], Loss_AVG: {:.4f}'.format( epoch, opt['epoch'], loss_all))
        if (epoch) % 1 == 0:
            torch.save(model.state_dict(), save_path+'SRINet_epoch_{}.pth'.format(epoch))
    except KeyboardInterrupt: 
        print('Keyboard Interrupt: save model and exit.')

        print('save checkpoints successfully!')
        raise
        
 
if __name__ == '__main__':
    print("Start train...")
    for epoch in range(1, opt['epoch']):
        print('learning_rate', optimizer.param_groups[0]['lr'])
        train(train_loader, model, optimizer, epoch, save_path)
        scheduler_focal.step()
