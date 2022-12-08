import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import ImageEnhance
import scipy.io as sio
import torch

#several data augumentation strategies
def cv_random_flip(img, label,depth):
    flip_flag = random.randint(0, 1)
    # flip_flag2= random.randint(0,1)
    #left right flip
    if flip_flag == 1:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
        depth = depth.transpose(Image.FLIP_LEFT_RIGHT)
    #top bottom flip
    # if flip_flag2==1:
    #     img = img.transpose(Image.FLIP_TOP_BOTTOM)
    #     label = label.transpose(Image.FLIP_TOP_BOTTOM)
    #     depth = depth.transpose(Image.FLIP_TOP_BOTTOM)
    return img, label, depth
def randomCrop(image, label,depth):
    border=30
    image_width = image.size[0]
    image_height = image.size[1]
    crop_win_width = np.random.randint(image_width-border , image_width)
    crop_win_height = np.random.randint(image_height-border , image_height)
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)
    return image.crop(random_region), label.crop(random_region),depth.crop(random_region)
def randomRotation(image,label,depth):
    mode=Image.BICUBIC
    if random.random()>0.8:
        random_angle = np.random.randint(-15, 15)
        image=image.rotate(random_angle, mode)
        label=label.rotate(random_angle, mode)
        depth=depth.rotate(random_angle, mode)
    return image,label,depth
def colorEnhance(image):
    bright_intensity=random.randint(5,15)/10.0
    image=ImageEnhance.Brightness(image).enhance(bright_intensity)
    contrast_intensity=random.randint(5,15)/10.0
    image=ImageEnhance.Contrast(image).enhance(contrast_intensity)
    color_intensity=random.randint(0,20)/10.0
    image=ImageEnhance.Color(image).enhance(color_intensity)
    sharp_intensity=random.randint(0,30)/10.0
    image=ImageEnhance.Sharpness(image).enhance(sharp_intensity)
    return image
def randomGaussian(image, mean=0.1, sigma=0.35):
    def gaussianNoisy(im, mean=mean, sigma=sigma):
        for _i in range(len(im)):
            im[_i] += random.gauss(mean, sigma)
        return im
    img = np.asarray(image)
    width, height = img.shape
    img = gaussianNoisy(img[:].flatten(), mean, sigma)
    img = img.reshape([width, height])
    return Image.fromarray(np.uint8(img))
def randomPeper(img):

    img=np.array(img)
    noiseNum=int(0.0015*img.shape[0]*img.shape[1])
    for i in range(noiseNum):

        randX=random.randint(0,img.shape[0]-1)  

        randY=random.randint(0,img.shape[1]-1)  

        if random.randint(0,1)==0:  

            img[randX,randY]=0  

        else:  

            img[randX,randY]=255 
    return Image.fromarray(img)  

# dataset for training
#The current loader is not using the normalized depth maps for training and test. If you use the normalized depth maps
#(e.g., 0 represents background and 1 represents foreground.), the performance will be further improved.
class SalObjDataset(data.Dataset):
    def __init__(self, image_root, gt_root,depth_root, trainsize):
        self.trainsize = trainsize
        image_root_2= '/home/dataset/HFUT-Train-100-2/train_images_aug/'
        gt_root_2= '/home/dataset/HFUT-Train-100-2/train_masks_aug/'
        depth_root_2= '/home/dataset/HFUT-Train-100-2/train_focal_aug/'
        file_names_1 = os.listdir(image_root)
        file_names_2= os.listdir(image_root_2)

        self.images = []
        self.gts = []
        self.depths= []
        for i, name in enumerate(file_names_1):
            if not name.endswith('.jpg'):
                continue
            self.gts.append(
                os.path.join(gt_root, name[:-4]+'.png')
            )
            
            self.images.append(
                os.path.join(image_root, name)
            )
            self.depths.append(
                os.path.join(depth_root, name[:-4]+'.mat')
            )
        for i, name in enumerate(file_names_2):
            if not name.endswith('.jpg'):
                continue
            self.gts.append(
                os.path.join(gt_root_2, name[:-4]+'.png')
            )
            
            self.images.append(
                os.path.join(image_root_2, name)
            )
            self.depths.append(
                os.path.join(depth_root_2, name[:-4]+'.mat')
            )
        
        self.size = len(self.images)
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        self.depths_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        depth=self.mat_loader(self.depths[index])

        depth1= depth[:,:,0:3]
        depth2= depth[:,:,3:6]
        depth3= depth[:,:,6:9]
        depth4= depth[:,:,9:12]
        depth5= depth[:,:,12:15]
        depth6= depth[:,:,15:18]
        depth7= depth[:,:,18:21]
        depth8= depth[:,:,21:24]
        depth9= depth[:,:,24:27]
        depth10= depth[:,:,27:30]
        depth11= depth[:,:,30:33]
        depth12= depth[:,:,33:36]

        depth1 = Image.fromarray(depth1.astype('uint8')).convert('RGB')
        depth2 = Image.fromarray(depth2.astype('uint8')).convert('RGB')
        depth3 = Image.fromarray(depth3.astype('uint8')).convert('RGB')
        depth4 = Image.fromarray(depth4.astype('uint8')).convert('RGB')
        depth5 = Image.fromarray(depth5.astype('uint8')).convert('RGB')
        depth6 = Image.fromarray(depth6.astype('uint8')).convert('RGB')
        depth7 = Image.fromarray(depth7.astype('uint8')).convert('RGB')
        depth8 = Image.fromarray(depth8.astype('uint8')).convert('RGB')
        depth9 = Image.fromarray(depth9.astype('uint8')).convert('RGB')
        depth10 = Image.fromarray(depth10.astype('uint8')).convert('RGB')
        depth11 = Image.fromarray(depth11.astype('uint8')).convert('RGB')
        depth12 = Image.fromarray(depth12.astype('uint8')).convert('RGB')
        
        image = self.img_transform(image)
        gt = self.gt_transform(gt)

        depth1=self.depths_transform(depth1)
        depth2=self.depths_transform(depth2)
        depth3=self.depths_transform(depth3)
        depth4=self.depths_transform(depth4)
        depth5=self.depths_transform(depth5)
        depth6=self.depths_transform(depth6)
        depth7=self.depths_transform(depth7)
        depth8=self.depths_transform(depth8)
        depth9=self.depths_transform(depth9)
        depth10=self.depths_transform(depth10)
        depth11=self.depths_transform(depth11)
        depth12=self.depths_transform(depth12)

        depth= torch.cat([depth1, depth2, depth3, depth4, depth5, depth6, depth7, depth8, depth9, depth10, depth11, depth12], dim=0)
        
        return image, gt, depth

    def filter_files(self):
        assert len(self.images) == len(self.gts) and len(self.gts)==len(self.images)
        images = []
        gts = []
        depths=[]
        for img_path, gt_path,depth_path in zip(self.images, self.gts, self.depths):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            depth= sio.loadmat(depth_path, verify_compressed_data_integrity=False)
            if img.size == gt.size and gt.size==depth.size:
                images.append(img_path)
                gts.append(gt_path)
                depths.append(depth_path)
        self.images = images
        self.gts = gts
        self.depths=depths

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
    def mat_loader(self, path):
        depth= sio.loadmat(path, verify_compressed_data_integrity=False)
        depth = depth['img']
        # depth = np.array(depth, dtype=np.int32)
        # depth = Image.fromarray(depth.astype('uint8')).convert('RGB')
        return depth

    def resize(self, img, gt, depth):
        assert img.size == gt.size and gt.size==depth.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST),depth.resize((w, h), Image.NEAREST)
        else:
            return img, gt, depth

    def __len__(self):
        return self.size

#dataloader for training
def get_loader(image_root, gt_root,depth_root, batchsize, trainsize, shuffle=True, num_workers=4, pin_memory=True):

    dataset = SalObjDataset(image_root, gt_root, depth_root,trainsize)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader

#test dataset and loader
class Test_dataset(data.Dataset):
    def __init__(self, image_root, gt_root,depth_root, testsize):
        super().__init__()
        self.testsize = testsize
        
        file_names = os.listdir(image_root)
        file_names= sorted(file_names)
        self.images = []
        self.gts = []
        self.depths= []
        for i, name in enumerate(file_names):
            if not name.endswith('.jpg'):
                continue
            self.gts.append(
                os.path.join(gt_root, name[:-4]+'.png')
            )

            self.images.append(
                os.path.join(image_root, name)
            )
            self.depths.append(
                os.path.join(depth_root, name[:-4]+'.mat')
            )
        self.img_transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            # transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        self.depths_transform = transforms.Compose(
            [transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        self.size = len(self.images)

   
    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        depth=self.mat_loader(self.depths[index])
        
        depth1= depth[:,:,0:3]
        depth2= depth[:,:,3:6]
        depth3= depth[:,:,6:9]
        depth4= depth[:,:,9:12]
        depth5= depth[:,:,12:15]
        depth6= depth[:,:,15:18]
        depth7= depth[:,:,18:21]
        depth8= depth[:,:,21:24]
        depth9= depth[:,:,24:27]
        depth10= depth[:,:,27:30]
        depth11= depth[:,:,30:33]
        depth12= depth[:,:,33:36]

        depth1 = Image.fromarray(depth1.astype('uint8')).convert('RGB')
        depth2 = Image.fromarray(depth2.astype('uint8')).convert('RGB')
        depth3 = Image.fromarray(depth3.astype('uint8')).convert('RGB')
        depth4 = Image.fromarray(depth4.astype('uint8')).convert('RGB')
        depth5 = Image.fromarray(depth5.astype('uint8')).convert('RGB')
        depth6 = Image.fromarray(depth6.astype('uint8')).convert('RGB')
        depth7 = Image.fromarray(depth7.astype('uint8')).convert('RGB')
        depth8 = Image.fromarray(depth8.astype('uint8')).convert('RGB')
        depth9 = Image.fromarray(depth9.astype('uint8')).convert('RGB')
        depth10 = Image.fromarray(depth10.astype('uint8')).convert('RGB')
        depth11 = Image.fromarray(depth11.astype('uint8')).convert('RGB')
        depth12 = Image.fromarray(depth12.astype('uint8')).convert('RGB')

        image = self.img_transform(image)
        gt = self.gt_transform(gt)
        #3*256*256
        depth1=self.depths_transform(depth1)
        depth2=self.depths_transform(depth2)
        depth3=self.depths_transform(depth3)
        depth4=self.depths_transform(depth4)
        depth5=self.depths_transform(depth5)
        depth6=self.depths_transform(depth6)
        depth7=self.depths_transform(depth7)
        depth8=self.depths_transform(depth8)
        depth9=self.depths_transform(depth9)
        depth10=self.depths_transform(depth10)
        depth11=self.depths_transform(depth11)
        depth12=self.depths_transform(depth12)

        depth= torch.cat([depth1, depth2, depth3, depth4, depth5, depth6, depth7, depth8, depth9, depth10, depth11, depth12], dim=0)

        name= self.images[index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        
        return image, gt, depth, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
    def mat_loader(self, path):
        depth= sio.loadmat(path, verify_compressed_data_integrity=False)
        depth = depth['img']
        # depth = np.array(depth, dtype=np.int32)
        return depth
    def __len__(self):
        return self.size

