import math
import torchvision.transforms.functional as F
import numpy as np
import cv2
import os.path
import random
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset, make_label_dataset
from data.base_dataset import BaseDataset, get_transform
from PIL import Image

class CLSdataset(BaseDataset):
	
	def initialize(self, opt):
		self.opt = opt  # 参数
		self.root = opt.dataroot  # 文件的根路径，并索引文件内容存放的位置
		self.gt_im_paths = os.path.join(opt.dataroot, 'image')
		# self.gt_spot_paths = os.path.join(opt.dataroot, 'mask_spot')
		# self.gt_blur_paths = os.path.join(opt.dataroot, 'gt_img_blur')
		self.region_paths = os.path.join(opt.dataroot, 'image_mask')
		self.dir_CLS_label = os.path.join(opt.dataroot, 'CLS_label')
		self.labels_records = make_label_dataset(self.dir_CLS_label)  # disc region
		
		self.transform = get_transform(opt)
	
		# self.degrad = DegradImage(opt)
		# transform_list = [transforms.ToTensor(),
		# transforms.Normalize((0.5, 0.5, 0.5),
		# (0.5, 0.5, 0.5))]
		# self.transform = transforms.Compose(transform_list)
	def __getitem__(self, index):
		gt_im_path = os.path.join(self.gt_im_paths, self.labels_records.iloc[index].image)+'.png'
		region_path = os.path.join(self.region_paths, self.labels_records.iloc[index].image)+'.png'
		gt_img = Image.open(gt_im_path).convert('RGB')
		gt_img = gt_img.resize((self.opt.loadSizeX, self.opt.loadSizeY), Image.BICUBIC)  # 通过线性插值进行图片的重新放缩
		gt_img = self.transform(gt_img)
		gt_img = gt_img.numpy()
		
		region = Image.open(region_path).convert('L')
		region = region.resize((self.opt.loadSizeX, self.opt.loadSizeY), Image.BICUBIC)
		region = np.expand_dims(region, axis=2)
		region = np.array(region, np.float32).transpose(2, 0, 1) / 255.0
		region[region >= 0.5] = 1
		region[region < 0.5] = 0
		
		labels_cls = torch.tensor(self.labels_records.iloc[index].DR_grade)
		
		return {'gt_img': gt_img, 'gt_im_paths': gt_im_path, 'region': region, 'labels_cls': labels_cls}
	
	def __len__(self):
		return len(self.labels_records)
	
	def name(self):
		return 'CLSdataset'