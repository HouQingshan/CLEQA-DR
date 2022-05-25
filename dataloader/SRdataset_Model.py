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

class SRdataset(BaseDataset):
	
	def initialize(self, opt):
		self.opt = opt  # 参数
		self.root = opt.dataroot  # 文件的根路径，并索引文件内容存放的位置
		self.gt_im_paths = os.path.join(opt.dataroot, 'gt_image')
		self.de_im_paths = os.path.join(opt.dataroot, 'de_image')
		self.image_mask_paths = os.path.join(opt.dataroot, 'image_mask')
		self.image_ves_mask_paths = os.path.join(opt.dataroot, 'image_ves_mask')
		
		self.gt_im = make_dataset(self.gt_im_paths)
		self.de_im = make_dataset(self.de_im_paths)
		self.image_mask = make_dataset(self.image_mask_paths)
		self.image_ves_mask = make_dataset(self.image_ves_mask_paths)
		
		self.gt_im = sorted(self.gt_im)
		self.de_im = sorted(self.de_im)
		self.image_mask = sorted(self.image_mask)
		self.image_ves_mask = sorted(self.image_ves_mask)
		
		self.gt_im_size = len(self.gt_im)
		self.de_im_size = len(self.gt_im)
		self.gt_ves_size = len(self.image_ves_mask)
		
		self.transform = get_transform(opt)
	
		# self.degrad = DegradImage(opt)
		# transform_list = [transforms.ToTensor(),
		# transforms.Normalize((0.5, 0.5, 0.5),
		# (0.5, 0.5, 0.5))]
		# self.transform = transforms.Compose(transform_list)
	
	def __getitem__(self, index):
		
		gt_im_path = self.gt_im[index % self.gt_im_size]
		de_im_path = self.de_im[index % self.gt_im_size]
		image_mask_path = self.image_mask[index % self.gt_im_size]
		image_ves_mask_path = self.image_ves_mask[index % self.gt_im_size]
		
		gt_img = Image.open(gt_im_path).convert('RGB')
		gt_img = gt_img.resize((self.opt.loadSizeX, self.opt.loadSizeY), Image.BICUBIC)
		gt_img = self.transform(gt_img)
		gt_img = gt_img.numpy()
		
		de_img = Image.open(de_im_path).convert('RGB')
		de_img = de_img.resize((self.opt.loadSizeX, self.opt.loadSizeY), Image.BICUBIC)
		de_img = self.transform(de_img)
		de_img = de_img.numpy()
		
		mask_region = Image.open(image_mask_path).convert('L')
		mask_region = mask_region.resize((self.opt.loadSizeX, self.opt.loadSizeY), Image.BICUBIC)
		mask_region = np.expand_dims(mask_region, axis=2)
		mask_region = np.array(mask_region, np.float32).transpose(2, 0, 1) / 255.0
		mask_region[mask_region >= 0.5] = 1
		mask_region[mask_region < 0.5] = 0
		
		ves_mask_region = Image.open(image_ves_mask_path).convert('L')
		ves_mask_region = ves_mask_region.resize((self.opt.loadSizeX, self.opt.loadSizeY), Image.BICUBIC)
		ves_mask_region = np.expand_dims(ves_mask_region, axis=2)
		ves_mask_region = np.array(ves_mask_region, np.float32).transpose(2, 0, 1) / 255.0
		ves_mask_region[ves_mask_region >= 0.5] = 1
		ves_mask_region[ves_mask_region < 0.5] = 0
		
		norm_image = de_img * 3.2 - 1.6
		
		return {'de_img': de_img, 'gt_img': gt_img, 'gt_img_path': gt_im_path, 'norm_image': norm_image,
		        'mask_region': mask_region, 'ves_mask_region': ves_mask_region}
		
	def __len__(self):
		return max(self.de_im_size, self.gt_im_size)
		
	def name(self):
		return 'SRdataset'