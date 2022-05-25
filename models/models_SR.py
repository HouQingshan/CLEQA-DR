import torch
import util.util as utilss
import numpy as np
from os.path import join, exists
from torch.autograd import Variable
from models.base_model import BaseModel
from . import networks
from models.losses import init_loss
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM

class model_SR(BaseModel):
	def name(self):
		return 'model_SR'
	
	def initialize(self, opt):
		BaseModel.initialize(self, opt)
		self.isTrain = opt.isTrain
		self.input_gt_img = self.Tensor(opt.batchSize, opt.output_nc, opt.fineSize, opt.fineSize)
		self.input_de_img = self.Tensor(opt.batchSize, opt.output_nc, opt.fineSize, opt.fineSize)
		self.input_img_norm = self.Tensor(opt.batchSize, opt.output_nc, opt.fineSize, opt.fineSize)
		self.input_img_mask = self.Tensor(opt.batchSize, 1, opt.fineSize, opt.fineSize)
		self.input_img_ves_mask = self.Tensor(opt.batchSize, 1, opt.fineSize, opt.fineSize)
		
		if exists(join(opt.project_root, opt.segm_vessel_weight_root)):
			opt.segm_vessel_weight_root = opt.segm_vessel_weight_root
		else:
			opt.segm_vessel_weight_root = 'none'
		self.net = networks.define_net(opt.model_choice, opt.segm_vessel_weight_root, opt.norm, self.gpu_ids)
		
		if not self.isTrain or opt.continue_train:
			self.load_network(self.net, 'SR', opt.which_epoch)
		
		if self.isTrain:
			self.old_lr = opt.lr
			self.optimizer_G = torch.optim.Adam(self.net.parameters(),
			                                    lr=opt.lr, betas=(opt.beta1, 0.999))
			self.content_loss, self.segmen_loss = init_loss(opt, self.Tensor)
	
		print('---------- Networks initialized -------------')
		networks.print_network(self.net)
		print('-----------------------------------------------')
		
	def set_input(self, input):
		# {'de_img': de_img, 'gt_img': gt_img, 'gt_img_path': gt_im_path,
		# 'norm_image': norm_image,'mask_region': mask_region,
		# 'ves_mask_region': ves_mask_region}
		input_de_img = input['de_img']
		input_gt_img = input['gt_img']
		input_norm_image = input['norm_image']
		input_mask_region = input['mask_region']
		input_ves_mask_region = input['ves_mask_region']
		self.image_paths = input['gt_img_path']
		
		self.input_de_img.resize_(input_de_img.size()).copy_(input_de_img)
		self.input_gt_img.resize_(input_gt_img.size()).copy_(input_gt_img)
		self.input_img_norm.resize_(input_norm_image.size()).copy_(input_norm_image)
		self.input_img_mask.resize_(input_mask_region.size()).copy_(input_mask_region)
		self.input_img_ves_mask.resize_(input_ves_mask_region.size()).copy_(input_ves_mask_region)

	def forward(self, opt):
		self.in_de_img = Variable(self.input_de_img)
		self.in_gt_img = Variable(self.input_gt_img)
		self.in_img_norm = Variable(self.input_img_norm)
		self.in_img_mask = Variable(self.input_img_mask)
		self.in_img_ves_mask = Variable(self.input_img_ves_mask)
		self.SR_image, self.ves_mask_pre = self.net.forward(self.in_de_img, self.in_img_norm)
		
	def backward_net(self, opt, epoch):
		self.loss_ves_mask_pre = self.segmen_loss.get_loss(self.in_img_ves_mask * self.in_img_mask, self.ves_mask_pre * self.in_img_mask)

		self.loss_SR_image_pre = self.content_loss.get_loss(self.in_gt_img * self.in_img_mask, self.SR_image * self.in_img_mask)
		self.loss_all = 0.1*self.loss_ves_mask_pre + self.loss_SR_image_pre
		psnr_list = []
		ssim_list = []
		for i in range(len(self.image_paths)):
			in_gt_img = self.in_gt_img[i] * self.in_img_mask[i]
			in_gt_img = torch.unsqueeze(in_gt_img, 0)
			SR_image = self.SR_image[i] * self.in_img_mask[i]
			SR_image = torch.unsqueeze(SR_image, 0)
			in_gt_img = utilss.tensor2im(in_gt_img.data)
			in_gt_img = in_gt_img.astype(np.uint8)
			SR_image = utilss.tensor2im(SR_image.data)
			SR_image = SR_image.astype(np.uint8)
			psnr = PSNR(in_gt_img, SR_image)
			psnr_list.append(psnr)
			ssim = SSIM(in_gt_img, SR_image, channel_axis=-1)#channel_axis=-1 multichannel=True
			ssim_list.append(ssim)

		self.loss_all.backward()
		return psnr_list, ssim_list, self.loss_ves_mask_pre, self.loss_SR_image_pre
	
	def optimize_parameters(self, opt, epoch):
		self.forward(opt)
		self.optimizer_G.zero_grad()
		psnr,ssim,loss_ves,loss_SR=self.backward_net(opt, epoch)
		self.optimizer_G.step()
		return psnr,ssim,loss_ves,loss_SR

	def save(self, opt, label):
		self.save_network(self.net, 'SR', label, self.gpu_ids)
	
	def update_learning_rate(self):
		lrd = self.opt.lr / self.opt.niter_decay
		lr = self.old_lr - lrd
		for param_group in self.optimizer_G.param_groups:
			param_group['lr'] = lr
		print('update learning rate: %f -> %f' % (self.old_lr, lr))
		self.old_lr = lr
