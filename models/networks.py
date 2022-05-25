import torch
import torch.nn as nn
import functools
import torch.nn.functional as F

from functools import partial
from torchvision import models
from typing import Type, Any, Callable, Union, List, Optional
from torch import Tensor
nonlinearity = partial(F.relu, inplace=True)

# 进行卷积和正则化层的参数初始化
def weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		m.weight.data.normal_(0.0, 0.02)
		if hasattr(m.bias, 'data'):
			m.bias.data.fill_(0)
	elif classname.find('BatchNorm2d') != -1:
		m.weight.data.normal_(1.0, 0.02)
		m.bias.data.fill_(0)


# 得到正则层
def get_norm_layer(norm_type='instance'):
	if norm_type == 'batch':
		norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
	elif norm_type == 'instance':
		norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
	else:
		raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
	return norm_layer


# 载入训练好的私人权重,k[7:]
def load_pretrained_personal_vesmask_model(model, path, use_gpu):
	if use_gpu:
		pretrain_dict = torch.load(path)
	else:
		pretrain_dict = torch.load(path, map_location=torch.device('cpu'))
	model_dict = {}
	state_dict = model.state_dict()
	for k, v in pretrain_dict.items():
		k_ori = k[7:]
		if k_ori in state_dict:
			model_dict[k_ori] = v
	state_dict.update(model_dict)
	model.load_state_dict(state_dict)

# # 载入训练好的私人多卡权重,'_'+k[7:]
# def load_pretrained_personal_weight_with_muli_gpu(model, path, use_gpu):
# 	if use_gpu:
# 		pretrain_dict = torch.load(path)
# 	else:
# 		pretrain_dict = torch.load(path, map_location=torch.device('cpu'))
# 	model_dict = {}
# 	state_dict = model.state_dict()
# 	for k, v in pretrain_dict.items():
# 		k_ori = '_' + k[7:]
# 		if k_ori in state_dict:
# 			model_dict[k_ori] = v
# 	state_dict.update(model_dict)
# 	model.load_state_dict(state_dict)

# 载入训练好的私人权重,k
def load_pretrained_personal_model(model, path, use_gpu):
	if use_gpu:
		pretrain_dict = torch.load(path)
	else:
		pretrain_dict = torch.load(path, map_location=torch.device('cpu'))
	model_dict = {}
	state_dict = model.state_dict()
	for k, v in pretrain_dict.items():
		if k in state_dict:
			model_dict[k] = v
	state_dict.update(model_dict)
	model.load_state_dict(state_dict)

# 载入预训练的imageNet权重
def load_pretrained_resnet_weight(model, use_gpu):
	params = {
		"model_pre": "resnet50"
	}
	if use_gpu:
		pretrain_dict = getattr(models, params["model_pre"])(pretrained=True)
		pretrain_dicts = pretrain_dict.state_dict()
	model_dict = {}
	state_dict = model.state_dict()
	for k, v in pretrain_dicts.items():
		if k in state_dict:
			model_dict[k] = v
	state_dict.update(model_dict)
	model.load_state_dict(state_dict)

# 载入预训练不同键值的imageNet权重
def load_pretrained_resnet_updata_weight(model, use_gpu):
	params = {
		"model_pre": "resnet50"
	}
	if use_gpu:
		pretrain_dict = getattr(models, params["model_pre"])(pretrained=True)
		pretrain_dicts = pretrain_dict.state_dict()

	model_dict = {}
	state_dict = model.state_dict()
	for k, v in pretrain_dicts.items():
		k_up = '_' + k
		if k_up in state_dict:
			model_dict[k_up] = v
	state_dict.update(model_dict)
	model.load_state_dict(state_dict)

def print_network(net):
	num_params = 0
	for param in net.parameters():
		num_params += param.numel()
	print(net)
	print('Total number of parameters: %d' % num_params)


def define_net(model_choice, segmdataroot, norm='batch', gpu_ids=[]):
	net = None
	use_gpu = len(gpu_ids) > 0
	norm_layer = get_norm_layer(norm_type=norm)
	if use_gpu:
		assert (torch.cuda.is_available())
	if model_choice == 'EyesImage_IQA_with_masklabel':
		net = EyesImage_IQA_with_masklabel(norm_layer=norm_layer)
	elif model_choice == 'EyesImage_IQA_with_image_blur_label':
		net = EyesImage_IQA_with_image_blur_label(norm_layer=norm_layer)
	elif model_choice == 'EyesImage_IQAresnet_with_masklabel':
		net = EyesImage_IQAresnet_with_masklabel(norm_layer=norm_layer)
	elif model_choice == 'EyesImage_IQAresnet_with_image_blur_label':
		net = EyesImage_IQAresnet_with_image_blur_label(norm_layer=norm_layer)
	elif model_choice == 'EyesImage_CLS_with_resnet':
		net = EyesImage_CLS_with_resnet(norm_layer=norm_layer)
	elif model_choice == 'EyesImage_CLS_with_MyNet':
		net = EyesImage_CLS_with_MyNet(norm_layer=norm_layer)
	elif model_choice == 'EyesImage_SR':
		net = EyesImage_SR(norm_layer=norm_layer)
	elif model_choice == 'EyesImage_joint':
		net = EyesImage_joint(norm_layer=norm_layer)
	else:
		raise NotImplementedError('Generator model name [%s] is not recognized' % model_choice)
	
	# if len(gpu_ids) > 0:
	# 	net.cuda(gpu_ids[0])
	# net.apply(weights_init)
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	if torch.cuda.device_count() > 1:
		net = nn.DataParallel(net)
	net.to(device)
	
	# net.cuda()
	# torch.nn.DataParallel(net, device_ids=[0,1,2,3]) #range(torch.cuda.device_count())
	# net.apply(weights_init)
	
	if model_choice == 'EyesImage_IQA_with_masklabel':
		if segmdataroot == './weights/Spot.th':
			load_pretrained_personal_model(net, segmdataroot, use_gpu)
			weights_rootIQA = './weights/MyNet_IQA.th'
			load_pretrained_personal_model(net, weights_rootIQA, use_gpu)
			
	elif model_choice == 'EyesImage_IQA_with_image_blur_label':
		if segmdataroot == './weights/SR_blur.th':
			# load_pretrained_personal_model(net, segmdataroot, use_gpu)
			weights_rootIQA = './weights/MyNet_IQA.th'
			# load_pretrained_personal_model(net, weights_rootIQA, use_gpu)
			
	elif model_choice == 'EyesImage_IQAresnet_with_masklabel':
		if segmdataroot == './weights/Spot.th':
			load_pretrained_personal_model(net, segmdataroot, use_gpu)
			weights_rootIQA = './weights/resnet_IQA.th'
			load_pretrained_personal_model(net, weights_rootIQA, use_gpu)

	# IQA&blur
	elif model_choice == 'EyesImage_IQAresnet_with_image_blur_label':
		if segmdataroot == './weights/SR_blur.th':
			load_pretrained_personal_model(net, segmdataroot, use_gpu)
			weights_rootIQA = './weights/resnet_IQA.th'
			load_pretrained_personal_model(net, weights_rootIQA, use_gpu)
		else:
			load_pretrained_resnet_updata_weight(net, use_gpu)
	###############
	# CLS
	elif model_choice == 'EyesImage_CLS_with_resnet':
		if segmdataroot == './weights/CLS_resnet.th':
			load_pretrained_personal_model(net,segmdataroot, use_gpu)
		else:
			load_pretrained_resnet_weight(net, use_gpu)
	
	elif model_choice == 'EyesImage_CLS_with_MyNet':
		if segmdataroot == './weights/CLS_MyNet.th':
			load_pretrained_personal_model(net, segmdataroot, use_gpu)

	# SR
	elif model_choice == 'EyesImage_SR':
		if segmdataroot == './weights/Vessel.th':
			load_pretrained_personal_vesmask_model(net, segmdataroot, use_gpu)
			# weights_rootSR = './weights/SR_branch.th'
			# load_pretrained_personal_model(net, weights_rootSR, use_gpu)

	# joint
	elif model_choice == 'EyesImage_joint':
		weightCLSroot = './weights/CLS_res.pth'
		load_pretrained_personal_model(net, weightCLSroot, use_gpu)
		weightSRroot = './weights/SR.pth'
		load_pretrained_personal_model(net, weightSRroot, use_gpu)
		weightIQAroot = './weights/IQAres_blur.pth'
		load_pretrained_personal_model(net, weightIQAroot, use_gpu)
		
	return net

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, dilation: int = 1) -> nn.Conv2d:
	return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
					 padding=dilation, bias=False, dilation=dilation)

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
	return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
	expansion: int = 1
	
	def __init__(
			self,
			inplanes: int,
			planes: int,
			stride: int = 1,
			downsample: Optional[nn.Module] = None,
			norm_layer: Optional[Callable[..., nn.Module]] = None
	) -> None:
		super(BasicBlock, self).__init__()
		if norm_layer is None:
			norm_layer = nn.BatchNorm2d
		
		self.conv1 = conv3x3(inplanes, planes, stride)
		self.bn1 = norm_layer(planes)
		self.relu = nn.ReLU(inplace=True)
		self.conv2 = conv3x3(planes, planes)
		self.bn2 = norm_layer(planes)
		self.downsample = downsample
		self.stride = stride
	
	def forward(self, x: Tensor) -> Tensor:
		identity = x
		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)
		out = self.conv2(out)
		out = self.bn2(out)
		
		if self.downsample is not None:
			identity = self.downsample(x)
		
		out += identity
		out = self.relu(out)
		
		return out

class Bottleneck(nn.Module):
	expansion: int = 4
	
	def __init__(
			self,
			inplanes: int,
			planes: int,
			stride: int = 1,
			downsample: Optional[nn.Module] = None,
			norm_layer: Optional[Callable[..., nn.Module]] = None
	) -> None:
		
		super(Bottleneck, self).__init__()
		if norm_layer is None:
			norm_layer = nn.BatchNorm2d
		width = planes
		
		self.conv1 = conv1x1(inplanes, width)
		self.bn1 = norm_layer(width)
		self.conv2 = conv3x3(width, width, stride)
		self.bn2 = norm_layer(width)
		self.conv3 = conv1x1(width, planes * self.expansion)
		self.bn3 = norm_layer(planes * self.expansion)
		self.relu = nn.ReLU(inplace=True)
		self.downsample = downsample
		self.stride = stride
	
	def forward(self, x: Tensor) -> Tensor:
		identity = x
		
		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)
		out = self.conv2(out)
		out = self.bn2(out)
		out = self.relu(out)
		out = self.conv3(out)
		out = self.bn3(out)
		
		if self.downsample is not None:
			identity = self.downsample(x)
		
		out += identity
		out = self.relu(out)
		
		return out
# 基于spot_mask label以及MyNet的质量评估
class EyesImage_IQA_with_masklabel(nn.Module):
	def __init__(self, norm_layer=nn.BatchNorm2d):
		super(EyesImage_IQA_with_masklabel, self).__init__()
		
		if type(norm_layer) == functools.partial:
			use_bias = norm_layer.func == nn.BatchNorm2d
		else:
			use_bias = norm_layer == nn.BatchNorm2d
			
		#######
		filters = [64, 128, 256, 512]
		resnet = models.resnet34(pretrained=True)
		self._firstconv = resnet.conv1
		self._firstbn = resnet.bn1
		self._firstrelu = resnet.relu
		self._firstmaxpool = resnet.maxpool
		self._encoder1 = resnet.layer1
		self._encoder2 = resnet.layer2
		self._encoder3 = resnet.layer3
		self._encoder4 = resnet.layer4
		
		self._dblock = DACblock(512)
		self._spp = SPPblock(512)
		
		self._decoder4 = DecoderBlock(516, filters[2])
		self._decoder3 = DecoderBlock(filters[2], filters[1])
		self._decoder2 = DecoderBlock(filters[1], filters[0])
		self._decoder1 = DecoderBlock(filters[0], filters[0])
		
		self._finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
		self._finalrelu1 = nonlinearity
		self._finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
		self._finalrelu2 = nonlinearity
		self._finalconv3 = nn.Conv2d(32, 1, 3, padding=1)
		
		#######
		self._SG_conv1 = nn.Conv2d(6, 32, 5, padding=2, bias=use_bias)
		self._SG_relu1 = nn.ReLU()
		self._SG_conv1_1 = ResBlock(in_channels=32, in_kernel=5, in_pad=2, in_bias=use_bias)
		self._SG_conv1_2 = ResBlock(in_channels=32, in_kernel=5, in_pad=2, in_bias=use_bias)
		self._SG_conv1_3 = ResBlock(in_channels=32, in_kernel=5, in_pad=2, in_bias=use_bias)
		
		self._SG_conv2_0 = nn.Conv2d(32, 64, 5, padding=2, stride=2, bias=use_bias)
		self._SG_relu2_0 = nn.ReLU()
		
		self._SG_conv2 = nn.Conv2d(128, 64, 5, padding=2, bias=use_bias)
		self._SG_relu2 = nn.ReLU()
		self._SG_conv2_1 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
		self._SG_conv2_2 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
		self._SG_conv2_3 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
		
		self._SG_conv3_0 = nn.Conv2d(64, 64, 5, padding=2, stride=2, bias=use_bias)
		self._SG_relu3_0 = nn.ReLU()
		
		self._SG_conv3 = nn.Conv2d(128, 64, 5, padding=2, bias=use_bias)
		self._SG_relu3 = nn.ReLU()
		self._SG_conv3_1 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
		self._SG_conv3_2 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
		self._SG_conv3_3 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
		
		self._SG_conv4_0 = nn.Conv2d(64, 128, 5, padding=2, stride=2, bias=use_bias)
		self._SG_relu4_0 = nn.ReLU()
		
		self._SG_conv4 = nn.Conv2d(256, 128, 5, padding=2, bias=use_bias)
		self._SG_relu4 = nn.ReLU()
		self._SG_conv4_1 = ResBlock(in_channels=128, in_kernel=5, in_pad=2, in_bias=use_bias)
		self._SG_conv4_2 = ResBlock(in_channels=128, in_kernel=5, in_pad=2, in_bias=use_bias)
		self._SG_conv4_3 = ResBlock(in_channels=128, in_kernel=5, in_pad=2, in_bias=use_bias)
		
		self._SG_conv5 = nn.Conv2d(128, 256, 5, padding=2, stride=2, bias=use_bias)
		self._SG_relu5 = nn.ReLU()
		
		self._SG_conv6 = nn.Conv2d(256, 512, 5, padding=2, stride=2, bias=use_bias)
		self._SG_relu6 = nn.ReLU()
		
		self._Savgpool = nn.AdaptiveAvgPool2d((1, 1))
		self._Sfc = nn.Linear(512, 5)
		
	def forward(self, input_512, input_norm): # input_norm & input_512 2 3 512 512
		### part1
		x = self._firstconv(input_norm) # x 2 64 256 256
		x = self._firstbn(x)      # x 2 64 256 256
		x = self._firstrelu(x)    # x 2 64 256 256
		x = self._firstmaxpool(x) # x 2 64 128 128
		
		e1 = self._encoder1(x)  # e1 2 64 128 128
		e2 = self._encoder2(e1) # e2 2 128 64 64
		e3 = self._encoder3(e2) # e3 2 256 32 32
		e4 = self._encoder4(e3) # e4 2 512 16 16
		
		# Center
		e4 = self._dblock(e4) # e4 2 512 16 16
		e4 = self._spp(e4)    # e4 2 512 16 16
		
		d4 = self._decoder4(e4) + e3 # d4 2 256 32 32
		d3 = self._decoder3(d4) + e2 # d3 2 128 64 64
		d2 = self._decoder2(d3) + e1 # d2 2 64 128 128
		d1 = self._decoder1(d2)      # d1 2 64 256 256
		
		out = self._finaldeconv1(d1) # 2 32 512 512
		out = self._finalrelu1(out)  # 2 32 512 512
		out = self._finalconv2(out)  # 2 32 512 512
		out = self._finalrelu2(out)  # 2 32 512 512
		out = self._finalconv3(out)  # 2 1 512 512
		
		out = torch.sigmoid(out)    # 2 1 512 512
		
		### part2
		
		input_copy_512 = torch.cat([input_512, input_512], 1) # input_copy_512 2 6 512 512
		x = self._SG_conv1(input_copy_512) # x 2 32 512 512
		x = self._SG_relu1(x)              # x 2 32 512 512
		x = self._SG_conv1_1(x)            # x 2 32 512 512
		x = self._SG_conv1_2(x)            # x 2 32 512 512
		x_512 = self._SG_conv1_3(x)        # x_512 2 32 512 512
		
		x = self._SG_conv2_0(x_512)        # x 2 64 256 256
		x = self._SG_relu2_0(x)            # x 2 64 256 256
		con_2 = torch.cat([x, d1], 1)    # con_2 2 128 256 256
		x = self._SG_conv2(con_2)          # x 2 64 256 256
		x = self._SG_relu2(x)              # x 2 64 256 256
		x = self._SG_conv2_1(x)            # x 2 64 256 256
		x = self._SG_conv2_2(x)            # x 2 64 256 256
		x_256 = self._SG_conv2_3(x)        # x_256 2 64 256 256
		
		x = self._SG_conv3_0(x_256)        # x 2 64 128 128
		x = self._SG_relu3_0(x)            # x 2 64 128 128
		con_4 = torch.cat([x, d2], 1)    # con_4 2 128 128 128
		x = self._SG_conv3(con_4)          # x 2 64 128 128
		x = self._SG_relu3(x)              # x 2 64 128 128
		x = self._SG_conv3_1(x)            # x 2 64 128 128
		x = self._SG_conv3_2(x)            # x 2 64 128 128
		x_128 = self._SG_conv3_3(x)        # x_128 2 64 128 128
		
		x = self._SG_conv4_0(x_128)        # x 2 128 64 64
		x = self._SG_relu4_0(x)            # x 2 128 64 64
		con_8 = torch.cat([x, d3], 1)    # x 2 256 64 64
		x = self._SG_conv4(con_8)          # x 2 128 64 64
		x = self._SG_relu4(x)              # x 2 128 64 64
		x = self._SG_conv4_1(x)            # x 2 128 64 64
		x = self._SG_conv4_2(x)            # x 2 128 64 64
		x = self._SG_conv4_3(x)            # x 2 128 64 64 -> 2 524288
		
		x = self._SG_conv5(x) # 2 256 32 32
		x = self._SG_relu5(x) # 2 256 32 32
		
		x = self._SG_conv6(x) # 2 512 16 16
		x = self._SG_relu6(x) # 2 512 16 16
		
		x = self._Savgpool(x) # 2 512 1 1
		x = torch.flatten(x, 1) # 2 512
		label_pre = self._Sfc(x)
		
		return label_pre, out
# 基于blur_image label以及MyNet的质量评估
class EyesImage_IQA_with_image_blur_label(nn.Module):
	def __init__(self, norm_layer=nn.BatchNorm2d):
		super(EyesImage_IQA_with_image_blur_label, self).__init__()
		if type(norm_layer) == functools.partial:
			use_bias = norm_layer.func == nn.BatchNorm2d
		else:
			use_bias = norm_layer == nn.BatchNorm2d
		#######
		filters = [64, 128, 256, 512]
		resnet = models.resnet34(pretrained=True)
		self._firstconv = resnet.conv1
		self._firstbn = resnet.bn1
		self._firstrelu = resnet.relu
		self._firstmaxpool = resnet.maxpool
		self._encoder1 = resnet.layer1
		self._encoder2 = resnet.layer2
		self._encoder3 = resnet.layer3
		self._encoder4 = resnet.layer4
		
		self._dblock = DACblock(512)
		self._spp = SPPblock(512)
		
		self._decoder4 = DecoderBlock(516, filters[2])
		self._decoder3 = DecoderBlock(filters[2], filters[1])
		self._decoder2 = DecoderBlock(filters[1], filters[0])
		self._decoder1 = DecoderBlock(filters[0], filters[0])
		
		self._finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
		self._finalrelu1 = nonlinearity
		self._finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
		self._finalrelu2 = nonlinearity
		self._finalconv3 = nn.Conv2d(32, 3, 3, padding=1)
		
		#######
		self._SG_conv1 = nn.Conv2d(6, 32, 5, padding=2, bias=use_bias)
		self._SG_relu1 = nn.ReLU()
		self._SG_conv1_1 = ResBlock(in_channels=32, in_kernel=5, in_pad=2, in_bias=use_bias)
		self._SG_conv1_2 = ResBlock(in_channels=32, in_kernel=5, in_pad=2, in_bias=use_bias)
		self._SG_conv1_3 = ResBlock(in_channels=32, in_kernel=5, in_pad=2, in_bias=use_bias)
		
		self._SG_conv2_0 = nn.Conv2d(32, 64, 5, padding=2, stride=2, bias=use_bias)
		self._SG_relu2_0 = nn.ReLU()
		
		self._SG_conv2 = nn.Conv2d(128, 64, 5, padding=2, bias=use_bias)
		self._SG_relu2 = nn.ReLU()
		self._SG_conv2_1 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
		self._SG_conv2_2 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
		self._SG_conv2_3 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
		
		self._SG_conv3_0 = nn.Conv2d(64, 64, 5, padding=2, stride=2, bias=use_bias)
		self._SG_relu3_0 = nn.ReLU()
		
		self._SG_conv3 = nn.Conv2d(128, 64, 5, padding=2, bias=use_bias)
		self._SG_relu3 = nn.ReLU()
		self._SG_conv3_1 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
		self._SG_conv3_2 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
		self._SG_conv3_3 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
		
		self._SG_conv4_0 = nn.Conv2d(64, 128, 5, padding=2, stride=2, bias=use_bias)
		self._SG_relu4_0 = nn.ReLU()
		
		self._SG_conv4 = nn.Conv2d(256, 128, 5, padding=2, bias=use_bias)
		self._SG_relu4 = nn.ReLU()
		self._SG_conv4_1 = ResBlock(in_channels=128, in_kernel=5, in_pad=2, in_bias=use_bias)
		self._SG_conv4_2 = ResBlock(in_channels=128, in_kernel=5, in_pad=2, in_bias=use_bias)
		self._SG_conv4_3 = ResBlock(in_channels=128, in_kernel=5, in_pad=2, in_bias=use_bias)
		
		self._SG_conv5_0 = nn.Conv2d(128, 256, 5, padding=2, stride=2, bias=use_bias)
		self._SG_relu5_0 = nn.ReLU()
		
		self._SG_conv5 = nn.Conv2d(512, 256, 5, padding=2, bias=use_bias)
		self._SG_relu5 = nn.ReLU()
		self._SG_conv5_1 = ResBlock(in_channels=256, in_kernel=5, in_pad=2, in_bias=use_bias)
		self._SG_conv5_2 = ResBlock(in_channels=256, in_kernel=5, in_pad=2, in_bias=use_bias)
		self._SG_conv5_3 = ResBlock(in_channels=256, in_kernel=5, in_pad=2, in_bias=use_bias)
		
		self._SG_conv6 = nn.Conv2d(256, 512, 5, padding=2, stride=2, bias=use_bias)
		self._SG_relu6 = nn.ReLU()
		
		self._Savgpool = nn.AdaptiveAvgPool2d((1, 1))
		self._Sfc = nn.Linear(512, 5)
	
	def forward(self, input_512):
		###
		x = self._firstconv(input_512)  # x 2 64 256 256
		x = self._firstbn(x)  # x 2 64 256 256
		x = self._firstrelu(x)  # x 2 64 256 256
		x = self._firstmaxpool(x)  # x 2 64 128 128
		
		e1 = self._encoder1(x)  # e1 2 64 128 128
		e2 = self._encoder2(e1)  # e2 2 128 64 64
		e3 = self._encoder3(e2)  # e3 2 256 32 32
		e4 = self._encoder4(e3)  # e4 2 512 16 16
		
		# Center
		e4 = self._dblock(e4)  # e4 2 512 16 16
		e4 = self._spp(e4)  # e4 2 512 16 16
		
		d4 = self._decoder4(e4) + e3  # d4 2 256 32 32
		d3 = self._decoder3(d4) + e2  # d3 2 128 64 64
		d2 = self._decoder2(d3) + e1  # d2 2 64 128 128
		d1 = self._decoder1(d2)  # d1 2 64 256 256
		
		out = self._finaldeconv1(d1)  # 2 32 512 512
		out = self._finalrelu1(out)  # 2 32 512 512
		out = self._finalconv2(out)  # 2 32 512 512
		out = self._finalrelu2(out)  # 2 32 512 512
		out = self._finalconv3(out)  # 2 3 512 512
		out = out + input_512
		out = torch.sigmoid(out)  # 2 3 512 512
		
		input_copy_512 = torch.cat([input_512, input_512], 1)  # input_copy_512 2 6 512 512
		x = self._SG_conv1(input_copy_512)  # x 2 32 512 512
		x = self._SG_relu1(x)  # x 2 32 512 512
		x = self._SG_conv1_1(x)  # x 2 32 512 512
		x = self._SG_conv1_2(x)  # x 2 32 512 512
		x_512 = self._SG_conv1_3(x)  # x_512 2 32 512 512
		
		x = self._SG_conv2_0(x_512)  # x 2 64 256 256
		x = self._SG_relu2_0(x)  # x 2 64 256 256
		con_2 = torch.cat([x, d1], 1)  # con_2 2 128 256 256
		x = self._SG_conv2(con_2)  # x 2 64 256 256
		x = self._SG_relu2(x)  # x 2 64 256 256
		x = self._SG_conv2_1(x)  # x 2 64 256 256
		x = self._SG_conv2_2(x)  # x 2 64 256 256
		x_256 = self._SG_conv2_3(x)  # x_256 2 64 256 256
		
		x = self._SG_conv3_0(x_256)  # x 2 64 128 128
		x = self._SG_relu3_0(x)  # x 2 64 128 128
		con_4 = torch.cat([x, d2], 1)  # con_4 2 128 128 128
		x = self._SG_conv3(con_4)  # x 2 64 128 128
		x = self._SG_relu3(x)  # x 2 64 128 128
		x = self._SG_conv3_1(x)  # x 2 64 128 128
		x = self._SG_conv3_2(x)  # x 2 64 128 128
		x_128 = self._SG_conv3_3(x)  # x_128 2 64 128 128
		
		x = self._SG_conv4_0(x_128)  # x 2 128 64 64
		x = self._SG_relu4_0(x)  # x 2 128 64 64
		con_8 = torch.cat([x, d3], 1)  # x 2 256 64 64
		x = self._SG_conv4(con_8)  # x 2 128 64 64
		x = self._SG_relu4(x)  # x 2 128 64 64
		x = self._SG_conv4_1(x)  # x 2 128 64 64
		x = self._SG_conv4_2(x)  # x 2 128 64 64
		x_64 = self._SG_conv4_3(x)  # x 2 128 64 64 -> 2 524288
		
		x = self._SG_conv5_0(x_64)  # 2 256 32 32
		x = self._SG_relu5_0(x)  # 2 256 32 32
		con_16 = torch.cat([x, d4], 1)
		x = self._SG_conv5(con_16)
		x = self._SG_relu5(x)
		x = self._SG_conv5_1(x)
		x = self._SG_conv5_2(x)
		x = self._SG_conv5_3(x)
		
		x = self._SG_conv6(x)  # 2 512 16 16
		x = self._SG_relu6(x)  # 2 512 16 16
		
		x = self._Savgpool(x)  # 2 512 1 1
		x = torch.flatten(x, 1)  # 2 512
		label_pre = self._Sfc(x)
		
		return label_pre, out
# 基于spot_mask label以及Resnet的质量评估
class EyesImage_IQAresnet_with_masklabel(nn.Module):
	def __init__(self, norm_layer=nn.BatchNorm2d):
		super(EyesImage_IQAresnet_with_masklabel, self).__init__()
		if type(norm_layer) == functools.partial:
			norm_layer = nn.BatchNorm2d
		self._norm_layer = norm_layer
		self.inplanes = 64
		layers = [3, 4, 6, 3]
		block = Bottleneck
		
		self._conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
		self._bn1 = norm_layer(self.inplanes)
		self._relu = nn.ReLU(inplace=True)
		self._maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
		self._layer1 = self._make_layer(block, 64, layers[0])
		self._layer2 = self._make_layer(block, 128, layers[1], stride=2)
		self._layer3 = self._make_layer(block, 256, layers[2], stride=2)
		self._layer4 = self._make_layer(block, 512, layers[3], stride=2)
		self._avgpool = nn.AdaptiveAvgPool2d((1, 1))
		self._fc1 = nn.Linear(512 * block.expansion, 5)
		
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
			elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)
		
		#######
		filters = [64, 128, 256, 512]
		resnet = models.resnet34(pretrained=True)
		self._firstconv = resnet.conv1
		self._firstbn = resnet.bn1
		self._firstrelu = resnet.relu
		self._firstmaxpool = resnet.maxpool
		self._encoder1 = resnet.layer1
		self._encoder2 = resnet.layer2
		self._encoder3 = resnet.layer3
		self._encoder4 = resnet.layer4
		
		self._dblock = DACblock(512)
		self._spp = SPPblock(512)
		
		self._decoder4 = DecoderBlock(516, filters[2])
		self._decoder3 = DecoderBlock(filters[2], filters[1])
		self._decoder2 = DecoderBlock(filters[1], filters[0])
		self._decoder1 = DecoderBlock(filters[0], filters[0])
		
		self._finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
		self._finalrelu1 = nonlinearity
		self._finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
		self._finalrelu2 = nonlinearity
		self._finalconv3 = nn.Conv2d(32, 1, 3, padding=1)
		
		#####
		self.equal1 = nn.Conv2d(128, 64, 3, padding=1)
		self.equal2 = nn.Conv2d(320, 256, 3, padding=1)
		self.equal3 = nn.Conv2d(640, 512, 3, padding=1)
		self.equal4 = nn.Conv2d(1280, 1024, 3, padding=1)
		
	def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
					stride: int = 1) -> nn.Sequential:
		norm_layer = self._norm_layer
		downsample = None
		
		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(
				conv1x1(self.inplanes, planes * block.expansion, stride),
				norm_layer(planes * block.expansion),
			)
		
		layers = []
		layers.append(block(self.inplanes, planes, stride, downsample, norm_layer))
		
		self.inplanes = planes * block.expansion
		
		for _ in range(1, blocks):
			layers.append(block(self.inplanes, planes, norm_layer=norm_layer))
		
		return nn.Sequential(*layers)
	
	def forward(self, input_512, input_norm):
		###
		x = self._firstconv(input_norm)  # x 2 64 256 256
		x = self._firstbn(x)  # x 2 64 256 256
		x = self._firstrelu(x)  # x 2 64 256 256
		x = self._firstmaxpool(x)  # x 2 64 128 128
		
		e1 = self._encoder1(x)  # e1 2 64 128 128
		e2 = self._encoder2(e1)  # e2 2 128 64 64
		e3 = self._encoder3(e2)  # e3 2 256 32 32
		e4 = self._encoder4(e3)  # e4 2 512 16 16
		
		#
		e4 = self._dblock(e4)  # e4 2 512 16 16
		e4 = self._spp(e4)  # e4 2 512 16 16
		
		d4 = self._decoder4(e4) + e3  # d4 2 256 32 32
		d3 = self._decoder3(d4) + e2  # d3 2 128 64 64
		d2 = self._decoder2(d3) + e1  # d2 2 64 128 128
		d1 = self._decoder1(d2)  # d1 2 64 256 256
		
		out = self._finaldeconv1(d1)  # 2 32 512 512
		out = self._finalrelu1(out)  # 2 32 512 512
		out = self._finalconv2(out)  # 2 32 512 512
		out = self._finalrelu2(out)  # 2 32 512 512
		out = self._finalconv3(out)  # 2 1 512 512
		
		out = torch.sigmoid(out)  # 2 1 512 512
		
		#####################
		x1 = self._conv1(input_512) # x1 2 64 256 256
		x1 = self._bn1(x1) # x1 2 64 256 256
		x1 = self._relu(x1) # x1 2 64 256 256  # d1 2 64 256 256
		cat0 = torch.cat([x1, d1], 1) # 2 128 256 256
		x1 = self.equal1(cat0) # x1 2 64 256 256
		x1 = self._maxpool(x1) # x1: 2 64 128 128
		
		x1 = self._layer1(x1) # x1 2 256 128 128  # d2 2 64 128 128
		cat1 = torch.cat([x1, d2], 1) # cat1 2 320 128 128
		x1 = self.equal2(cat1)
		
		x1 = self._layer2(x1) # x1 2 512 64 64  # d3 2 128 64 64
		cat2 = torch.cat([x1, d3], 1)  # cat2 2 640 128 128
		x1 = self.equal3(cat2)
		
		x1 = self._layer3(x1) # x1 2 1024 32 32  # d4 2 256 32 32
		cat3 = torch.cat([x1, d4], 1)  # cat3 2 1280 128 128
		x1 = self.equal4(cat3)
		
		x1 = self._layer4(x1) # x1 2 2048 16 16
		x1 = self._avgpool(x1) # x1 2 2048 1 1
		x1 = torch.flatten(x1, 1) # x1 2 2048
		label_pre = self._fc1(x1) # x1
		return label_pre, out
# 基于blur_image label以及Resnet的质量评估
class EyesImage_IQAresnet_with_image_blur_label(nn.Module):
	def __init__(self, norm_layer=nn.BatchNorm2d):
		super(EyesImage_IQAresnet_with_image_blur_label, self).__init__()
		if type(norm_layer) == functools.partial:
			norm_layer = nn.BatchNorm2d
		self._norm_layer = norm_layer
		self.inplanes = 64
		layers = [3, 4, 6, 3]
		block = Bottleneck
		
		self._conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
		self._bn1 = norm_layer(self.inplanes)
		self._relu = nn.ReLU(inplace=True)
		self._maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
		self._layer1 = self._make_layer(block, 64, layers[0])
		self._layer2 = self._make_layer(block, 128, layers[1], stride=2)
		self._layer3 = self._make_layer(block, 256, layers[2], stride=2)
		self._layer4 = self._make_layer(block, 512, layers[3], stride=2)
		self._avgpool = nn.AdaptiveAvgPool2d((1, 1))
		self._fc1 = nn.Linear(512 * block.expansion, 5)
		
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
			elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)
	
		filters = [64, 128, 256, 512]
		resnet = models.resnet34(pretrained=True)
		self._firstconv = resnet.conv1
		self._firstbn = resnet.bn1
		self._firstrelu = resnet.relu
		self._firstmaxpool = resnet.maxpool
		self._encoder1 = resnet.layer1
		self._encoder2 = resnet.layer2
		self._encoder3 = resnet.layer3
		self._encoder4 = resnet.layer4
	
		self._dblock = DACblock(512)
		self._spp = SPPblock(512)
	
		self._decoder4 = DecoderBlock(516, filters[2])
		self._decoder3 = DecoderBlock(filters[2], filters[1])
		self._decoder2 = DecoderBlock(filters[1], filters[0])
		self._decoder1 = DecoderBlock(filters[0], filters[0])
	
		self._finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
		self._finalrelu1 = nonlinearity
		self._finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
		self._finalrelu2 = nonlinearity
		self._finalconv3 = nn.Conv2d(32, 3, 3, padding=1)
		self._finalconv4 = nn.Conv2d(3, 3, 3, padding=1)
		
		#####
		self.equal1 = nn.Conv2d(128, 64, 3, padding=1)
		self.equal2 = nn.Conv2d(320, 256, 3, padding=1)
		self.equal3 = nn.Conv2d(640, 512, 3, padding=1)
		self.equal4 = nn.Conv2d(1280, 1024, 3, padding=1)
	
	def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
					stride: int = 1) -> nn.Sequential:
		norm_layer = self._norm_layer
		downsample = None
		
		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(
				conv1x1(self.inplanes, planes * block.expansion, stride),
				norm_layer(planes * block.expansion),
			)
		
		layers = []
		layers.append(block(self.inplanes, planes, stride, downsample, norm_layer))
		
		self.inplanes = planes * block.expansion
		
		for _ in range(1, blocks):
			layers.append(block(self.inplanes, planes, norm_layer=norm_layer))
		
		return nn.Sequential(*layers)
	
	def forward(self, input_512):
		###
		x = self._firstconv(input_512)  # x 2 64 256 256
		x = self._firstbn(x)  # x 2 64 256 256
		x = self._firstrelu(x)  # x 2 64 256 256
		x = self._firstmaxpool(x)  # x 2 64 128 128
		
		e1 = self._encoder1(x)  # e1 2 64 128 128
		e2 = self._encoder2(e1)  # e2 2 128 64 64
		e3 = self._encoder3(e2)  # e3 2 256 32 32
		e4 = self._encoder4(e3)  # e4 2 512 16 16
		
		# Center
		e4 = self._dblock(e4)  # e4 2 512 16 16
		e4 = self._spp(e4)  # e4 2 512 16 16
		
		d4 = self._decoder4(e4) + e3  # d4 2 256 32 32
		d3 = self._decoder3(d4) + e2  # d3 2 128 64 64
		d2 = self._decoder2(d3) + e1  # d2 2 64 128 128
		d1 = self._decoder1(d2)  # d1 2 64 256 256
		
		out = self._finaldeconv1(d1)  # 2 32 512 512
		out = self._finalrelu1(out)  # 2 32 512 512
		out = self._finalconv2(out)  # 2 32 512 512
		out = self._finalrelu2(out)  # 2 32 512 512
		out = self._finalconv3(out)  # 2 3 512 512
		out = out + input_512
		out = self._finalconv4(out)
		out = torch.sigmoid(out)  # 2 3 512 512
		
		#####################
		x1 = self._conv1(input_512)  # x1 2 64 256 256
		x1 = self._bn1(x1)  # x1 2 64 256 256
		x1 = self._relu(x1)  # x1 2 64 256 256  # d1 2 64 256 256
		cat0 = torch.cat([x1, d1], 1)  # 2 128 256 256
		x1 = self.equal1(cat0)  # x1 2 64 256 256
		x1 = self._maxpool(x1)  # x1: 2 64 128 128
		
		x1 = self._layer1(x1)  # x1 2 256 128 128  # d2 2 64 128 128
		cat1 = torch.cat([x1, d2], 1)  # cat1 2 320 128 128
		x1 = self.equal2(cat1)
		
		x1 = self._layer2(x1)  # x1 2 512 64 64  # d3 2 128 64 64
		cat2 = torch.cat([x1, d3], 1)  # cat2 2 640 128 128
		x1 = self.equal3(cat2)
		
		x1 = self._layer3(x1)  # x1 2 1024 32 32  # d4 2 256 32 32
		cat3 = torch.cat([x1, d4], 1)  # cat3 2 1280 128 128
		x1 = self.equal4(cat3)
		
		x1 = self._layer4(x1)  # x1 2 2048 16 16
		x1 = self._avgpool(x1)  # x1 2 2048 1 1
		x1 = torch.flatten(x1, 1)  # x1 2 2048
		label_pre = self._fc1(x1)  # x1
		return label_pre, out
# 基于 Resnet 的疾病分级
class EyesImage_CLS_with_resnet(nn.Module):
	def __init__(self, norm_layer=nn.BatchNorm2d):
		super(EyesImage_CLS_with_resnet, self).__init__()
		if type(norm_layer) == functools.partial:
			norm_layer = nn.BatchNorm2d
		self._norm_layer = norm_layer
		self.inplanes = 64
		layers = [3, 4, 6, 3]
		block = Bottleneck
		
		self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
		self.bn1 = norm_layer(self.inplanes)
		self.relu = nn.ReLU(inplace=True)
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
		self.layer1 = self._make_layer(block, 64, layers[0])
		self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
		self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
		self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
		self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
		self.fc1 = nn.Linear(512 * block.expansion, 5)
		
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
			elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)
	
	def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
					stride: int = 1) -> nn.Sequential:
		norm_layer = self._norm_layer
		downsample = None
		
		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(
				conv1x1(self.inplanes, planes * block.expansion, stride),
				norm_layer(planes * block.expansion),
			)
		
		layers = []
		layers.append(block(self.inplanes, planes, stride, downsample, norm_layer))
		
		self.inplanes = planes * block.expansion
		
		for _ in range(1, blocks):
			layers.append(block(self.inplanes, planes, norm_layer=norm_layer))
		
		return nn.Sequential(*layers)
	
	def forward(self, input_512):
		x1 = self.conv1(input_512)  # x1 2 64 256 256
		x1 = self.bn1(x1)  # x1 2 64 256 256
		x1 = self.relu(x1)  # x1 2 64 256 256  # d1 2 64 256 256
		x1 = self.maxpool(x1)  # x1: 2 64 128 128
		x1 = self.layer1(x1)  # x1 2 256 128 128  # d2 2 64 128 128
		x1 = self.layer2(x1)  # x1 2 512 64 64  # d3 2 128 64 64
		x1 = self.layer3(x1)  # x1 2 1024 32 32  # d4 2 256 32 3
		x1 = self.layer4(x1)  # x1 2 2048 16 16
		x1 = self.avgpool(x1)  # x1 2 2048 1 1
		x1 = torch.flatten(x1, 1)  # x1 2 2048
		label_pre = self.fc1(x1)  # x1
		return label_pre
# 基于 MyNet 的疾病分级
class EyesImage_CLS_with_MyNet(nn.Module):
	def __init__(self, norm_layer=nn.BatchNorm2d):
		super(EyesImage_CLS_with_MyNet, self).__init__()
		
		if type(norm_layer) == functools.partial:
			use_bias = norm_layer.func == nn.BatchNorm2d
		else:
			use_bias = norm_layer == nn.BatchNorm2d
			
		self._G_conv1 = nn.Conv2d(6, 32, 5, padding=2, bias=use_bias)
		self._G_relu1 = nn.ReLU()
		self._G_conv1_1 = ResBlock(in_channels=32, in_kernel=5, in_pad=2, in_bias=use_bias)
		self._G_conv1_2 = ResBlock(in_channels=32, in_kernel=5, in_pad=2, in_bias=use_bias)
		self._G_conv1_3 = ResBlock(in_channels=32, in_kernel=5, in_pad=2, in_bias=use_bias)
		self._G_conv2_0 = nn.Conv2d(32, 64, 5, padding=2, stride=2, bias=use_bias)
		self._G_relu2_0 = nn.ReLU()
		#
		self._G_conv2 = nn.Conv2d(128, 64, 5, padding=2, bias=use_bias)
		self._G_relu2 = nn.ReLU()
		self._G_conv2_1 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
		self._G_conv2_2 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
		self._G_conv2_3 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
		self._G_conv3_0 = nn.Conv2d(64, 64, 5, padding=2, stride=2, bias=use_bias)
		self._G_relu3_0 = nn.ReLU()
		#
		self._G_conv3 = nn.Conv2d(128, 64, 5, padding=2, bias=use_bias)
		self._G_relu3 = nn.ReLU()
		self._G_conv3_1 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
		self._G_conv3_2 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
		self._G_conv3_3 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
		self._G_conv4_0 = nn.Conv2d(64, 128, 5, padding=2, stride=2, bias=use_bias)
		self._G_relu4_0 = nn.ReLU()
		#
		self._G_conv4 = nn.Conv2d(256, 128, 5, padding=2, bias=use_bias)
		self._G_relu4 = nn.ReLU()
		self._G_conv4_1 = ResBlock(in_channels=128, in_kernel=5, in_pad=2, in_bias=use_bias)
		self._G_conv4_2 = ResBlock(in_channels=128, in_kernel=5, in_pad=2, in_bias=use_bias)
		self._G_conv4_3 = ResBlock(in_channels=128, in_kernel=5, in_pad=2, in_bias=use_bias)
		self._G_conv5_0 = nn.Conv2d(128, 256, 5, padding=2, stride=2, bias=use_bias)
		self._G_relu5_0 = nn.ReLU()
		
		self._G_conv5 = nn.Conv2d(512, 256, 5, padding=2, bias=use_bias)
		self._G_relu5 = nn.ReLU()
		self._G_conv5_1 = ResBlock(in_channels=256, in_kernel=5, in_pad=2, in_bias=use_bias)
		self._G_conv5_2 = ResBlock(in_channels=256, in_kernel=5, in_pad=2, in_bias=use_bias)
		self._G_conv5_3 = ResBlock(in_channels=256, in_kernel=5, in_pad=2, in_bias=use_bias)
		
		self._G_conv6 = nn.Conv2d(256, 512, 5, padding=2, stride=2, bias=use_bias)
		self._G_relu6 = nn.ReLU()
		
		self._avgpool = nn.AdaptiveAvgPool2d((1, 1))
		self._fc = nn.Linear(512, 5)
		
	def forward(self, input_512):
		input_copy_512 = torch.cat([input_512, input_512], 1)  # input_copy_512 2 6 512 512
		x = self._G_conv1(input_copy_512)  # x 2 32 512 512
		x = self._G_relu1(x)  # x 2 32 512 512
		x = self._G_conv1_1(x)  # x 2 32 512 512
		x = self._G_conv1_2(x)  # x 2 32 512 512
		x_512 = self._G_conv1_3(x)  # x_512 2 32 512 512
		
		x = self._G_conv2_0(x_512)  # x 2 64 256 256
		x = self._G_relu2_0(x)  # x 2 64 256 256
		con_2 = torch.cat([x, x], 1)  # con_2 2 128 256 256
		x = self._G_conv2(con_2)  # x 2 64 256 256
		x = self._G_relu2(x)  # x 2 64 256 256
		x = self._G_conv2_1(x)  # x 2 64 256 256
		x = self._G_conv2_2(x)  # x 2 64 256 256
		x_256 = self._G_conv2_3(x)  # x_256 2 64 256 256
		
		x = self._G_conv3_0(x_256)  # x 2 64 128 128
		x = self._G_relu3_0(x)  # x 2 64 128 128
		con_4 = torch.cat([x, x], 1)  # con_4 2 128 128 128
		x = self._G_conv3(con_4)  # x 2 64 128 128
		x = self._G_relu3(x)  # x 2 64 128 128
		x = self._G_conv3_1(x)  # x 2 64 128 128
		x = self._G_conv3_2(x)  # x 2 64 128 128
		x_128 = self._G_conv3_3(x)  # x_128 2 64 128 128
		
		x = self._G_conv4_0(x_128)  # x 2 128 64 64
		x = self._G_relu4_0(x)  # x 2 128 64 64
		con_8 = torch.cat([x, x], 1)  # x 2 256 64 64
		x = self._G_conv4(con_8)  # x 2 128 64 64
		x = self._G_relu4(x)  # x 2 128 64 64
		x = self._G_conv4_1(x)  # x 2 128 64 64
		x = self._G_conv4_2(x)  # x 2 128 64 64
		x_64 = self._G_conv4_3(x)  # x 2 128 64 64 -> 2 524288
		
		x = self._G_conv5_0(x_64)  # 2 256 32 32
		x = self._G_relu5_0(x)  # 2 256 32 32
		con_16 = torch.cat([x, x], 1)
		x = self._G_conv5(con_16)
		x = self._G_relu5(x)
		x = self._G_conv5_1(x)
		x = self._G_conv5_2(x)
		x = self._G_conv5_3(x)
		
		x = self._G_conv6(x)  # 2 512 16 16
		x = self._G_relu6(x)  # 2 512 16 16
		
		x = self._avgpool(x)  # 2 512 1 1
		x = torch.flatten(x, 1)  # 2 512
		label_pre = self._fc(x)
		
		return label_pre
# 图像质量修正
class EyesImage_SR(nn.Module):
	def __init__(self, norm_layer=nn.BatchNorm2d):
		super(EyesImage_SR, self).__init__()
		if type(norm_layer) == functools.partial:
			use_bias = norm_layer.func == nn.InstanceNorm2d
		else:
			use_bias = norm_layer == nn.InstanceNorm2d
		# 这里是图像质量修正网络的SR分支
		self.G_conv1 = nn.Conv2d(6, 32, 5, padding=2, bias=use_bias)
		self.G_relu1 = nn.ReLU()
		self.G_conv1_1 = ResBlock(in_channels=32, in_kernel=5, in_pad=2, in_bias=use_bias)
		self.G_conv1_2 = ResBlock(in_channels=32, in_kernel=5, in_pad=2, in_bias=use_bias)
		self.G_conv1_3 = ResBlock(in_channels=32, in_kernel=5, in_pad=2, in_bias=use_bias)
		self.G_conv2_0 = nn.Conv2d(32, 64, 5, padding=2, stride=2, bias=use_bias)
		self.G_relu2_0 = nn.ReLU()
		#
		self.G_conv2 = nn.Conv2d(128, 64, 5, padding=2, bias=use_bias)
		self.G_relu2 = nn.ReLU()
		self.G_conv2_1 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
		self.G_conv2_2 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
		self.G_conv2_3 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
		self.G_conv3_0 = nn.Conv2d(64, 64, 5, padding=2, stride=2, bias=use_bias)
		self.G_relu3_0 = nn.ReLU()
		#
		self.G_conv3 = nn.Conv2d(128, 64, 5, padding=2, bias=use_bias)
		self.G_relu3 = nn.ReLU()
		self.G_conv3_1 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
		self.G_conv3_2 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
		self.G_conv3_3 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
		self.G_conv4_0 = nn.Conv2d(64, 128, 5, padding=2, stride=2, bias=use_bias)
		self.G_relu4_0 = nn.ReLU()
		#
		self.G_conv4 = nn.Conv2d(256, 128, 5, padding=2, bias=use_bias)
		self.G_relu4 = nn.ReLU()
		self.G_conv4_1 = ResBlock(in_channels=128, in_kernel=5, in_pad=2, in_bias=use_bias)
		self.G_conv4_2 = ResBlock(in_channels=128, in_kernel=5, in_pad=2, in_bias=use_bias)
		self.G_conv4_3 = ResBlock(in_channels=128, in_kernel=5, in_pad=2, in_bias=use_bias)
		
		self.G_deconv4_3 = ResBlock(in_channels=128, in_kernel=5, in_pad=2, in_bias=use_bias)
		self.G_deconv4_2 = ResBlock(in_channels=128, in_kernel=5, in_pad=2, in_bias=use_bias)
		self.G_deconv4_1 = ResBlock(in_channels=128, in_kernel=5, in_pad=2, in_bias=use_bias)
		self.G_deconv4_0 = nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=use_bias)
		
		self.G_deconv3_3 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
		self.G_deconv3_2 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
		self.G_deconv3_1 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
		self.G_deconv3_0 = nn.ConvTranspose2d(64, 64, 4, 2, 1, bias=use_bias)
		
		self.G_deconv2_3 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
		self.G_deconv2_2 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
		self.G_deconv2_1 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
		self.G_deconv2_0 = nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=use_bias)
		
		self.G_deconv1_3 = ResBlock(in_channels=32, in_kernel=5, in_pad=2, in_bias=use_bias)
		self.G_deconv1_2 = ResBlock(in_channels=32, in_kernel=5, in_pad=2, in_bias=use_bias)
		self.G_deconv1_1 = ResBlock(in_channels=32, in_kernel=5, in_pad=2, in_bias=use_bias)
		self.G_deconv1_0 = nn.Conv2d(32, 3, 5, padding=2, bias=use_bias)
		
		# 这里是质量修正网络的ves分割分支
		filters = [64, 128, 256, 512]
		resnet = models.resnet34(pretrained=True)
		self.firstconv = resnet.conv1
		self.firstbn = resnet.bn1
		self.firstrelu = resnet.relu
		self.firstmaxpool = resnet.maxpool
		self.encoder1 = resnet.layer1
		self.encoder2 = resnet.layer2
		self.encoder3 = resnet.layer3
		self.encoder4 = resnet.layer4
		
		self.dblock = DACblock(512)
		self.spp = SPPblock(512)
		
		self.decoder4 = DecoderBlock(516, filters[2])
		self.decoder3 = DecoderBlock(filters[2], filters[1])
		self.decoder2 = DecoderBlock(filters[1], filters[0])
		self.decoder1 = DecoderBlock(filters[0], filters[0])
		
		self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
		self.finalrelu1 = nonlinearity
		self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
		self.finalrelu2 = nonlinearity
		self.finalconv3 = nn.Conv2d(32, 1, 3, padding=1)
		
	def forward(self, input_512, input_norm):
		# 这里是图像质量修正网络的ves分割分支
		x = self.firstconv(input_norm)
		x = self.firstbn(x)
		x = self.firstrelu(x)
		x = self.firstmaxpool(x)
		
		e1 = self.encoder1(x)
		e2 = self.encoder2(e1)
		e3 = self.encoder3(e2)
		e4 = self.encoder4(e3)
		
		# Center
		e4 = self.dblock(e4)
		e4 = self.spp(e4)
		
		# Decoder
		d4 = self.decoder4(e4) + e3
		d3 = self.decoder3(d4) + e2
		d2 = self.decoder2(d3) + e1
		d1 = self.decoder1(d2)
		
		#
		out = self.finaldeconv1(d1)
		out = self.finalrelu1(out)
		out = self.finalconv2(out)
		out = self.finalrelu2(out)
		out = self.finalconv3(out)
		
		out = torch.sigmoid(out)
		
		# 这里是图像质量修正网络的ves分割分支
		input_copy_512 = torch.cat([input_512, input_512], 1)
		x = self.G_conv1(input_copy_512)
		x = self.G_relu1(x)
		x = self.G_conv1_1(x)
		x = self.G_conv1_2(x)
		x_512 = self.G_conv1_3(x)
		
		x = self.G_conv2_0(x_512)
		x = self.G_relu2_0(x)
		con_2 = torch.cat([x, d1], 1)
		x = self.G_conv2(con_2)
		x = self.G_relu2(x)
		x = self.G_conv2_1(x)
		x = self.G_conv2_2(x)
		x_256 = self.G_conv2_3(x)
		
		x = self.G_conv3_0(x_256)
		x = self.G_relu3_0(x)
		con_4 = torch.cat([x, d2], 1)
		x = self.G_conv3(con_4)
		x = self.G_relu3(x)
		x = self.G_conv3_1(x)
		x = self.G_conv3_2(x)
		x_128 = self.G_conv3_3(x)
		
		x = self.G_conv4_0(x_128)
		x = self.G_relu4_0(x)
		con_8 = torch.cat([x, d3], 1)
		x = self.G_conv4(con_8)
		x = self.G_relu4(x)
		x = self.G_conv4_1(x)
		x = self.G_conv4_2(x)
		x = self.G_conv4_3(x)
		
		x = self.G_deconv4_3(x)
		x = self.G_deconv4_2(x)
		x = self.G_deconv4_1(x)
		x = self.G_deconv4_0(x)
		
		x = x + x_128
		
		x = self.G_deconv3_3(x)
		x = self.G_deconv3_2(x)
		x = self.G_deconv3_1(x)
		x = self.G_deconv3_0(x)
		
		x = x + x_256
		
		x = self.G_deconv2_3(x)
		x = self.G_deconv2_2(x)
		x = self.G_deconv2_1(x)
		x = self.G_deconv2_0(x)
		
		x = x + x_512
		
		x = self.G_deconv1_3(x)
		x = self.G_deconv1_2(x)
		x = self.G_deconv1_1(x)
		x = self.G_deconv1_0(x)
		output_512 = torch.sigmoid(x)
		
		return output_512, out
# 联合网络
class EyesImage_joint(nn.Module):
	def __init__(self, norm_layer=nn.BatchNorm2d):
		super(EyesImage_joint, self).__init__()
		if type(norm_layer) == functools.partial:
			use_bias = norm_layer.func == nn.InstanceNorm2d
		else:
			use_bias = norm_layer == nn.InstanceNorm2d
		self._norm_layer = nn.BatchNorm2d
		self.inplanes = 64
		self.inplanes2 = 64
		layers = [3, 4, 6, 3]
		block = Bottleneck
		
		# 这里是分类网络分支
		self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
		self.bn1 = norm_layer(self.inplanes)
		self.relu = nn.ReLU(inplace=True)
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
		self.layer1 = self._make_layer(block, 64, layers[0])
		self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
		self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
		self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
		self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
		self.fc1 = nn.Linear(512 * block.expansion, 5)
		
		# 这里是IQA网络的质量评估分支
		self._conv1 = nn.Conv2d(3, self.inplanes2, kernel_size=7, stride=2, padding=3, bias=False)
		self._bn1 = norm_layer(self.inplanes2)
		self._relu = nn.ReLU(inplace=True)
		self._maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
		self._layer1 = self._make_layer2(block, 64, layers[0])
		self._layer2 = self._make_layer2(block, 128, layers[1], stride=2)
		self._layer3 = self._make_layer2(block, 256, layers[2], stride=2)
		self._layer4 = self._make_layer2(block, 512, layers[3], stride=2)
		self._avgpool = nn.AdaptiveAvgPool2d((1, 1))
		self._fc1 = nn.Linear(512 * block.expansion, 5)
		
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
			elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)
				
		# 这里是图像质量修正网络的SR分支
		self.G_conv1 = nn.Conv2d(6, 32, 5, padding=2, bias=use_bias)
		self.G_relu1 = nn.ReLU()
		self.G_conv1_1 = ResBlock(in_channels=32, in_kernel=5, in_pad=2, in_bias=use_bias)
		self.G_conv1_2 = ResBlock(in_channels=32, in_kernel=5, in_pad=2, in_bias=use_bias)
		self.G_conv1_3 = ResBlock(in_channels=32, in_kernel=5, in_pad=2, in_bias=use_bias)
		self.G_conv2_0 = nn.Conv2d(32, 64, 5, padding=2, stride=2, bias=use_bias)
		self.G_relu2_0 = nn.ReLU()
		# concat 1/2
		self.G_conv2 = nn.Conv2d(128, 64, 5, padding=2, bias=use_bias)
		self.G_relu2 = nn.ReLU()
		self.G_conv2_1 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
		self.G_conv2_2 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
		self.G_conv2_3 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
		self.G_conv3_0 = nn.Conv2d(64, 64, 5, padding=2, stride=2, bias=use_bias)
		self.G_relu3_0 = nn.ReLU()
		# concat 1/4
		self.G_conv3 = nn.Conv2d(128, 64, 5, padding=2, bias=use_bias)
		self.G_relu3 = nn.ReLU()
		self.G_conv3_1 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
		self.G_conv3_2 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
		self.G_conv3_3 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
		self.G_conv4_0 = nn.Conv2d(64, 128, 5, padding=2, stride=2, bias=use_bias)
		self.G_relu4_0 = nn.ReLU()
		# concat 1/8
		self.G_conv4 = nn.Conv2d(256, 128, 5, padding=2, bias=use_bias)
		self.G_relu4 = nn.ReLU()
		self.G_conv4_1 = ResBlock(in_channels=128, in_kernel=5, in_pad=2, in_bias=use_bias)
		self.G_conv4_2 = ResBlock(in_channels=128, in_kernel=5, in_pad=2, in_bias=use_bias)
		self.G_conv4_3 = ResBlock(in_channels=128, in_kernel=5, in_pad=2, in_bias=use_bias)
		
		self.G_deconv4_3 = ResBlock(in_channels=128, in_kernel=5, in_pad=2, in_bias=use_bias)
		self.G_deconv4_2 = ResBlock(in_channels=128, in_kernel=5, in_pad=2, in_bias=use_bias)
		self.G_deconv4_1 = ResBlock(in_channels=128, in_kernel=5, in_pad=2, in_bias=use_bias)
		self.G_deconv4_0 = nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=use_bias)
		
		self.G_deconv3_3 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
		self.G_deconv3_2 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
		self.G_deconv3_1 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
		self.G_deconv3_0 = nn.ConvTranspose2d(64, 64, 4, 2, 1, bias=use_bias)
		
		self.G_deconv2_3 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
		self.G_deconv2_2 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
		self.G_deconv2_1 = ResBlock(in_channels=64, in_kernel=5, in_pad=2, in_bias=use_bias)
		self.G_deconv2_0 = nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=use_bias)
		
		self.G_deconv1_3 = ResBlock(in_channels=32, in_kernel=5, in_pad=2, in_bias=use_bias)
		self.G_deconv1_2 = ResBlock(in_channels=32, in_kernel=5, in_pad=2, in_bias=use_bias)
		self.G_deconv1_1 = ResBlock(in_channels=32, in_kernel=5, in_pad=2, in_bias=use_bias)
		self.G_deconv1_0 = nn.Conv2d(32, 3, 5, padding=2, bias=use_bias)
		
		
		# 这里是图像修正网络的CE-net血管分割分支
		filters = [64, 128, 256, 512]
		resnet = models.resnet34(pretrained=True)
		self.firstconv = resnet.conv1
		self.firstbn = resnet.bn1
		self.firstrelu = resnet.relu
		self.firstmaxpool = resnet.maxpool
		self.encoder1 = resnet.layer1
		self.encoder2 = resnet.layer2
		self.encoder3 = resnet.layer3
		self.encoder4 = resnet.layer4
		
		self.dblock = DACblock(512)
		self.spp = SPPblock(512)
		
		self.decoder4 = DecoderBlock(516, filters[2])
		self.decoder3 = DecoderBlock(filters[2], filters[1])
		self.decoder2 = DecoderBlock(filters[1], filters[0])
		self.decoder1 = DecoderBlock(filters[0], filters[0])
		
		self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
		self.finalrelu1 = nonlinearity
		self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
		self.finalrelu2 = nonlinearity
		self.finalconv3 = nn.Conv2d(32, 1, 3, padding=1)
		
		# 这里是IQA网络的模糊重建分支
		self._firstconv = resnet.conv1
		self._firstbn = resnet.bn1
		self._firstrelu = resnet.relu
		self._firstmaxpool = resnet.maxpool
		self._encoder1 = resnet.layer1
		self._encoder2 = resnet.layer2
		self._encoder3 = resnet.layer3
		self._encoder4 = resnet.layer4
		
		self._dblock = DACblock(512)
		self._spp = SPPblock(512)
		
		self._decoder4 = DecoderBlock(516, filters[2])
		self._decoder3 = DecoderBlock(filters[2], filters[1])
		self._decoder2 = DecoderBlock(filters[1], filters[0])
		self._decoder1 = DecoderBlock(filters[0], filters[0])
		
		self._finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
		self._finalrelu1 = nonlinearity
		self._finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
		self._finalrelu2 = nonlinearity
		self._finalconv3 = nn.Conv2d(32, 3, 3, padding=1)
		self._finalconv4 = nn.Conv2d(3, 3, 3, padding=1)
		#####
		self.equal1 = nn.Conv2d(128, 64, 3, padding=1)
		self.equal2 = nn.Conv2d(320, 256, 3, padding=1)
		self.equal3 = nn.Conv2d(640, 512, 3, padding=1)
		self.equal4 = nn.Conv2d(1280, 1024, 3, padding=1)
		
	def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
					stride: int = 1) -> nn.Sequential:
		norm_layer = self._norm_layer
		downsample = None
		
		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(
				conv1x1(self.inplanes, planes * block.expansion, stride),
				norm_layer(planes * block.expansion),
			)
		
		layers = []
		layers.append(block(self.inplanes, planes, stride, downsample, norm_layer))
		
		self.inplanes = planes * block.expansion
		
		for _ in range(1, blocks):
			layers.append(block(self.inplanes, planes, norm_layer=norm_layer))
		
		return nn.Sequential(*layers)
	
	def _make_layer2(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
					stride: int = 1) -> nn.Sequential:
		norm_layer = self._norm_layer
		downsample = None
		
		if stride != 1 or self.inplanes2 != planes * block.expansion:
			downsample = nn.Sequential(
				conv1x1(self.inplanes2, planes * block.expansion, stride),
				norm_layer(planes * block.expansion),
			)
		
		layers = []
		layers.append(block(self.inplanes2, planes, stride, downsample, norm_layer))
		
		self.inplanes2 = planes * block.expansion
		
		for _ in range(1, blocks):
			layers.append(block(self.inplanes2, planes, norm_layer=norm_layer))
		
		return nn.Sequential(*layers)
	
	def forward(self, input_512, input_norm):  # 1 3 512 512
		# 这里是图像修正网络的ves分割分支
		x = self.firstconv(input_norm) # 1 64 256 256
		x = self.firstbn(x)
		x = self.firstrelu(x)
		x = self.firstmaxpool(x) # 1 64 128 128
		
		e1 = self.encoder1(x) # 1 64 128 128
		e2 = self.encoder2(e1) # 1 128 64 64
		e3 = self.encoder3(e2) # 1 256 32 32
		e4 = self.encoder4(e3) # 1 512 16 16
		
		# Center
		e4 = self.dblock(e4) # 1 512 16 16
		e4 = self.spp(e4) # 1 516 16 16
		
		# Decoder
		d4 = self.decoder4(e4) + e3 # 1 256 32 32
		d3 = self.decoder3(d4) + e2 # 1 128 64 64
		d2 = self.decoder2(d3) + e1 # 1 64 128 128
		d1 = self.decoder1(d2) # 1 64 256 256
		
		out = self.finaldeconv1(d1) # 1 32 512 512
		out = self.finalrelu1(out) # 1 32 512 512
		out = self.finalconv2(out) # 1 32 512 512
		out = self.finalrelu2(out) # 1 32 512 512
		out = self.finalconv3(out) # 1 1 512 512
		
		out_ves = torch.sigmoid(out) # 1 1 512 512
		
		# 这里是图像修正网络的SR分支
		input_copy_512 = torch.cat([input_512, input_512], 1) # 1 6 512 512
		x1 = self.G_conv1(input_copy_512) # 1 32 512 512
		x1 = self.G_relu1(x1)
		x1 = self.G_conv1_1(x1)
		x1 = self.G_conv1_2(x1)
		x_512 = self.G_conv1_3(x1)
		
		x1 = self.G_conv2_0(x_512)
		x1 = self.G_relu2_0(x1)
		con_2 = torch.cat([x1, d1], 1)
		x1 = self.G_conv2(con_2)
		x1 = self.G_relu2(x1)
		x1 = self.G_conv2_1(x1)
		x1 = self.G_conv2_2(x1)
		x_256 = self.G_conv2_3(x1)
		
		x1 = self.G_conv3_0(x_256)
		x1 = self.G_relu3_0(x1)
		con_4 = torch.cat([x1, d2], 1)
		x1 = self.G_conv3(con_4)
		x1 = self.G_relu3(x1)
		x1 = self.G_conv3_1(x1)
		x1 = self.G_conv3_2(x1)
		x_128 = self.G_conv3_3(x1)
		
		x1 = self.G_conv4_0(x_128)
		x1 = self.G_relu4_0(x1)
		con_8 = torch.cat([x1, d3], 1)
		x1 = self.G_conv4(con_8)
		x1 = self.G_relu4(x1)
		x1 = self.G_conv4_1(x1)
		x1 = self.G_conv4_2(x1)
		x1 = self.G_conv4_3(x1)
		
		x1 = self.G_deconv4_3(x1)
		x1 = self.G_deconv4_2(x1)
		x1 = self.G_deconv4_1(x1)
		x1 = self.G_deconv4_0(x1)
		
		x1 = x1 + x_128
		
		x1 = self.G_deconv3_3(x1)
		x1 = self.G_deconv3_2(x1)
		x1 = self.G_deconv3_1(x1)
		x1 = self.G_deconv3_0(x1)
		
		x1 = x1 + x_256
		
		x1 = self.G_deconv2_3(x1)
		x1 = self.G_deconv2_2(x1)
		x1 = self.G_deconv2_1(x1)
		x1 = self.G_deconv2_0(x1)
		
		x1 = x1 + x_512
		
		x1 = self.G_deconv1_3(x1)
		x1 = self.G_deconv1_2(x1)
		x1 = self.G_deconv1_1(x1)
		x1 = self.G_deconv1_0(x1)
		output_512 = torch.sigmoid(x1)
		
		# 这里是疾病分类分支
		
		x2 = self.conv1(output_512)  # x1 2 64 256 256
		x2 = self.bn1(x2)  # x1 2 64 256 256
		x2 = self.relu(x2)  # x1 2 64 256 256  # d1 2 64 256 256
		x2 = self.maxpool(x2)  # x1: 2 64 128 128
		x2 = self.layer1(x2)  # x1 2 256 128 128  # d2 2 64 128 128
		x2 = self.layer2(x2)  # x1 2 512 64 64  # d3 2 128 64 64
		x2 = self.layer3(x2)  # x1 2 1024 32 32  # d4 2 256 32 3
		x2 = self.layer4(x2)  # x1 2 2048 16 16
		x2 = self.avgpool(x2)  # x1 2 2048 1 1
		x2 = torch.flatten(x2, 1)  # x1 2 2048
		label_pre_cls = self.fc1(x2)  # x1
		
		# 这里是IQA的模糊重建分支
		x3 = self._firstconv(output_512)  # x 2 64 256 256
		x3 = self._firstbn(x3)  # x 2 64 256 256
		x3 = self._firstrelu(x3)  # x 2 64 256 256
		x3 = self._firstmaxpool(x3)  # x 2 64 128 128
		
		e11 = self._encoder1(x3)  # e1 2 64 128 128
		e22 = self._encoder2(e11)  # e2 2 128 64 64
		e33 = self._encoder3(e22)  # e3 2 256 32 32
		e44 = self._encoder4(e33)  # e4 2 512 16 16
		
		# Center
		e44 = self._dblock(e44)  # e4 2 512 16 16
		e44 = self._spp(e44)  # e4 2 516 16 16
		
		d44 = self._decoder4(e44) + e33  # d4 2 256 32 32
		d33 = self._decoder3(d44) + e22  # d3 2 128 64 64
		d22 = self._decoder2(d33) + e11  # d2 2 64 128 128
		d11 = self._decoder1(d22)  # d1 2 64 256 256
		
		out1 = self._finaldeconv1(d11)  # 2 32 512 512
		out1 = self._finalrelu1(out1)  # 2 32 512 512
		out1 = self._finalconv2(out1)  # 2 32 512 512
		out1 = self._finalrelu2(out1)  # 2 32 512 512
		out1 = self._finalconv3(out1)  # 2 3 512 512
		out1 = out1 + output_512
		out1 = self._finalconv4(out1)
		
		out_blur = torch.sigmoid(out1)  # 2 3 512 512
		
		# 这里是IQA的质量分类分支
		x4 = self._conv1(output_512)  # x1 2 64 256 256
		x4 = self._bn1(x4)  # x1 2 64 256 256
		x4 = self._relu(x4)  # x1 2 64 256 256  # d1 2 64 256 256
		cat0 = torch.cat([x4, d11], 1)  # 2 128 256 256
		x4 = self.equal1(cat0)  # x1 2 64 256 256
		x4 = self._maxpool(x4)  # x1: 2 64 128 128
		
		x4 = self._layer1(x4)  # x1 2 256 128 128  # d2 2 64 128 128
		cat1 = torch.cat([x4, d22], 1)  # cat1 2 320 128 128
		x4 = self.equal2(cat1)
		
		x4 = self._layer2(x4)  # x1 2 512 64 64  # d3 2 128 64 64
		cat2 = torch.cat([x4, d33], 1)  # cat2 2 640 128 128
		x4 = self.equal3(cat2)
		
		x4 = self._layer3(x4)  # x1 2 1024 32 32  # d4 2 256 32 32
		cat3 = torch.cat([x4, d44], 1)  # cat3 2 1280 128 128
		x4 = self.equal4(cat3)
		
		x4 = self._layer4(x4)  # x1 2 2048 16 16
		x4 = self._avgpool(x4)  # x1 2 2048 1 1
		x4 = torch.flatten(x4, 1)  # x1 2 2048
		
		label_pre_IQA = self._fc1(x4)  # x1
		
		return out_ves, output_512, label_pre_cls, out_blur, label_pre_IQA
		
class DACblock(nn.Module):
	def __init__(self, channel):
		super(DACblock, self).__init__()
		self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
		self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=3, padding=3)
		self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=5, padding=5)
		self.conv1x1 = nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0)
		for m in self.modules():
			if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
				if m.bias is not None:
					m.bias.data.zero_()

	def forward(self, x):
		dilate1_out = nonlinearity(self.dilate1(x))
		dilate2_out = nonlinearity(self.conv1x1(self.dilate2(x)))
		dilate3_out = nonlinearity(self.conv1x1(self.dilate2(self.dilate1(x))))
		dilate4_out = nonlinearity(self.conv1x1(self.dilate3(self.dilate2(self.dilate1(x)))))
		out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out
		return out

class SPPblock(nn.Module):
	def __init__(self, in_channels):
		super(SPPblock, self).__init__()
		self.pool1 = nn.MaxPool2d(kernel_size=[2, 2], stride=2)
		self.pool2 = nn.MaxPool2d(kernel_size=[3, 3], stride=3)
		self.pool3 = nn.MaxPool2d(kernel_size=[5, 5], stride=5)
		self.pool4 = nn.MaxPool2d(kernel_size=[6, 6], stride=6)

		self.conv = nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=1, padding=0)

	def forward(self, x):
		self.in_channels, h, w = x.size(1), x.size(2), x.size(3)
		self.layer1 = F.interpolate(self.conv(self.pool1(x)), size=(h, w), mode='bilinear', align_corners=True)
		self.layer2 = F.interpolate(self.conv(self.pool2(x)), size=(h, w), mode='bilinear', align_corners=True)
		self.layer3 = F.interpolate(self.conv(self.pool3(x)), size=(h, w), mode='bilinear', align_corners=True)
		self.layer4 = F.interpolate(self.conv(self.pool4(x)), size=(h, w), mode='bilinear', align_corners=True)

		out = torch.cat([self.layer1, self.layer2, self.layer3, self.layer4, x], 1)

		return out

class DecoderBlock(nn.Module):
	def __init__(self, in_channels, n_filters):
		super(DecoderBlock, self).__init__()

		self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
		self.norm1 = nn.BatchNorm2d(in_channels // 4)
		self.relu1 = nonlinearity

		self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
		self.norm2 = nn.BatchNorm2d(in_channels // 4)
		self.relu2 = nonlinearity

		self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
		self.norm3 = nn.BatchNorm2d(n_filters)
		self.relu3 = nonlinearity

	def forward(self, x):
		x = self.conv1(x)
		x = self.norm1(x)
		x = self.relu1(x)
		x = self.deconv2(x)
		x = self.norm2(x)
		x = self.relu2(x)
		x = self.conv3(x)
		x = self.norm3(x)
		x = self.relu3(x)
		return x

class ResBlock(nn.Module):
	def __init__(self, in_channels, in_kernel, in_pad, in_bias):
		super(ResBlock, self).__init__()

		self.conv1 = nn.Conv2d(in_channels, in_channels, in_kernel, padding=in_pad, bias=in_bias)
		self.relu1 = nonlinearity
		self.conv2 = nn.Conv2d(in_channels, in_channels, in_kernel, padding=in_pad, bias=in_bias)
		self.relu2 = nonlinearity

	def forward(self, x):
		x0 = self.conv1(x)
		x = self.relu2(x0)
		x = self.conv2(x)
		x = x0 + x
		out = self.relu2(x)
		return out
