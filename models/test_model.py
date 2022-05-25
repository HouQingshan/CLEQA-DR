from .base_model import BaseModel
from . import networks
from torch.autograd import Variable
from collections import OrderedDict
from os.path import join, exists

import util.util as util
from util.metrics import *


class TestModel(BaseModel):
    def name(self):
        return 'TestModel'

    def initialize(self, opt):
        assert (not opt.isTrain)
        BaseModel.initialize(self, opt)

        if opt.model_choice == 'EyesImage_IQA_with_masklabel' or opt.model_choice == 'EyesImage_IQAresnet_with_masklabel' \
                or opt.model_choice == 'EyesImage_IQA_with_image_blur_label' or opt.model_choice == 'EyesImage_IQAresnet_with_image_blur_label':

            self.input_gt_img = self.Tensor(opt.batchSize, opt.output_nc, opt.fineSize, opt.fineSize)
            self.input_norm_img = self.Tensor(opt.batchSize, opt.input_nc, opt.fineSize, opt.fineSize)
            self.input_gt_blur = self.Tensor(opt.batchSize, 3, opt.fineSize, opt.fineSize)
            self.input_gt_spot = self.Tensor(opt.batchSize, 1, opt.fineSize, opt.fineSize)
            self.input_region = self.Tensor(opt.batchSize, 1, opt.fineSize, opt.fineSize)
            self.label = self.Tensor(opt.batchSize)

            if opt.model_choice == 'EyesImage_IQA_with_masklabel' or opt.model_choice == 'EyesImage_IQAresnet_with_masklabel':
                if exists(join(opt.project_root, opt.segm_spot_weight_root)):
                    opt.segm_spot_weight_root = opt.segm_spot_weight_root
                else:
                    opt.segm_spot_weight_root = 'none'
                self.net = networks.define_net(opt.model_choice, opt.segm_spot_weight_root, opt.norm, self.gpu_ids)
            elif opt.model_choice == 'EyesImage_IQA_with_image_blur_label' or opt.model_choice == 'EyesImage_IQAresnet_with_image_blur_label':
                if exists((join(opt.project_root, opt.SR_blur_weight_root))):
                    opt.SR_blur_weight_root = opt.SR_blur_weight_root
                else:
                    opt.SR_blur_weight_root = 'none'
                self.net = networks.define_net(opt.model_choice, opt.SR_blur_weight_root, opt.norm, self.gpu_ids)

            if opt.model_choice == 'EyesImage_IQA_with_masklabel':
                self.load_network(self.net, 'IQA_mask', opt.which_epoch)
            elif opt.model_choice == 'EyesImage_IQAresnet_with_masklabel':
                self.load_network(self.net, 'IQAres_mask', opt.which_epoch)
            elif opt.model_choice == 'EyesImage_IQA_with_image_blur_label':
                self.load_network(self.net, 'IQA_blur', opt.which_epoch)
            elif opt.model_choice == 'EyesImage_IQAresnet_with_image_blur_label':
                self.load_network(self.net, 'IQAres_blur', opt.which_epoch)

        elif opt.model_choice == 'EyesImage_CLS_with_resnet' or opt.model_choice == 'EyesImage_CLS_with_MyNet':
            self.input_img = self.Tensor(opt.batchSize, opt.output_nc, opt.fineSize, opt.fineSize)
            self.input_img_mask = self.Tensor(opt.batchSize, 1, opt.fineSize, opt.fineSize)
            self.label = self.Tensor(opt.batchSize)

            if exists(join(opt.project_root, opt.CLS_weight_root)):
                opt.CLS_weight_root = opt.CLS_weight_root
            else:
                opt.CLS_weight_root = 'none'
            self.net = networks.define_net(opt.model_choice, opt.CLS_weight_root, opt.norm, self.gpu_ids)

            if opt.model_choice == 'EyesImage_CLS_with_resnet':
                self.load_network(self.net, 'CLS_res', opt.which_epoch)
            elif opt.model_choice == 'EyesImage_CLS_with_MyNet':
                self.load_network(self.net, 'CLS_MyNet', opt.which_epoch)


        elif opt.model_choice == 'EyesImage_SR':
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

            self.load_network(self.net, 'SR', opt.which_epoch)


        elif opt.model_choice == 'EyesImage_joint':
            self.input_gt_img = self.Tensor(opt.batchSize, opt.output_nc, opt.fineSize, opt.fineSize)
            self.input_de_img = self.Tensor(opt.batchSize, opt.output_nc, opt.fineSize, opt.fineSize)
            self.input_norm_img = self.Tensor(opt.batchSize, opt.output_nc, opt.fineSize, opt.fineSize)
            self.img_region_mask = self.Tensor(opt.batchSize, 1, opt.fineSize, opt.fineSize)
            self.img_ves_mask = self.Tensor(opt.batchSize, 1, opt.fineSize, opt.fineSize)
            self.label_cls = self.Tensor(opt.batchSize)
            self.label_IQA = self.Tensor(opt.batchSize)

            if exists(join(opt.project_root, opt.segm_vessel_weight_root)):
                opt.segm_vessel_weight_root = 'null'
            self.net = networks.define_net(opt.model_choice, opt.segm_vessel_weight_root, opt.norm, self.gpu_ids)

            self.load_network(self.net, 'joint', opt.which_epoch)

        else:
            print('no model was named %s' % opt.model_choice)

        print('---------- Networks initialized -------------')
        networks.print_network(self.net)
        print('-----------------------------------------------')

    def set_input(self, input):
        if self.opt.model_choice == 'EyesImage_IQA_with_masklabel' \
                or self.opt.model_choice == 'EyesImage_IQAresnet_with_masklabel' \
                or self.opt.model_choice == 'EyesImage_IQA_with_image_blur_label' \
                or self.opt.model_choice == 'EyesImage_IQAresnet_with_image_blur_label':

            input_gt_img = input['gt_img']
            input_gt_blur = input['gt_blur']
            input_norm_img = input['norm_img']
            input_gt_spot = input['gt_spot']
            input_region = input['region']
            label = input['labels_cls']
            self.image_paths = input['gt_im_paths']

            self.label = self.Tensor(label.shape)
            self.label = self.label.clone().copy_(label)
            self.input_gt_img = self.input_gt_img.clone().resize_(input_gt_img.size()).copy_(input_gt_img)
            self.input_gt_blur = self.input_gt_blur.clone().resize_(input_gt_blur.size()).copy_(input_gt_blur)
            self.input_norm_img = self.input_norm_img.clone().resize_(input_norm_img.size()).copy_(input_norm_img)
            self.input_gt_spot = self.input_gt_spot.clone().resize_(input_gt_spot.size()).copy_(input_gt_spot)
            self.input_region = self.input_region.clone().resize_(input_region.size()).copy_(input_region)

        elif self.opt.model_choice == 'EyesImage_CLS_with_resnet' \
                or self.opt.model_choice == 'EyesImage_CLS_with_MyNet':

            input_gt_img = input['gt_img']
            input_region = input['region']
            label = input['labels_cls']
            self.image_paths = input['gt_im_paths']

            self.label = self.Tensor(label.shape)
            self.label = self.label.clone().copy_(label)
            self.input_img = self.input_img.clone().resize_(input_gt_img.size()).copy_(input_gt_img)
            self.input_img_mask = self.input_img_mask.clone().resize_(input_region.size()).copy_(input_region)

        elif self.opt.model_choice == 'EyesImage_SR':
            input_de_img = input['de_img']
            input_gt_img = input['gt_img']
            input_norm_image = input['norm_image']
            input_mask_region = input['mask_region']
            input_ves_mask_region = input['ves_mask_region']
            self.image_paths = input['gt_img_path']

            self.input_de_img = self.input_de_img.clone().resize_(input_de_img.size()).copy_(input_de_img)
            self.input_gt_img = self.input_gt_img.clone().resize_(input_gt_img.size()).copy_(input_gt_img)
            self.input_img_norm = self.input_img_norm.clone().resize_(input_norm_image.size()).copy_(input_norm_image)
            self.input_img_mask = self.input_img_mask.clone().resize_(input_mask_region.size()).copy_(input_mask_region)
            self.input_img_ves_mask = self.input_img_ves_mask.clone().resize_(input_ves_mask_region.size()).copy_(
                input_ves_mask_region)

        elif self.opt.model_choice == 'EyesImage_joint':
            input_de_img = input['de_img']
            input_gt_img = input['gt_img']
            input_norm_image = input['norm_image']
            input_mask_region = input['mask_region']
            input_ves_mask_region = input['ves_mask_region']
            label_IQA = input['labels_IQA']
            label_cls = input['labels_cls']
            self.image_paths = input['gt_img_path']

            self.label_IQA = self.Tensor(label_IQA.shape)
            self.label_cls = self.Tensor(label_cls.shape)
            self.label_IQA = self.label_IQA.clone().copy_(label_IQA)
            self.label_cls = self.label_cls.clone().copy_(label_cls)
            self.input_de_img = self.input_de_img.clone().resize_(input_de_img.size()).copy_(input_de_img)
            self.input_gt_img = self.input_gt_img.clone().resize_(input_gt_img.size()).copy_(input_gt_img)
            self.input_norm_img = self.input_norm_img.clone().resize_(input_norm_image.size()).copy_(input_norm_image)
            self.img_region_mask = self.img_region_mask.clone().resize_(input_mask_region.size()).copy_(
                input_mask_region)
            self.img_ves_mask = self.img_ves_mask.clone().resize_(input_ves_mask_region.size()).copy_(
                input_ves_mask_region)

    def test(self):
        if self.opt.model_choice == 'EyesImage_IQA_with_masklabel' \
                or self.opt.model_choice == 'EyesImage_IQAresnet_with_masklabel' \
                or self.opt.model_choice == 'EyesImage_IQA_with_image_blur_label' \
                or self.opt.model_choice == 'EyesImage_IQAresnet_with_image_blur_label':

            self.input_norm_img = Variable(self.input_norm_img, requires_grad=False)
            self.in_gt_img = Variable(self.input_gt_img, requires_grad=False)
            self.in_gt_blur = Variable(self.input_gt_blur, requires_grad=False)
            self.input_gt_spot = Variable(self.input_gt_spot, requires_grad=False)
            self.input_region = Variable(self.input_region, requires_grad=False)
            self.in_labels = Variable(self.label, requires_grad=False)
            self.in_labels2 = self.in_labels.long()
            if self.opt.model_choice == 'EyesImage_IQA_with_masklabel' or self.opt.model_choice == 'EyesImage_IQAresnet_with_masklabel':
                self.labels_pre, self.spot_map = self.net.forward(self.in_gt_img, self.input_norm_img)
                out = self.labels_pre
                label = self.in_labels2
                return out, label
            elif self.opt.model_choice == 'EyesImage_IQA_with_image_blur_label' or self.opt.model_choice == 'EyesImage_IQAresnet_with_image_blur_label':
                self.labels_pre, self.imgblur = self.net.forward(self.in_gt_img)
                out = self.labels_pre
                label = self.in_labels2
                return out, label

        elif self.opt.model_choice == 'EyesImage_CLS_with_resnet' \
                or self.opt.model_choice == 'EyesImage_CLS_with_MyNet':
            self.in_gt_img = Variable(self.input_img, requires_grad=False)
            self.input_region = Variable(self.input_img_mask, requires_grad=False)
            self.in_labels = Variable(self.label, requires_grad=False)
            self.in_labels2 = self.in_labels.long()
            self.label_pre = self.net.forward(self.in_gt_img)
            out = self.label_pre
            label = self.in_labels2
            return out, label

        elif self.opt.model_choice == 'EyesImage_SR':
            self.in_de_img = Variable(self.input_de_img, requires_grad=False)
            self.in_gt_img = Variable(self.input_gt_img, requires_grad=False)
            self.in_img_norm = Variable(self.input_img_norm, requires_grad=False)
            self.in_img_mask = Variable(self.input_img_mask, requires_grad=False)
            self.in_img_ves_mask = Variable(self.input_img_ves_mask, requires_grad=False)
            self.SR_image, self.ves_mask_pre = self.net.forward(self.in_de_img, self.in_img_norm)

        elif self.opt.model_choice == 'EyesImage_joint':
            self.in_de_img = Variable(self.input_de_img, requires_grad=False)
            self.in_gt_img = Variable(self.input_gt_img, requires_grad=False)
            self.in_img_norm = Variable(self.input_norm_img, requires_grad=False)
            self.in_img_mask = Variable(self.img_region_mask, requires_grad=False)
            self.in_img_ves_mask = Variable(self.img_ves_mask, requires_grad=False)
            self.in_labels_IQA = Variable(self.label_IQA, requires_grad=False)
            self.in_labels_IQA2 = self.in_labels_IQA.long()
            self.in_labels_cls = Variable(self.label_cls, requires_grad=False)
            self.in_labels_cls2 = self.in_labels_cls.long()
            # out_ves, output_512, label_pre_cls, out_blur, label_pre_IQA
            self.ves_mask_pre, self.SR_image, self.labels_cls_pre, self.blur_image, self.labels_IQA_pre = self.net.forward(
                self.in_de_img, self.in_img_norm)
            IQA_out = self.labels_IQA_pre
            IQA_label = self.in_labels_IQA2
            cls_out = self.labels_cls_pre
            cls_label = self.in_labels_cls2
            return IQA_out, IQA_label, cls_out, cls_label

    def get_image_paths(self):
        return self.image_paths

    def get_current_visuals(self):
        if self.opt.model_choice == 'EyesImage_IQA_with_masklabel' or self.opt.model_choice == 'EyesImage_IQAresnet_with_masklabel':
            in_gt_img = util.tensor2im_2(self.in_gt_img.data,len(self.image_paths))
            spot_map = util.tensor2im_2(self.spot_map.data,len(self.image_paths))
            # Acc
            # return Acc

            return OrderedDict([('in_gt_img', in_gt_img), ('spot_map', spot_map)])

        elif self.opt.model_choice == 'EyesImage_IQA_with_image_blur_label' or self.opt.model_choice == 'EyesImage_IQAresnet_with_image_blur_label':
            in_gt_img = util.tensor2im_2(self.in_gt_img.data,len(self.image_paths))
            imgblur = util.tensor2im_2(self.imgblur.data,len(self.image_paths))
            # Acc
            # return Acc

            return OrderedDict([('in_gt_img', in_gt_img), ('imgblur', imgblur)])

        elif self.opt.model_choice == 'EyesImage_CLS_with_resnet' \
                or self.opt.model_choice == 'EyesImage_CLS_with_MyNet':
            pass
        # Acc
        # return Acc

        elif self.opt.model_choice == 'EyesImage_SR':
            in_gt_img = util.tensor2im_2(self.in_gt_img.data,len(self.image_paths))
            in_de_img = util.tensor2im_2(self.in_de_img.data,len(self.image_paths))
            SR_image = util.tensor2im_2(self.SR_image.data,len(self.image_paths))
            ves_mask_pre = util.tensor2im_2(self.ves_mask_pre.data,len(self.image_paths))
            return OrderedDict([('in_gt_img', in_gt_img), ('in_de_img', in_de_img), ('SR_image', SR_image),
                                ('ves_mask_pre', ves_mask_pre)])

        elif self.opt.model_choice == 'EyesImage_joint':
            in_de_img = util.tensor2im_2(self.in_de_img.data,len(self.image_paths))
            in_gt_img = util.tensor2im_2(self.in_gt_img.data,len(self.image_paths))
            SR_image = util.tensor2im_2(self.SR_image.data,len(self.image_paths))
            ves_mask_pre = util.tensor2im_2(self.ves_mask_pre.data,len(self.image_paths))
            blur_image = util.tensor2im_2(self.blur_image.data,len(self.image_paths))
            # Acc_cls, ACC_IQA
            # return Acc

            return OrderedDict([('in_de_img', in_de_img), ('in_gt_img', in_gt_img), ('SR_image', SR_image),
                                ('ves_mask_pre', ves_mask_pre), ('blur_image', blur_image)])
