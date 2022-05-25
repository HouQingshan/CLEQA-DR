import argparse
import os
import numpy as np
import torch
import cv2

from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM
from PIL import Image
from util import util
from models.models import create_model
from data.IQAdataset_Model import IQAdataset
from data.CLSdataset_Model import CLSdataset
from data.SRdataset_Model import SRdataset
from data.jointdataset_Model import jointdataset
from util.metrics import *


class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--project_root', type=str, default='G:/CLEAQ') #/home/houqs/temp_use/MyProject
        self.parser.add_argument('--CLS_weight_root', type=str, default='none'
                                 , help='./weights/CLS_resnet.th'
                                        './weights/CLS_MyNet.th')
        self.parser.add_argument('--segm_vessel_weight_root', type=str, default='./weights/Vessel.th'
                                 , help='./weights/Vessel.th')
        self.parser.add_argument('--segm_spot_weight_root', type=str, default='none'
                                 , help='./weights/Spot.th')
        self.parser.add_argument('--SR_blur_weight_root', type=str, default='none'
                                 , help='./weights/SR_blur.th')
        self.parser.add_argument('--dataroot', type=str, default='./dataset/dataset_for_IQA/Kaggle/test'
                                 , help='./dataset/dataset_for_IQA/Kaggle/test,'
                                        './dataset/dataset_for_CLS/Kaggle/test,'
                                        './dataset/dataset_for_joint/Kaggle/test,'
                                        './dataset/dataset_for_SR/Kaggle/test')

        self.parser.add_argument('--maskroot', type=str, default='./dataset/dataset_for_IQA/test/debug/mask_ori',
                                 help='./dataset/dataset_for_IQA/Kaggle/test/mask_ori'
                                      './dataset/dataset_for_CLS/Kaggle/test/image_mask'
                                      './dataset/dataset_for_joint/Kaggle/test/gt_image_mask'
                                      './dataset/dataset_for_SR/Kaggle/test/image_mask')
        self.parser.add_argument('--batchSize', type=int, default=3)
        self.parser.add_argument('--loadSizeX', type=int, default=512)
        self.parser.add_argument('--loadSizeY', type=int, default=512)
        self.parser.add_argument('--fineSize', type=int, default=512)
        self.parser.add_argument('--input_nc', type=int, default=3)
        self.parser.add_argument('--output_nc', type=int, default=3)

        self.parser.add_argument('--model_choice', type=str,
                                 default='EyesImage_IQAresnet_with_image_blur_label',
                                 help='chooses which nets to use. '
                                      'EyesImage_IQA_with_masklabel, '
                                      'EyesImage_IQA_with_image_blur_label,'
                                      'EyesImage_IQAresnet_with_masklabel,'
                                      'EyesImage_IQAresnet_with_image_blur_label,'
                                      'EyesImage_CLS_with_resnet,'
                                      'EyesImage_CLS_with_MyNet,'
                                      'EyesImage_SR,'
                                      'EyesImage_joint')
        self.parser.add_argument('--checkpoints_dir', type=str,
                                 default='./model_pth/')
        self.parser.add_argument('--test_option', type=str,
                                 default='./result/')

        self.parser.add_argument('--gpu_ids', type=str, default='0,1,2,3',
                                 help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')

        self.parser.add_argument('--name', type=str, default='experiment_CLSres',
                                 help='name of the experiment. It decides where to store samples and models.'
                                      'experiment_IQA_mask'
                                      'experiment_IQA_blur'
                                      'experiment_IQAres_mask'
                                      'experiment_IQAres_blur'
                                      'experiment_CLS'
                                      'experiment_CLSres'
                                      'experiment_SR'
                                      'experiment_joint')
        self.parser.add_argument('--test_name_opt', type=str,
                                 default='experiment_IQAres_blur_test_option',
                                 help='experiment_IQA_mask_test_option'
                                      'experiment_IQA_blur_test_option'
                                      'experiment_IQAres_mask_test_option'
                                      'experiment_IQAres_blur_test_option'
                                      'experiment_CLS_test_option'
                                      'experiment_CLSres_test_option'
                                      'experiment_SR_test_option'
                                      'experiment_joint_test_option')

        self.parser.add_argument('--dataset_mode', type=str, default='IQA',
                                 help='chooses how datasets are loaded. [IQA | CLS | SR | joint]')
        self.parser.add_argument('--model', type=str, default='test',
                                 help='chooses which models to use. test')

        self.parser.add_argument('--nThreads', default=0, type=int, help='# threads for loading data')

        self.parser.add_argument('--norm', type=str, default='instance',
                                 help='instance normalization or batch normalization')

        self.parser.add_argument('--serial_batches', action='store_true',
                                 help='if true, takes images in order to make batches, otherwise takes them randomly')

        self.parser.add_argument('--max_dataset_size', type=int, default=float("inf"),
                                 help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')

        self.parser.add_argument('--resize_or_crop', type=str, default='scale',
                                 help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop|scale]')

        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()

        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain
        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_ids in str_ids:
            id = int(str_ids)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

        print('---------------Options----------------')
        args = vars(self.opt)
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('---------------ends----------------')

        expr_dir = os.path.join(self.opt.test_option, self.opt.test_name_opt)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
        return self.opt


class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
        self.parser.add_argument('--results_dir', type=str,
                                 default='./result/experiment_IQAres_blur_test_option',
                                 help='saves results here.'
                                      './result/experiment_IQA_mask_test_option'
                                      './result/experiment_IQA_blur_test_option'
                                      './result/experiment_IQAres_mask_test_option'
                                      './result/experiment_IQAres_blur_test_option'
                                      './result/experiment_CLS_test_option'
                                      './result/experiment_CLSres_test_option'
                                      './result/experiment_SR_test_option'
                                      './result/experiment_joint_test_option')
        self.parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        self.parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        self.parser.add_argument('--which_epoch', type=str, default='latest',
                                 help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--how_many', type=int, default=100000, help='how many test images to run')
        self.isTrain = False


class BaseDataLoader():
    def __init__(self):
        pass

    def initialize(self, opt):
        self.opt = opt
        pass

    def load_data(self):
        return None


def CreateDataset(opt):
    dataset = None
    if opt.dataset_mode == 'IQA':
        dataset = IQAdataset()
    elif opt.dataset_mode == 'CLS':
        dataset = CLSdataset()
    elif opt.dataset_mode == 'SR':
        dataset = SRdataset()
    elif opt.dataset_mode == 'joint':
        dataset = jointdataset()
    else:
        raise ValueError("Dataset [%s] not recognized." % opt.dataset_mode)

    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt)
    return dataset


class EyesDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'EyesDatasetDataLoader'

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        self.dataset = CreateDataset(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.nThreads))

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)


def CreateDataLoader(opt):
    data_loader = EyesDatasetDataLoader()
    print(data_loader.name())
    data_loader.initialize(opt)
    return data_loader


opt = TestOptions().parse()
# opt.nThreads = 0
# opt.batchSize = 1
# opt.serial_batches = True
# opt.no_flip = True

model = create_model(opt)
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()

# haven't changed
if opt.model_choice == 'EyesImage_IQA_with_masklabel' or opt.model_choice == 'EyesImage_IQAresnet_with_masklabel':
    for i, data in enumerate(dataset):
        if i >= opt.how_many:
            break

        model.set_input(data)
        model.test()
        visuals = model.get_current_visuals()
        img_path = model.get_image_paths()
        img_path_len = len(img_path)

        for j in range(0, img_path_len):
            folder, img_name = os.path.split(img_path[j])

            mask_dir = os.path.join(opt.maskroot, img_name[0:-4] + '.png')
            mask = np.array(Image.open(mask_dir).convert('L'))
            mask = cv2.resize(mask, (opt.fineSize, opt.fineSize))
            mask[mask >= 200] = 255
            mask[mask != 255] = 0
            print('process image... %s' % img_path[j])
            mask = np.expand_dims(mask, axis=2)
            mask = np.array(mask, np.float32) / 255.0

            in_gt_img = visuals['in_gt_img'] * mask
            in_gt_img = in_gt_img.astype(np.uint8)

            spot_map = visuals['spot_map']
            spot_map = (spot_map, spot_map, spot_map)
            spot_map = np.array(spot_map)
            out_spot_map = spot_map[:, :, :, 0]
            out_spot_map = (np.transpose(out_spot_map, (1, 2, 0))) * mask
            out_spot_map = out_spot_map.astype(np.uint8)

            dir = os.path.join(opt.results_dir, '%s%s' % (img_name[0:-4], '.png'))
            util.save_image(in_gt_img, dir)

            dir = os.path.join(opt.results_dir, '%s%s' % (img_name[0:-4], '_spot_mask.png'))
            util.save_image(out_spot_map, dir)

# IQA&blur
elif opt.model_choice == 'EyesImage_IQA_with_image_blur_label' or opt.model_choice == 'EyesImage_IQAresnet_with_image_blur_label':
    psnr_list = []
    ssim_list = []
    flag = True
    for i, data in enumerate(dataset):
        if i >= opt.how_many:
            break

        model.set_input(data)
        out, label = model.test()
        if flag:
            out_list = out
            label_list = label
            flag = False
        else:
            out_list = torch.cat((out_list, out), dim=0)
            label_list = torch.cat((label_list, label), dim=0)

        visuals = model.get_current_visuals()
        img_path = model.get_image_paths()

        for k in range(len(img_path)):
            folder, img_name = os.path.split(img_path[k])

            mask_dir = os.path.join(opt.maskroot, img_name[0:-4] + '.png')
            mask = np.array(Image.open(mask_dir).convert('L'))
            mask = cv2.resize(mask, (opt.fineSize, opt.fineSize))
            mask[mask >= 200] = 255
            mask[mask != 255] = 0
            print('process image... %s' % img_path[k])
            mask = np.expand_dims(mask, axis=2)
            mask = np.array(mask, np.float32) / 255.0

            in_gt_img = visuals['in_gt_img'][k] * mask
            in_gt_img = in_gt_img.astype(np.uint8)

            imgblur = visuals['imgblur'][k] * mask
            imgblur = imgblur.astype(np.uint8)

            psnr = PSNR(in_gt_img, imgblur)
            ssim = SSIM(in_gt_img, imgblur, channel_axis=-1)
            psnr_list.append(psnr)
            ssim_list.append(ssim)

            dir = os.path.join(opt.results_dir, '%s%s' % (img_name[0:-4], '.png'))
            util.save_image(in_gt_img, dir)

            dir = os.path.join(opt.results_dir, '%s%s' % (img_name[0:-4], '_blur.png'))
            util.save_image(imgblur, dir)
    acc = calculate_accuracy(out_list, label_list)
    kappa = compute_kappa(out_list, label_list)
    f1 = compute_f1(out_list, label_list)
    psnr_sum = 0
    ssim_sum = 0
    for n in range(len(psnr_list)):
        psnr_sum += psnr_list[n]
        ssim_sum += ssim_list[n]
    psnr_avg = psnr_sum / len(psnr_list)
    ssim_avg = ssim_sum / len(ssim_list)
    print('psnr_avg: %f, ssim_avg: %f, acc: %f, kappa: %f, f1: %f' % (psnr_avg, ssim_avg, acc, kappa, f1))

# CLS
elif opt.model_choice == 'EyesImage_CLS_with_resnet' or opt.model_choice == 'EyesImage_CLS_with_MyNet':
    flag = True
    for i, data in enumerate(dataset):
        if i >= opt.how_many:
            break

        model.set_input(data)
        out, label = model.test()
        if flag:
            out_list = out
            label_list = label
            flag = False
        else:
            out_list = torch.cat((out_list, out), dim=0)
            label_list = torch.cat((label_list, label), dim=0)
        img_path = model.get_image_paths()
        for k in range(len(img_path)):
            print('process image... %s' % img_path[k])
    acc = calculate_accuracy(out_list, label_list)
    kappa = compute_kappa(out_list, label_list)
    f1 = compute_f1(out_list, label_list)
    print('acc: %f, kappa: %f, f1: %f' % (acc, kappa, f1))
# SR
elif opt.model_choice == 'EyesImage_SR':
    psnr_list = []
    ssim_list = []

    for i, data in enumerate(dataset):
        if i >= opt.how_many:
            break

        model.set_input(data)
        model.test()
        visuals = model.get_current_visuals()
        img_path = model.get_image_paths()

        for k in range(len(img_path)):
            folder, img_name = os.path.split(img_path[k])

            mask_dir = os.path.join(opt.maskroot, img_name[0:-4] + '.png')
            mask = np.array(Image.open(mask_dir).convert('L'))
            mask = cv2.resize(mask, (opt.fineSize, opt.fineSize))
            mask[mask >= 200] = 255
            mask[mask != 255] = 0
            print('process image... %s' % img_path[k])
            mask = np.expand_dims(mask, axis=2)
            mask = np.array(mask, np.float32) / 255.0

            in_gt_img = visuals['in_gt_img'][k] * mask
            in_gt_img = in_gt_img.astype(np.uint8)

            in_de_img = visuals['in_de_img'][k] * mask
            in_de_img = in_de_img.astype(np.uint8)

            SR_image = visuals['SR_image'][k] * mask
            SR_image = SR_image.astype(np.uint8)

            psnr = PSNR(in_gt_img, SR_image)
            ssim = SSIM(in_gt_img, SR_image, channel_axis=-1)
            psnr_list.append(psnr)
            ssim_list.append(ssim)

            ves_map = visuals['ves_mask_pre'][k]
            ves_map = (ves_map, ves_map, ves_map)
            ves_map = np.array(ves_map)
            out_ves_map = ves_map[:, :, :, 0]
            out_ves_map = (np.transpose(out_ves_map, (1, 2, 0))) * mask
            out_ves_map = out_ves_map.astype(np.uint8)

            dir = os.path.join(opt.results_dir, '%s%s' % (img_name[0:-4], '.png'))
            util.save_image(in_gt_img, dir)

            dir = os.path.join(opt.results_dir, '%s%s' % (img_name[0:-4], '_de.png'))
            util.save_image(in_de_img, dir)

            dir = os.path.join(opt.results_dir, '%s%s' % (img_name[0:-4], '_SR.png'))
            util.save_image(SR_image, dir)

            dir = os.path.join(opt.results_dir, '%s%s' % (img_name[0:-4], '_ves_mask.png'))
            util.save_image(out_ves_map, dir)
    psnr_sum = 0
    ssim_sum = 0
    for n in range(len(psnr_list)):
        psnr_sum += psnr_list[n]
        ssim_sum += ssim_list[n]
    psnr_avg = psnr_sum / len(psnr_list)
    ssim_avg = ssim_sum / len(ssim_list)
    print('psnr_avg: %f, ssim_avg: %f' % (psnr_avg, ssim_avg))

# joint
elif opt.model_choice == 'EyesImage_joint':
    blur_psnr_list = []
    blur_ssim_list = []
    # SR_psnr_list = []
    # SR_ssim_list = []
    flag = True
    for i, data in enumerate(dataset):
        if i >= opt.how_many:
            break

        model.set_input(data)
        IQA_out, IQA_label, cls_out, cls_label = model.test()
        if flag:
            IQA_out_list = IQA_out
            IQA_label_list = IQA_label
            cls_out_list = cls_out
            cls_label_list = cls_label
            flag = False
        else:
            IQA_out_list = torch.cat((IQA_out_list, IQA_out), dim=0)
            IQA_label_list = torch.cat((IQA_label_list, IQA_label), dim=0)
            cls_out_list = torch.cat((cls_out_list, cls_out), dim=0)
            cls_label_list = torch.cat((cls_label_list, cls_label), dim=0)

        visuals = model.get_current_visuals()
        img_path = model.get_image_paths()

        for k in range(len(img_path)):
            folder, img_name = os.path.split(img_path[k])

            mask_dir = os.path.join(opt.maskroot, img_name[0:-4] + '.png')
            mask = np.array(Image.open(mask_dir).convert('L'))
            mask = cv2.resize(mask, (opt.fineSize, opt.fineSize))
            mask[mask >= 200] = 255
            mask[mask != 255] = 0
            print('process image... %s' % img_path[k])
            mask = np.expand_dims(mask, axis=2)
            mask = np.array(mask, np.float32) / 255.0

            in_gt_img = visuals['in_gt_img'][k] * mask
            in_gt_img = in_gt_img.astype(np.uint8)

            in_de_img = visuals['in_de_img'][k] * mask
            in_de_img = in_de_img.astype(np.uint8)

            # SR_image = visuals['SR_image'][k] * mask
            # SR_image = SR_image.astype(np.uint8)

            blur_image = visuals['blur_image'][k] * mask
            blur_image = blur_image.astype(np.uint8)

            blur_psnr = PSNR(in_de_img, blur_image)
            blur_ssim = SSIM(in_de_img, blur_image, channel_axis=-1)
            blur_psnr_list.append(blur_psnr)
            blur_ssim_list.append(blur_ssim)

            # SR_image = visuals['SR_image'][k] * mask
            # SR_image = SR_image.astype(np.uint8)
            #
            # SR_psnr = PSNR(in_gt_img, SR_image)
            # SR_ssim = SSIM(in_gt_img, SR_image, channel_axis=-1)
            # SR_psnr_list.append(SR_psnr)
            # SR_ssim_list.append(SR_ssim)

            ves_map = visuals['ves_mask_pre'][k]
            ves_map = (ves_map, ves_map, ves_map)
            ves_map = np.array(ves_map)
            out_ves_map = ves_map[:, :, :, 0]
            out_ves_map = (np.transpose(out_ves_map, (1, 2, 0))) * mask
            out_ves_map = out_ves_map.astype(np.uint8)

            dir = os.path.join(opt.results_dir, '%s%s' % (img_name[0:-4], '.png'))
            util.save_image(in_gt_img, dir)

            dir = os.path.join(opt.results_dir, '%s%s' % (img_name[0:-4], '_de.png'))
            util.save_image(in_de_img, dir)

            # dir = os.path.join(opt.results_dir, '%s%s' % (img_name[0:-4], '_SR.png'))
            # util.save_image(SR_image, dir)

            dir = os.path.join(opt.results_dir, '%s%s' % (img_name[0:-4], '_blur.png'))
            util.save_image(blur_image, dir)

            dir = os.path.join(opt.results_dir, '%s%s' % (img_name[0:-4], '_ves_mask.png'))
            util.save_image(out_ves_map, dir)
    IQA_acc = calculate_accuracy(IQA_out_list, IQA_label_list)
    IQA_kappa = compute_kappa(IQA_out_list, IQA_label_list)
    IQA_f1 = compute_f1(IQA_out_list, IQA_label_list)
    IQA_metrix = compute_confusion_matrix(IQA_out_list, IQA_label_list)

    cls_acc = calculate_accuracy(cls_out_list, cls_label_list)
    cls_kappa = compute_kappa(cls_out_list, cls_label_list)
    cls_f1 = compute_f1(cls_out_list, cls_label_list)
    cls_metrix = compute_confusion_matrix(cls_out_list, cls_label_list)

    blur_psnr_sum = 0
    blur_ssim_sum = 0
    for n in range(len(blur_psnr_list)):
        blur_psnr_sum += blur_psnr_list[n]
        blur_ssim_sum += blur_ssim_list[n]
    blur_psnr_avg = blur_psnr_sum / len(blur_psnr_list)
    blur_ssim_avg = blur_ssim_sum / len(blur_ssim_list)

    # SR_psnr_sum = 0
    # SR_ssim_sum = 0
    # for n in range(len(SR_psnr_list)):
    #     SR_psnr_sum += SR_psnr_list[n]
    #     SR_ssim_sum += SR_ssim_list[n]
    # SR_psnr_avg = SR_psnr_sum / len(SR_psnr_list)
    # SR_ssim_avg = SR_ssim_sum / len(SR_ssim_list)
    print('blur_psnr_avg: %f, blur_ssim_avg: %f, IQA_acc: %f, IQA_kappa: %f, IQA_f1: %f, cls_acc: %f, cls_kappa: %f, '
          'cls_f1: %f' % (blur_psnr_avg, blur_ssim_avg, IQA_acc, IQA_kappa, IQA_f1, cls_acc, cls_kappa, cls_f1))
    print(IQA_metrix)
    print(cls_metrix)