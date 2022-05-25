import argparse
import torch
import os
import time
from util import util
from data.IQAdataset_Model import IQAdataset
from data.CLSdataset_Model import CLSdataset
from data.SRdataset_Model import SRdataset
from data.jointdataset_Model import jointdataset
from models.models import create_model


class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--project_root', type=str, default='/home/ubuntu/CLEAQ') #/home/houqs/temp_use/MyProject
        self.parser.add_argument('--CLS_weight_root', type=str, default='none'
                                 , help='./weights/CLS_resnet.th'
                                        './weights/CLS_MyNet.th')
        self.parser.add_argument('--segm_vessel_weight_root', type=str, default='./weights/Vessel.th'
                                 , help='./weights/Vessel.th')

        self.parser.add_argument('--segm_spot_weight_root', type=str, default='none'
                                 , help='./weights/Spot.th')
        self.parser.add_argument('--SR_blur_weight_root', type=str, default='none'
                                 , help='./weights/SR_blur.th')

        self.parser.add_argument('--dataroot', type=str, default='./dataset/dataset_for_joint/Kaggle/train'#./dataset/dataset_for_IQA/Kaggle/debug
                                 , help='./dataset/dataset_for_IQA/Kaggle/train,'
                                        './dataset/dataset_for_CLS/Kaggle/train,'
                                        './dataset/dataset_for_joint/Kaggle/train,'
                                        './dataset/dataset_for_SR/Kaggle/train')

        self.parser.add_argument('--batchSize', type=int, default=32)#64
        self.parser.add_argument('--loadSizeX', type=int, default=512)
        self.parser.add_argument('--loadSizeY', type=int, default=512)
        self.parser.add_argument('--fineSize', type=int, default=512)
        self.parser.add_argument('--input_nc', type=int, default=3)
        self.parser.add_argument('--output_nc', type=int, default=3)

        self.parser.add_argument('--model_choice', type=str,
                                 default='EyesImage_joint',
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

        self.parser.add_argument('--gpu_ids', type=str, default='0,1,2,3',
                                 help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')

        self.parser.add_argument('--name', type=str, default='experiment_joint',
                                 help='name of the experiment. It decides where to store samples and models.'
                                      'experiment_IQA_mask'
                                      'experiment_IQA_blur'
                                      'experiment_IQAres_mask'
                                      'experiment_IQAres_blur'
                                      'experiment_CLS'
                                      'experiment_CLSres'
                                      'experiment_SR'
                                      'experiment_joint')

        self.parser.add_argument('--dataset_mode', type=str, default='joint',
                                 help='chooses how datasets are loaded. [IQA | CLS | SR | joint]')
        self.parser.add_argument('--model', type=str, default='model_joint',
                                 help='chooses which models to use. model_IQA, model_CLS, model_SR, model_joint')

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

        expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
        return self.opt


class TrainOptions(BaseOptions):

    def initialize(self):
        BaseOptions.initialize(self)

        self.parser.add_argument('--save_latest_freq', type=int, default=5000,
                                 help='frequency of saving the latest results')
        self.parser.add_argument('--save_epoch_freq', type=int, default=1,
                                 help='frequency of saving checkpoints at the end of epochs')

        self.parser.add_argument('--continue_train', default=True, action='store_true',
                                 help='continue training: load the latest models')
        self.parser.add_argument('--epoch_count', type=int, default=1,
                                 help='the starting epoch count, we save the models by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')

        self.parser.add_argument('--which_epoch', type=str, default='latest',
                                 help='which epoch to load? set to latest to use latest cached models')

        self.parser.add_argument('--niter', type=int, default=150, help='# of iter at starting learning rate')
        self.parser.add_argument('--niter_decay', type=int, default=150,
                                 help='# of iter to linearly decay learning rate to zero')
        self.parser.add_argument('--beta1', type=float, default=0.9, help='momentum term of adam')
        self.parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate for adam')

        self.parser.add_argument('--lambda_ves', type=float, default=0.1, help='weight for vessel segmentation module')

        self.isTrain = True


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


def train(opt, data_loader, model):
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print('#training images = %d' % dataset_size)
    total_steps = 0
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        epoch_iter = 0
        if opt.model_choice == 'EyesImage_SR':
            psnr_batch_list = []
            ssim_batch_list = []
            for i, data in enumerate(dataset):
                total_steps += opt.batchSize
                epoch_iter += opt.batchSize
                model.set_input(data)
                psnr_list, ssim_list, loss_ves, loss_SR = model.optimize_parameters(opt, epoch)

                psnr_sum = 0
                ssim_sum = 0
                for j in range(len(psnr_list)):
                    psnr_sum += psnr_list[j]
                    ssim_sum += ssim_list[j]

                psnr_batch = psnr_sum / len(psnr_list)
                ssim_batch = ssim_sum / len(ssim_list)
                psnr_batch_list.append(psnr_batch)
                ssim_batch_list.append(ssim_batch)

                psnr_batch_sum = 0
                ssim_batch_sum = 0

                for k in range(len(psnr_batch_list)):
                    psnr_batch_sum += psnr_batch_list[k]
                    ssim_batch_sum += ssim_batch_list[k]

                psnr_avg = psnr_batch_sum / (i + 1)
                ssim_avg = ssim_batch_sum / (i + 1)

                print('epoch = %d, loss_ves= %f, loss_SR= %f, psnr_avg= %f, ssim_avg= %f' % (
                    epoch, loss_ves, loss_SR, psnr_avg, ssim_avg))
            if epoch % opt.save_epoch_freq == 0:
                print('saving the model at the end of epoch %d, iters %d' %
                      (epoch, total_steps))
                model.save(opt, 'latest')
                model.save(opt, epoch)
            print('End of epoch %d / %d \t Time Taken: %d sec' %
                  (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
            if epoch > opt.niter:
                model.update_learning_rate()
        elif opt.model_choice == 'EyesImage_CLS_with_resnet':
            acc_list = []
            kappa_list = []
            f1_list = []
            for i, data in enumerate(dataset):
                total_steps += opt.batchSize
                epoch_iter += opt.batchSize
                model.set_input(data)
                acc, kappa, f1, loss = model.optimize_parameters(opt, epoch)

                acc_sum = 0
                kappa_sum = 0
                f1_sum = 0
                acc_list.append(acc)
                kappa_list.append(kappa)
                f1_list.append(f1)
                for j in range(len(acc_list)):
                    acc_sum += acc_list[j]
                    kappa_sum += kappa_list[j]
                    f1_sum += f1_list[j]
                acc_avg = acc_sum / (i + 1)
                kappa_avg = kappa_sum / (i + 1)
                f1_avg = f1_sum / (i + 1)
                print('epoch = %d, loss_net: %f, acc: %f, kappa: %f, F1: %f' % (epoch, loss, acc_avg, kappa_avg, f1_avg))

            if epoch % opt.save_epoch_freq == 0:
                print('saving the model at the end of epoch %d, iters %d' %
                      (epoch, total_steps))
                model.save(opt, 'latest')
                model.save(opt, epoch)

            print('End of epoch %d / %d \t Time Taken: %d sec' %
                  (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

            if epoch > opt.niter:
                model.update_learning_rate()
        elif opt.model_choice == 'EyesImage_IQAresnet_with_image_blur_label':
            psnr_batch_list = []
            ssim_batch_list = []
            acc_list = []
            kappa_list = []
            f1_list = []
            for i, data in enumerate(dataset):
                total_steps += opt.batchSize
                epoch_iter += opt.batchSize
                model.set_input(data)
                psnr_list, ssim_list, acc, kappa, f1, loss_blur, loss_IQA = model.optimize_parameters(opt, epoch)

                psnr_sum = 0
                ssim_sum = 0

                for j in range(len(psnr_list)):
                    psnr_sum += psnr_list[j]
                    ssim_sum += ssim_list[j]

                psnr_batch = psnr_sum / len(psnr_list)
                ssim_batch = ssim_sum / len(ssim_list)
                psnr_batch_list.append(psnr_batch)
                ssim_batch_list.append(ssim_batch)

                psnr_batch_sum = 0
                ssim_batch_sum = 0

                for k in range(len(psnr_batch_list)):
                    psnr_batch_sum += psnr_batch_list[k]
                    ssim_batch_sum += ssim_batch_list[k]

                psnr_avg = psnr_batch_sum / (i + 1)
                ssim_avg = ssim_batch_sum / (i + 1)

                acc_sum = 0
                kappa_sum = 0
                f1_sum = 0
                acc_list.append(acc)
                kappa_list.append(kappa)
                f1_list.append(f1)
                for j in range(len(acc_list)):
                    acc_sum += acc_list[j]
                    kappa_sum += kappa_list[j]
                    f1_sum += f1_list[j]
                acc_avg = acc_sum / (i + 1)
                kappa_avg = kappa_sum /(i+1)
                f1_avg = f1_sum / (i+1)

                print('epoch = %d, loss_blur= %f, loss_IQA= %f, psnr_avg= %f, ssim_avg= %f, acc_avg= %f, kappa_avg= %f, f1_avg= %f' % (
                    epoch, loss_blur, loss_IQA, psnr_avg, ssim_avg, acc_avg, kappa_avg, f1_avg))

            if epoch % opt.save_epoch_freq == 0:
                print('saving the model at the end of epoch %d, iters %d' %
                      (epoch, total_steps))
                model.save(opt, 'latest')
                model.save(opt, epoch)
            print('End of epoch %d / %d \t Time Taken: %d sec' %
                  (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
            if epoch > opt.niter:
                model.update_learning_rate()
        elif opt.model_choice == 'EyesImage_joint':
            blur_psnr_batch_list = []
            blur_ssim_batch_list = []
            # SR_psnr_batch_list = []
            # SR_ssim_batch_list = []
            cls_acc_list = []
            cls_kappa_list = []
            cls_f1_list = []
            IQA_acc_list = []
            IQA_kappa_list = []
            IQA_f1_list = []
            for i, data in enumerate(dataset):
                total_steps += opt.batchSize
                epoch_iter += opt.batchSize
                model.set_input(data)
                # cls_acc, cls_kappa, cls_f1, IQA_acc, IQA_kappa, IQA_f1, SR_psnr_list, SR_ssim_list, blur_psnr_list, blur_ssim_list, cls_loss, IQA_loss, ves_loss, SR_loss, blur_loss = model.optimize_parameters(
                #     opt, epoch)
                cls_acc, cls_kappa, cls_f1, IQA_acc, IQA_kappa, IQA_f1, blur_psnr_list, blur_ssim_list, cls_loss, IQA_loss, ves_loss, blur_loss = model.optimize_parameters(
                    opt, epoch)

                blur_psnr_sum = 0
                blur_ssim_sum = 0

                # SR_psnr_sum = 0
                # SR_ssim_sum = 0

                for j in range(len(blur_psnr_list)):
                    blur_psnr_sum += blur_psnr_list[j]
                    blur_ssim_sum += blur_ssim_list[j]

                blur_psnr_batch = blur_psnr_sum / len(blur_psnr_list)
                blur_ssim_batch = blur_ssim_sum / len(blur_ssim_list)
                blur_psnr_batch_list.append(blur_psnr_batch)
                blur_ssim_batch_list.append(blur_ssim_batch)

                blur_psnr_batch_sum = 0
                blur_ssim_batch_sum = 0

                for k in range(len(blur_psnr_batch_list)):
                    blur_psnr_batch_sum += blur_psnr_batch_list[k]
                    blur_ssim_batch_sum += blur_ssim_batch_list[k]

                blur_psnr_avg = blur_psnr_batch_sum / (i + 1)
                blur_ssim_avg = blur_ssim_batch_sum / (i + 1)

                # for j in range(len(SR_psnr_list)):
                #     SR_psnr_sum += SR_psnr_list[j]
                #     SR_ssim_sum += SR_ssim_list[j]
                #
                # SR_psnr_batch = SR_psnr_sum / len(SR_psnr_list)
                # SR_ssim_batch = SR_ssim_sum / len(SR_ssim_list)
                # SR_psnr_batch_list.append(SR_psnr_batch)
                # SR_ssim_batch_list.append(SR_ssim_batch)
                #
                # SR_psnr_batch_sum = 0
                # SR_ssim_batch_sum = 0
                #
                # for k in range(len(SR_psnr_batch_list)):
                #     SR_psnr_batch_sum += SR_psnr_batch_list[k]
                #     SR_ssim_batch_sum += SR_ssim_batch_list[k]
                #
                # SR_psnr_avg = SR_psnr_batch_sum / (i + 1)
                # SR_ssim_avg = SR_ssim_batch_sum / (i + 1)

                cls_acc_sum = 0
                cls_kappa_sum = 0
                cls_f1_sum = 0
                cls_acc_list.append(cls_acc)
                cls_kappa_list.append(cls_kappa)
                cls_f1_list.append(cls_f1)
                for j in range(len(cls_acc_list)):
                    cls_acc_sum += cls_acc_list[j]
                    cls_kappa_sum += cls_kappa_list[j]
                    cls_f1_sum += cls_f1_list[j]
                cls_acc_avg = cls_acc_sum / (i + 1)
                cls_kappa_avg = cls_kappa_sum / (i + 1)
                cls_f1_avg = cls_f1_sum / (i + 1)

                IQA_acc_sum = 0
                IQA_kappa_sum = 0
                IQA_f1_sum = 0
                IQA_acc_list.append(IQA_acc)
                IQA_kappa_list.append(IQA_kappa)
                IQA_f1_list.append(IQA_f1)
                for j in range(len(IQA_acc_list)):
                    IQA_acc_sum += IQA_acc_list[j]
                    IQA_kappa_sum += IQA_kappa_list[j]
                    IQA_f1_sum += IQA_f1_list[j]
                IQA_acc_avg = IQA_acc_sum / (i + 1)
                IQA_kappa_avg = IQA_kappa_sum / (i+1)
                IQA_f1_avg = IQA_f1_sum / (i+1)

                # print('epoch = %d, loss_cls= %f, loss_IQA= %f, loss_ves= %f, loss_SR= %f, loss_blur= %f, '
                #       'blur_psnr_avg= %f, blur_ssim_avg= %f, SR_psnr_avg= %f, SR_ssim_avg= %f, cls_acc_avg= %f, '
                #       'cls_kappa_avg= %f , cls_f1_avg= %f, IQA_acc_avg= %f, IQA_kappa_avg= %f, IQA_f1_avg= %f' % (
                #           epoch, cls_loss, IQA_loss, ves_loss, SR_loss, blur_loss, blur_psnr_avg, blur_ssim_avg,
                #           SR_psnr_avg, SR_ssim_avg, cls_acc_avg, cls_kappa_avg, cls_f1_avg, IQA_acc_avg, IQA_kappa_avg, IQA_f1_avg))
                print('epoch = %d, loss_cls= %f, loss_IQA= %f, loss_ves= %f, loss_blur= %f, '
                      'blur_psnr_avg= %f, blur_ssim_avg= %f, cls_acc_avg= %f, '
                      'cls_kappa_avg= %f , cls_f1_avg= %f, IQA_acc_avg= %f, IQA_kappa_avg= %f, IQA_f1_avg= %f' % (
                          epoch, cls_loss, IQA_loss, ves_loss, blur_loss, blur_psnr_avg, blur_ssim_avg,
                          cls_acc_avg, cls_kappa_avg, cls_f1_avg, IQA_acc_avg, IQA_kappa_avg, IQA_f1_avg))

            if epoch % opt.save_epoch_freq == 0:
                print('saving the model at the end of epoch %d, iters %d' %
                      (epoch, total_steps))
                model.save(opt, 'latest')
                model.save(opt, epoch)
            print('End of epoch %d / %d \t Time Taken: %d sec' %
                  (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
            if epoch > opt.niter:
                model.update_learning_rate()


opt = TrainOptions().parse()
data_loader = CreateDataLoader(opt)
model = create_model(opt)
train(opt, data_loader, model)
