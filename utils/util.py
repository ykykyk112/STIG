import argparse
import os
import time
import torch
import numpy as np
import random

class OptionConfigurator() :

    def __init__(self) :
        self.parser = argparse.ArgumentParser(formatter_class = argparse.RawDescriptionHelpFormatter)

    def initialize(self, parser) :
        now = time.localtime()
        parser.add_argument('--epoch', type = int, default = 10, help = 'Setting for training epoch')
        parser.add_argument('--device', type = int, default = 0, help = 'Setting for training epoch')
        parser.add_argument('--batch_size', type = int, default = 1, help = 'Size of mini-batch of input image tensor')
        parser.add_argument('--sample_frequency', type = int, default = 100, help = 'Sampling frequency to visualize translated output')
        parser.add_argument('--average_frequency', type = int, default = 1000, help = 'Sampling frequency to visualize averaged profile of translated output')
        parser.add_argument('--num_patch', type = int, default = 256, help = 'Number of patches for get NCELoss')
        parser.add_argument('--size', type = int, default = 256, help = 'Width and height of input image tensor')
        parser.add_argument('--sigma', type = int, default = 8, help = 'Width and height of replacing box of fake magnitude')
        parser.add_argument('--input_nc', type = int, default = 3, help = 'Input channel of image tensor')
        parser.add_argument('--output_nc', type = int, default = 3, help = 'Output channel of image tensor')
        parser.add_argument('--lr', type=float, default=0.00008, help='Initial learning rate for adam')
        parser.add_argument('--beta1', type=float, default=0.5, help='Initial beta1 rate for adam')
        parser.add_argument('--beta2', type=float, default=0.999, help='Initial beta2 rate for adam')
        parser.add_argument('--lambda_GAN', type=float, default=3.0, help='Weight value of GAN loss for training generator')
        parser.add_argument('--lambda_NCE', type=float, default=10.0, help='Weight value of NCE loss for training generator')
        parser.add_argument('--lambda_PSD', type=float, default=3.0, help='Weight value of NCE loss for training generator')
        parser.add_argument('--lambda_LF', type=float, default=3.0, help='Weight value of NCE loss for training generator')
        parser.add_argument('--lambda_identity', type=float, default=1.0, help='Weight value of identity loss for training generator')
        parser.add_argument('--data', type = str, default = 'cyclegan', help = 'Dataset for training the spoofing model')
        parser.add_argument('--norm', type = str, default = 'instance', help = 'Normalization method of Resnet based decoder and encoder')
        parser.add_argument('--nce_layers', type=str, default='0,4,8,12,16', help='Compute NCE loss on which layers')
        parser.add_argument('--nce_idt', type = str2bool, default = True, help = 'Identity loss for STIG')
        parser.add_argument('--nce_T', type=float, default=0.07, help='Temperature for NCE loss')
        parser.add_argument('--dst', type = str, default = '{:02d}_{:02d}_{:02d}-{:02d}_{:02d}_{:02d}'.format(now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec),help = 'folder that save model parameter and log.txt')

        parser.add_argument('--eval_mode', type = str, default = 'magnitude_fid', help = 'evaluation profile name')
        parser.add_argument('--class_epoch', type = int, default = 20, help = 'Setting for classification training epoch')
        parser.add_argument('--class_batch_size', type = int, default = 32, help = 'Setting for classification batch size')
        parser.add_argument('--classifier', type = str, default = 'cnn', help = 'Classification model to detect deep fake images')
        parser.add_argument('--is_train', type = str2bool, default = True, help = 'Flag for training a detector')
        parser.add_argument('--eval_root', type = str, default = 'none', help = 'Directory path of fake dataset for classification')

        parser.add_argument('--inference_data', help = 'root of dataset directory for inference samples')

        return parser

    def print_options(self, opt) :

        message = ''
        message += '----------------- Options ---------------\n\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<15}{}\n'.format(str(k), str(v), comment)
        message += '\n----------------- End -------------------'
        print(message)

    def log_options(self, opt, save_path) :

        with open(os.path.join(save_path, 'log.txt'), 'w') as r:
            message = ''
            message += '----------------- Options ---------------\n\n'
            for k, v in sorted(vars(opt).items()):
                comment = ''
                default = self.parser.get_default(k)
                if v != default:
                    comment = '\t[default: %s]' % str(default)
                message += '{:>25}: {:<15}{}\n'.format(str(k), str(v), comment)
            message += '\n----------------- End -------------------'
            r.writelines(message)
            r.close()

    def parse_options(self) :
        self.parser = self.initialize(self.parser)
        opt = self.parser.parse_args()
        self.print_options(opt)
        return opt

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1', 'True'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0', 'False'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__' :
    parser = OptionConfigurator()
    options = parser.parse_options()
    print(options)

def fix_randomseed(seed_number) :
    torch.manual_seed(seed_number)
    #torch.use_deterministic_algorithms(True)
    torch.cuda.manual_seed(seed_number)
    torch.cuda.manual_seed_all(seed_number)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed_number)
    random.seed(seed_number)