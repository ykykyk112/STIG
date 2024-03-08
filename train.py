import torch
from model.loss import RectAverage
from utils.util import OptionConfigurator, fix_randomseed
from utils.dataset import get_dataset_from_option
from model.stig_model import STIG
#from classifier.cnn import build_classifier_from_opt
from tqdm import tqdm
import os
from utils.visualizing import Visualizer
from utils.log import Logger
from utils.metric import FIDCalculator
from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__' :

    def inverse_norm(tensor) :
        return (tensor + 1.) * 0.5

    # Preparing training step
    # Fix the randomness for reproducibility
    # Set GPU environment and tensorboard logging
    fix_randomseed(42)
    
    opt = OptionConfigurator().parse_options()
    train_loader, test_loader = get_dataset_from_option(opt)

    is_gpu = torch.cuda.is_available()
    device = torch.device(opt.device) if is_gpu else torch.device('cpu')

    model = STIG(opt, device).to(device)

    save_path = os.path.join('./results', opt.dst)
    os.mkdir(save_path)
    os.mkdir(os.path.join(save_path, 'tensorboard'))
    os.mkdir(os.path.join(save_path, 'sample'))
    os.mkdir(os.path.join(save_path, 'eval'))
    os.mkdir(os.path.join(save_path, 'average'))
    os.mkdir(os.path.join(save_path, 'psd'))

    OptionConfigurator().log_options(opt, save_path)

    ts_board = SummaryWriter(os.path.join(save_path, 'tensorboard'))
    n_sample = len(train_loader) * opt.batch_size

    # Load utilities for logging, visualization, ...
    visualizer = Visualizer(opt)
    logger = Logger(ts_board)
    fid_calculator = FIDCalculator(opt, os.path.join(save_path, 'eval'))
    radial_profile = RectAverage(opt.size, torch.device(opt.device))

    best_fid_score = 1e10

    for epoch in range(opt.epoch) :
        print('[{}-th epoch]'.format(epoch+1))

        # Training Procedure for spectrum translation
        for n, sample in enumerate(tqdm(train_loader, desc="{:17s}".format('Training State'), mininterval=0.0001)) :
            
            if not model.first_setting :
                model.data_dependent_init(sample, n_sample)

            model.set_input(sample)
            model.update_optimizer()
            
            logger.step(model, epoch * n_sample + n)
            visualizer.step(model, epoch, n)

            model.step_scheduler()

    # Inference Procedure
    # From the trained model parameter, save stig-refined images and magnitudes.
    for n, sample in enumerate(tqdm(test_loader, desc="{:17s}".format('Inference State'), mininterval=0.0001)) :

        if not fid_calculator.baseline_config :
            fid_calculator.set_baseline_config(sample, n)

        model.set_input(sample)
        model.forward()

        fid_calculator.step(model, n)

    model.save_checkpoint(save_path, epoch)

    ts_board.close()