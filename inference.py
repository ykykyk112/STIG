import torch
from utils.util import OptionConfigurator, fix_randomseed
from utils.dataset import get_inference_dataset_from_option
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
    loader = get_inference_dataset_from_option(opt)

    is_gpu = torch.cuda.is_available()
    device = torch.device(opt.device) if is_gpu else torch.device('cpu')

    model = STIG(opt, device).to(device)

    save_path = os.path.join('./results', opt.dst)
    os.makedirs(save_path, exist_ok = True)
    os.makedirs(os.path.join(save_path, 'inference'), exist_ok = True)
    #fid_calculator = FIDCalculator(opt, os.path.join(save_path, 'inference'), eval = True)

    for n, sample in enumerate(tqdm(loader, desc="{:17s}".format('Inference State'), mininterval=0.0001)) :

        model.set_input(sample, evaluation = True)
        model.evaluation()
