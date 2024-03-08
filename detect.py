import torch
from model.cnn import eval_from_opt, train_from_opt
from utils.util import OptionConfigurator, fix_randomseed

if __name__ == '__main__' :
    
    fix_randomseed(42)
    opt = OptionConfigurator().parse_options()

    device = torch.device(opt.device)
    
    print('[  Current classifier : {}  ]\n'.format(opt.classifier))

    if opt.classifier in ['cnn', 'vit'] :
        if opt.is_train :
            train_from_opt(opt)
        else :
            eval_from_opt(opt)

    else :
        raise TypeError('Non-existing classifier!')