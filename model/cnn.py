import os
import torch
from tqdm import tqdm
from torch import nn
from pytorch_pretrained_vit import ViT
from utils.dataset import ClassificationDataset, TrainValidDataset

'''
SimpleCNN : Shallow cnn-based network for fake image detection in the frequency domain.
Input : Real or generated magnitude spectrum -> torch.Tensor
Output : Prediction -> torch.Tensor

Augument
- in_channel : channel of inputs
- size : image size
'''
class SimpleCNN(nn.Module) :

    def __init__(self, in_channel = 3, size = 256) :
        super(SimpleCNN, self).__init__()
        self.n_node = size // 2**2
        self.linear_in = 32 * self.n_node * self.n_node
        self.in_channel = in_channel
        self.classifier = nn.Sequential(
            nn.Conv2d(self.in_channel, 3, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(3, 8, 3, 1, 1),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(8, 16, 3, 1, 1),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(),
        )
        self.linear = nn.Linear(self.linear_in, 2)
    
    def forward(self, x) :
        out = self.classifier(x)
        out = out.view(-1, self.linear_in)
        out = self.linear(out)
        return out

# function for building classifier for fake image detection
# select classifier [cnn, vit] from option configurator
def build_classifier_from_opt(opt) :
    
    name = opt.classifier
    size = opt.size
    in_channel = 3

    if name == 'cnn' :
        model = SimpleCNN(in_channel = in_channel, size = size)
    elif name == 'vit' :
        model = ViT('B_16', pretrained=True, num_classes = 2, image_size = size)
    else :
        raise TypeError('no matching classifier')

    optimizer = torch.optim.SGD(model.parameters(), lr = opt.lr, momentum = 0.9)#, eps = 1e-7)
    setattr(model, 'optimizer', optimizer)
    criterion = torch.nn.CrossEntropyLoss()
    setattr(model, 'criterion', criterion)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(model.optimizer, gamma = 0.95)
    setattr(model, 'scheduler', scheduler)

    return model

# function for evaluation of classifier for fake image detection
# save evaluation results (in text file) in the 'opt.dst' folder
def eval_from_opt(opt) :

    device = torch.device(opt.device)

    root = './results'
    load_path = os.path.join(root, opt.dst, '{}_classifier/'.format(opt.classifier))

    model = build_classifier_from_opt(opt)
    param = torch.load(os.path.join(load_path, 'model.pt'), map_location = torch.device('cpu'))
    model.load_state_dict(param)
    model = model.to(device)

    print('Pre-trained weights are loaded.')

    eval_roots = [str(i) for i in opt.eval_root.split(',')]

    test_loaders = []
    logs = []

    for eval_root in eval_roots :

        test_loaders.append(ClassificationDataset(opt, eval_root, only_fake = False).get_loader())

    print('\n[Evaluation on the {} unseen-dataset]\n'.format(len(eval_roots)))
    logs.append('[Evaluation on the {} unseen-dataset]\n'.format(len(eval_roots)))

    eval_accs, eval_losses = evaluation(model, test_loaders, opt)

    for (name, acc, loss) in zip(eval_roots, eval_accs, eval_losses) :
        print('[Profile of {}]'.format(name))
        logs.append('[Profile of {}]'.format(name))
        print('accuracy & loss : {:.4f}% | {:.4f}\n'.format(acc, loss))
        logs.append('accuracy & loss : {:.4f}% | {:.4f}\n'.format(acc, loss))

    with open(os.path.join(load_path, 'eval_log.txt'), 'w') as r :
        for line in logs :
            r.write(line)
        r.close()

# function for training of classifier for fake image detection
# save the training parameters and training results (in text file) in the 'opt.dst' folder
def train_from_opt(opt) :

    device = torch.device(opt.device)

    root = './results'
    save_root = os.path.join(root, opt.dst, '{}_classifier/'.format(opt.classifier))

    os.makedirs(save_root, exist_ok=True)

    model = build_classifier_from_opt(opt)
    model = model.to(device)

    train_loader, valid_loader = TrainValidDataset(opt).get_loader()

    train_model, logs = train(model, train_loader, valid_loader, opt)

    with open(os.path.join(save_root, 'train_log.txt'), 'w') as r :
        for line in logs :
            r.write(line)
        r.close()
    torch.save(train_model.state_dict(), os.path.join(save_root, 'model.pt'))


def train(model, train_loader, valid_loader, opt) :

    device = torch.device(opt.device)

    best_valid_acc, best_valid_loss = 0., 0.

    logs = []

    for class_epoch in range(opt.class_epoch) :
        print('[{:02d}-th epoch]'.format(class_epoch+1))
        logs.append('[{:02d}-th epoch]'.format(class_epoch+1))

        train_acc, valid_acc = 0., 0.
        train_loss, valid_loss = 0., 0.

        n_train_sample, n_valid_sample = 0, 0

        model.train()

        for n, (data, target) in enumerate(tqdm(train_loader, desc="{:17s}".format('Training Procedure'), mininterval=0.0001)) :

            data, target = data.to(device), target.to(device)

            data = to_spectrum(data)

            model.optimizer.zero_grad()
            
            output = model(data)
            loss = model.criterion(output, target)
            loss.backward()

            model.optimizer.step()

            _, pred = torch.max(output, dim = 1)
            train_acc += torch.sum(pred == target.data).item()
            train_loss += loss.item()
            n_train_sample += data.shape[0]

        model.eval()

        for n, (data, target) in enumerate(tqdm(valid_loader, desc="{:17s}".format('Validation Procedure'), mininterval=0.0001)) :

            data, target = data.to(device), target.to(device)

            data = to_spectrum(data)

            with torch.no_grad() :

                output = model(data)
                loss = model.criterion(output, target)

            _, pred = torch.max(output, dim = 1)
            valid_acc += torch.sum(pred == target.data).item()
            valid_loss += loss.item()
            n_valid_sample += data.shape[0]

        avg_train_acc = (train_acc / n_train_sample * 100.)
        avg_valid_acc = (valid_acc / n_valid_sample * 100.)

        avg_train_loss = train_loss / n_train_sample
        avg_valid_loss = valid_loss / n_valid_sample

        if avg_valid_acc >= best_valid_acc :

            best_valid_acc = avg_valid_acc
            best_valid_loss = avg_valid_loss
            best_epoch = class_epoch + 1

        print('\nTrain Acc : {:.4f}% | Valid Acc : {:.4f}% | Train Loss : {:.6f} | valid_loss : {:.6f} | Learning Rate : {:.7f}\n'.format(avg_train_acc, avg_valid_acc, avg_train_loss, avg_valid_loss, model.optimizer.param_groups[0]['lr']))
        logs.append('\nTrain Acc : {:.4f}% | Valid Acc : {:.4f}% | Train Loss : {:.6f} | valid_loss : {:.6f} | Learning Rate : {:.7f}\n'.format(avg_train_acc, avg_valid_acc, avg_train_loss, avg_valid_loss, model.optimizer.param_groups[0]['lr']))
        model.scheduler.step()

    print('Best validation accuracy : {:.4f}% (from {:02d}-th epoch)\n'.format(best_valid_acc, best_epoch))
    logs.append('Best validation accuracy : {:.4f}% (from {:02d}-th epoch)\n'.format(best_valid_acc, best_epoch))

    return model, logs

@torch.no_grad()
def evaluation(model, dataloaders, opt) :

    device = torch.device(opt.device)

    results_acc, results_loss = [], []

    for n, loader in enumerate(dataloaders) :

        n_test_sample = 0
        test_acc, test_loss = 0., 0.

        model.eval()

        for n, (data, target) in enumerate(tqdm(loader, desc="{:17s}".format('Evaluation Procedure [{}/{}]'.format(n+1, len(dataloaders))), mininterval=0.0001)) :

            data, target = data.to(device), target.to(device)

            data = to_spectrum(data)


            with torch.no_grad() :

                output = model(data)
                loss = model.criterion(output, target)

            _, pred = torch.max(output, dim = 1)
            test_acc += torch.sum(pred == target.data).item()
            test_loss += loss.item()
            n_test_sample += data.shape[0]

        avg_test_acc = (test_acc / n_test_sample * 100.)
        avg_test_loss = (test_loss / n_test_sample * 100.)

        results_acc.append(avg_test_acc)
        results_loss.append(avg_test_loss)

    return results_acc, results_loss

def get_magnitude(img) :
    fft = torch.fft.fft2(img, dim = [-1, -2])
    fft = torch.fft.fftshift(fft, dim = [-1, -2])
    mag = torch.log1p(torch.abs(fft))
    v_min, v_max = torch.amin(mag, dim = (1, 2, 3)).view(-1, 1, 1, 1), torch.amax(mag, dim = (1, 2, 3)).view(-1, 1, 1, 1)
    mag_normed = (mag - v_min) / (v_max - v_min)
    return mag_normed

def to_spectrum(img) :

    mag = get_magnitude(img)
    mag = mag * 2. - 1.
    return mag
