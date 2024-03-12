from glob import glob
import os
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from torchvision import transforms
import torch

'''
RealFakeDataset : Dataset for training STIG model.
                  Consisting of the real and generated image pair. -> [torch.Tensor, torch.Tensor]
                  Call the function 'get_dataset_from_option' for bring the training dataset.
'''

class RealFakeDataset(Dataset) :
    def __init__(self, root_real, root_fake, opt) :

        self.set_options(opt)
        self.transform = transforms.Compose([
            transforms.Resize((self.size, self.size)),
            transforms.ToTensor(),
        ])

        self.files_real = glob(os.path.join(root_real, '*.png'))
        self.files_fake = glob(os.path.join(root_fake, '*.png'))
        
        # For loading celeba images
        self.files_real.extend(glob(os.path.join(root_real, '*.jpg')))
        self.files_fake.extend(glob(os.path.join(root_fake, '*.jpg')))

    def __getitem__(self, index):
        
        self.real = Image.open(self.files_real[index])
        self.fake = Image.open(self.files_fake[index])

        self.real = self.transform(self.real)
        self.fake = self.transform(self.fake)

        return {'clean' : self.real, 'noise' : self.fake}

    def __len__(self) :
        return max(len(self.files_fake), len(self.files_real))

    def set_options(self, opt) :
        self.batch_size = opt.batch_size
        self.size = opt.size

    def get_loader(self) :
        return DataLoader(self, self.batch_size, shuffle = False)

'''
TrainValidDataset : Dataset for training the classifier for fake image detection.
                    Consisting of the real or generated image batches. -> torch.Tensor
'''

class TrainValidDataset(Dataset):
    def __init__(self, opt) :

        self.set_options(opt)
        self.transform = transforms.Compose([
            transforms.Resize((self.size, self.size)),
            transforms.ToTensor(),
        ])

        self.fake_path = os.path.join(self.fake_root)
        self.real_path = os.path.join(self.real_root)

        self.fake_list = glob(os.path.join(self.fake_path, '*.png'))
        self.fake_list.extend(glob(os.path.join(self.fake_path, '*.jpg')))
        self.real_list = glob(os.path.join(self.real_path, '*.png'))
        self.real_list.extend(glob(os.path.join(self.real_path, '*.jpg')))

        self.img_list = self.fake_list + self.real_list
        self.label_list = [0] * len(self.fake_list) + [1] * len(self.real_list)

    def __getitem__(self, index):
        
        self.image = Image.open(self.img_list[index])
        self.image = self.transform(self.image)
        self.label = self.label_list[index]

        return self.image, self.label

    def __len__(self) :
        return len(self.img_list)

    def set_options(self, opt) :
        
        self.batch_size = opt.class_batch_size
        self.size = opt.size

        root = './results'

        self.real_root = os.path.join(root, opt.dst, 'eval', 'clean')
        if not os.path.exists(self.real_root) :
            raise ValueError('Non-exist foler!')
        self.fake_root = os.path.join(root, opt.dst, 'eval', 'noise')
        if not os.path.exists(self.fake_root) :
            raise ValueError('Non-exist foler!')

    def get_loader(self) :
        train_dataset, validation_dataset = random_split(self, [int(self.__len__() * 0.8), (self.__len__() - int(self.__len__() * 0.8))], generator=torch.Generator().manual_seed(42))
        return DataLoader(train_dataset, self.batch_size, shuffle = True), DataLoader(validation_dataset, self.batch_size, shuffle = True)

'''
ClassificationDataset : Dataset for evaluation the classifier for fake image detection.
                        Consisting of the real or generated image batches. -> torch.Tensor
                        You can choose the path of generated images and real images for evaluation the classifier.
'''

class ClassificationDataset(Dataset):

    def __init__(self, opt, eval_root, only_fake = True) :

        self.eval_root = eval_root
        self.only_fake = only_fake
        self.init_options(opt)
        self.set_options()
        self.transform = transforms.Compose([
            transforms.Resize((self.size, self.size)),
            transforms.ToTensor(),
        ])

    def __getitem__(self, index):
        
        self.image = Image.open(self.img_list[index])
        self.image = self.transform(self.image)
        self.label = self.label_list[index]

        return self.image, self.label

    def __len__(self) :

        return len(self.img_list)

    def init_options(self, opt) :

        self.opt = opt
        self.batch_size = opt.class_batch_size
        self.size = opt.size
        self.root = './results'

    def set_options(self) :

        self.fake_path = os.path.join(self.eval_root)
        if not os.path.exists(self.fake_path) :
            raise ValueError('Non-exist foler! ({})'.format(self.fake_path))

        self.fake_list = glob(os.path.join(self.fake_path, '*.png'))
        self.fake_list.extend(glob(os.path.join(self.fake_path, '*.jpg')))

        self.img_list = self.fake_list
        self.label_list = [0] * len(self.fake_list)

        if not self.only_fake :
            self.real_path = os.path.join(self.root, self.opt.dst, 'eval', 'clean')
            if not os.path.exists(self.real_path) :
                raise ValueError('Non-exist foler! ({})'.format(self.real_path))

            self.real_list = glob(os.path.join(self.real_path, '*.png'))
            self.real_list.extend(glob(os.path.join(self.real_path, '*.jpg')))

            self.img_list = self.img_list + self.real_list
            self.label_list = self.label_list + [1] * len(self.real_list)

    def get_loader(self) :

        dataloader = DataLoader(self, self.batch_size, shuffle = True)
        return dataloader
    
class InferenceDataset(Dataset):

    def __init__(self, opt) :

        self.init_options(opt)
        self.set_options()
        self.transform = transforms.Compose([
            transforms.Resize((self.size, self.size)),
            transforms.ToTensor(),
        ])

    def __getitem__(self, index):
        
        self.image = Image.open(self.img_list[index])
        self.image = self.transform(self.image)
        self.image = self.image.repeat(3, 1, 1)
        print(self.image.shape)

        return self.image

    def __len__(self) :

        return len(self.img_list)

    def init_options(self, opt) :

        self.opt = opt
        self.batch_size = opt.class_batch_size
        self.size = opt.size
        self.inference_data_root = opt.inference_data

    def set_options(self) :

        self.image_path = os.path.join(self.inference_data_root)
        if not os.path.exists(self.inference_data_root) :
            raise ValueError('Non-exist foler! ({})'.format(self.inference_data_root))

        self.img_list = glob(os.path.join(self.image_path, '*.png'))
        self.img_list.extend(glob(os.path.join(self.image_path, '*.jpg')))

    def get_loader(self) :

        dataloader = DataLoader(self, self.batch_size, shuffle = True)
        return dataloader

def get_dataset_from_option(opt) :
    
    data_name = opt.data

    data_root = os.path.join('./datasets/', data_name)
    
    train_dataset = RealFakeDataset(os.path.join(data_root, 'real'), os.path.join(data_root, 'fake'), opt)
    train_dataloader = train_dataset.get_loader()
    
    return train_dataloader, train_dataloader

def get_inference_dataset_from_option(opt) :

    dataset = InferenceDataset(opt)
    loader = dataset.get_loader()

    return loader