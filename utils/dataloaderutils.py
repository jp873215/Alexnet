from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import math
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler

class Dataloader():
    def get_train_data_loader(data_dir, batch_size, augmentation, random_seed, shuffle = True, valid_size = 0.1):
        # for each channel -> R G B we have different mean and std values
        # not some random values, derieved from avg pixel intensity
        # (input - mean) / std
        normalize = transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010],
        )

        valid_transform = transforms.Compose([
            transforms.Resize((277,277)),
            transforms.ToTensor(),
            normalize
        ])

        if augmentation:
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding = 4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ])
        
        else:
            train_transform = transforms.Compose([
                transforms.Resize((227,227)),
                transforms.ToTensor(),
                normalize
            ])
        
        train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform= train_transform)

        valid_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=valid_transform)

        num_train = len(train_dataset)
        split_idx = list(range(num_train))
        split = int(math.floor(valid_size * num_train)) 

        if (shuffle):
            np.random.seed(random_seed)
            # Shuffle does inplace shuffling, whereas permutation doesn't do it in - place and returns new array 
            np.random.shuffle(split_idx)

        train, valid = split_idx[split:], split_idx[:split]
        train_sample = SubsetRandomSampler(train)
        valid_sample = SubsetRandomSampler(valid)


        train_loader = DataLoader(
            train_dataset, batch_size = batch_size, sampler = train_sample
        )
        valid_loader = DataLoader(
            valid_dataset, batch_size = batch_size, sampler = valid_sample
        )
        return (train_loader, valid_loader)
    
    def get_test_loader(data_dir, batch_size, shuffle=True):
        normalize = transforms.Normalize(
            mean = [0.485, 0.456, 0.406],
            std = [0.229, 0.224, 0.225]
        )

        transform = transforms.Compose([
            transforms.Resize((277,277)),
            transforms.ToTensor(),
            normalize
        ])

        data = datasets.CIFAR100(
            root = data_dir, download=True, train= False, transform=transforms
        )

        data_loader = DataLoader(
            data, batch_size = batch_size, shuffle = shuffle
        )

        return data_loader
    
