from model import AlexNet
from training import Train
from utils.dataloaderutils import  Dataloader 
import torch
from torch import nn
import torch.optim as optim
classes = 10
epochs = 20
batch_size = 64
lr = 0.005


device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
model = AlexNet(classes)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, weight_decay=0.005, lr=lr)

train_data, valid_data = Dataloader.get_train_data_loader(data_dir="./cifar-10-batches-py", batch_size=64, augmentation=False, random_seed=1)
test_data = Dataloader.get_test_loader(data_dir="./cifar-10-batches-py", batch_size=64)

total_step = len(train_data)
Train.train_alex(epochs, train_data, valid_data, device, model, criterion, optimizer, total_step)
