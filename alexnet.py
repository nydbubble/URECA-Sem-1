import pandas as pd
from torch import np # Torch wrapper for Numpy

from PIL import Image

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import nn
import torch.optim as optim
from torch.autograd import Variable
#from sklearn.preprocessing import MultiLabelBinarizer

TRAIN_DATA = 'Anno/train.csv'

use_gpu = torch.cuda.is_available()
#use_gpu = False

"""attributefile = open("Anno/list_attr_cloth.txt", 'r')
attributes = [line.split() for line in attributefile.readlines()] #attributes starts from index 3
del attributes[0]
del attributes[0]""" # TODO: implement MLB for attribute prediction

#Preparing dataset
class DeepFashionDataset(Dataset):
    """Dataset wrapping images and target labels for DeepFashion Dataset

    Arguments:
        A CSV file path
        PIL transforms
    """

    def __init__(self, csv_path, transform=None):
    
        tmp_df = pd.read_csv(csv_path)
        
        #self.mlb = MultiLabelBinarizer(classes = attributes) # TODO: implement MLB for attribute prediction
        self.transform = transform

        self.X_train = tmp_df['image_name']
        #self.y_train = self.mlb.fit_transform(tmp_df['attribute_labels'].str.split()).astype(np.float32)
        self.y_train = np.array(list(tmp_df['attribute_labels'].str.split())).astype(np.float32)

    def __getitem__(self, index):
        img = Image.open(self.X_train[index])
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        
        label = torch.from_numpy(self.y_train[index])
        return img, label

    def __len__(self):
        return len(self.X_train.index)


transformations = transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

dset_train = DeepFashionDataset(TRAIN_DATA,transformations)


if use_gpu:
  train_loader = DataLoader(dset_train, batch_size = 256, shuffle = True, num_workers = 1, pin_memory = 1)
else:
  train_loader = DataLoader(dset_train,
                          batch_size=256,
                          shuffle=True,
                          num_workers=4, # 1 for CUDA
                         )

#preparing custom alexnet architechture
class AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        x = self.sigmoid(x)
        return x


def alexnet(pretrained=False, **kwargs):
    """AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = AlexNet(**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['alexnet']))
    return model


model = alexnet()

print(model)

criterion = torch.nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001,weight_decay=5e-4) # Changed to Fashion Forecast implementation

if use_gpu:   
    model.cuda()
    criterion.cuda()

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if use_gpu:
          data, target = Variable(data.cuda(async = True)), Variable(target.cuda(async=True))
        else:
          data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        #print(output)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))

print("Begin training")
for epoch in range(1, 2):
    train(epoch)

print("Training finished")
