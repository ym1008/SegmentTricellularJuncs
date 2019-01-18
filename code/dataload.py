import os
import numpy as np
from os.path import join
from torch.utils import data
from torchvision import transforms, datasets

class ConfocalData(data.Dataset):
    """
    Args:
        root (string): Root directory of where numpy arrays (dataset) exists.
        train (bool, optional): If True, creates dataset from training set with no labels needed, 
                    otherwise creates both input and label dataset.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
    """


    def __init__(self, root, train=True, transform=None):
        
        self.root = os.path.expanduser(root)
        self.train = train  # training set or test set
        self.transform=transform
       
        self.loadData()


    def loadData(self):

        #load data
        if self.train: 
            self.data = np.load(join(self.root, 'xtrain.npy'), mmap_mode='r') 
            self.labels = np.load(join(self.root, 'ytrain.npy'), mmap_mode='r') 
            self.labels = self.labels.astype(int)
            print 'Training data and labels: ', np.shape(self.data), np.shape(self.labels)
        else: 
            self.data = np.load(join(self.root, 'xtest.npy'), mmap_mode='r')
            self.labels = np.load(join(self.root, 'ytest.npy'), mmap_mode='r')
            self.labels = self.labels.astype(int)
            print 'Testing data and labels: ', np.shape(self.data), np.shape(self.labels)


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        img = self.data[index]
        img = np.uint8(img*255) # this was for omniglot data which was binary black and white... Need to think what to do with 1 channel data. 

        if self.transform:
            img = self.transform(img)

        return img, self.labels[index]
       

    def __len__(self):
        return len(self.data)



