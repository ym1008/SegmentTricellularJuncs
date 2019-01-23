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



    def __init__(self, root, filename, xtransform=None, ytransform=None):
        
        self.root = os.path.expanduser(root)
        self.filename = filename
        self.xtransform=xtransform
        self.ytransform=ytransform

        self.loadData()


    def loadData(self):

        self.data = np.load(join(self.root, 'x'+self.filename+'.npy'), mmap_mode='r') 
        self.labels = np.load(join(self.root, 'y'+self.filename+'.npy'), mmap_mode='r') 
        print 'Data and label shape: ', np.shape(self.data), np.shape(self.labels)
        

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        img = self.data[index]
        labels = self.labels[index]

        if self.xtransform:
            img = self.xtransform(img)

        if self.ytransform:
            labels = self.ytransform(labels)

        return img, labels
       

    def __len__(self):
        return len(self.data)



