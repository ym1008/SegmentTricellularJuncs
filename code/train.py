# DCGAN to be trained on UT-Zap-50K SHOES with no labels and have params stored
import sys
import os
import torch
import argparse
import matplotlib 
import numpy as np

from time import time
from PIL import Image
from os.path import join
from torchvision import transforms
from torchvision.utils import save_image

matplotlib.use('Agg')
from matplotlib import pyplot as plt

from torch import optim
from torch.autograd import Variable, grad
from torch.nn.functional import interpolate
import torch.nn.functional as F

sys.path.append('../refinenet/')
from refinenet_4cascade import RefineNet4Cascade
from utils import make_new_folder, save_input_args, plot_losses
from dataload import ConfocalData


def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--root', default='../data/', type=str)
	parser.add_argument('--outDir', default='../Experiments', type=str)
	parser.add_argument('--batchSize', default=4, type=int)
	parser.add_argument('--maxEpochs', default=150, type=int)
	parser.add_argument('--lr', default=1e-5, type=float)
	parser.add_argument('--imSize', default=224, type=int)  # divisible by 32.
	parser.add_argument('--feats', default=256, type=int)  #multiple of filters to use
	parser.add_argument('--commit', required=True, type=str)
	parser.add_argument('--gpuNo', default=0, type=int)

	return parser.parse_args()


def train(net, trainLoader, testLoader, opts):

	####### Define optimizer #######
	opt = optim.Adam(net.parameters(), lr=opts.lr)

	useCUDA = torch.cuda.is_available()

	if useCUDA:
		torch.cuda.set_device(opts.gpuNo)
		net.cuda()

	
	####### Create a new folder to save results and model info #######
	exDir = make_new_folder(opts.outDir)
	print 'Outputs will be saved to:', exDir
	save_input_args(exDir, opts)


	####### Start Training #######
	losses = {'train':[], 'test':[]} 

	for e in range(opts.maxEpochs):
		net.train()
		T = time()
		for i, (x,y) in enumerate(trainLoader, 0):

			bs, ncrops, c, h, w = x.size()
			x = x.view(-1, c, h, w)

			bs, ncrops, c, h, w = y.size()
			y = y.view(-1, c, h, w)

			x = Variable(x).cuda() if useCUDA else Variable(x)
			y = y.cuda() if useCUDA else y

			output = net(x)
			output = interpolate(output, scale_factor=(4,4))
		
			# calculate the loss. 
			loss = 2 * torch.sum(output * y, (1,2,3)) / (torch.sum(output**2 + y**2, (1, 2, 3)))
			loss = torch.mean(loss)

			losses['train'].append(loss.item())	
			opt.zero_grad()
			loss.backward() 
			opt.step()

			####### Print info #######
			if i%20==1:
				print '[%d, %d] train: %.5f, time: %.2f' \
					% (e, i, loss.item(), time()-T)


		save_image(output, join(exDir,'rec_train_e'+str(e)+'.png'), normalize=True, nrow = 4)
		if e == 0:
			save_image(y, join(exDir,'orig_train.png'), normalize=True, nrow = 4)


		# Test
		net.eval()
		T = time()
		testloss = 0
		n = 0
		for i, (x,y) in enumerate(testLoader, 0):

			x = Variable(x).cuda() if useCUDA else Variable(x)
			y = y.cuda() if useCUDA else y

			output = net(x)
			output = interpolate(output, scale_factor=(4,4))
		
			# calculate the loss. 
			loss = 2 * torch.sum(output * y, (1,2,3)) / (torch.sum(output**2 + y**2, (1, 2, 3)))
			testloss = testloss + torch.sum(loss).item()
			n = n + loss.shape[0]

		testloss = testloss / n
		losses['test'].append(testloss)	
		
		plot_losses(losses, exDir, epochs=e+1)

		# save the segmentations of the test images... train images too? Few examples? 
		save_image(output, join(exDir,'rec_test_e'+str(e)+'.png'), normalize=True, nrow = 4)
		if e == 0:
			save_image(y, join(exDir,'orig_test.png'), normalize=True, nrow = 4)
	

		####### Save params #######
		print 'saving params to: ', exDir
		torch.save(net.state_dict(), join(exDir, 'params_'+str(e)))
		



if __name__=='__main__':
	opts = get_args()

	####### Data set #######

	xtrainTransform = transforms.Compose([transforms.ToPILImage(), transforms.Grayscale(3),\
		transforms.ColorJitter(brightness=0.5, contrast=0.5), transforms.ToTensor(),\
		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), transforms.ToPILImage(),\
		transforms.TenCrop(opts.imSize), transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops]))])

	ytrainTransform = transforms.Compose([transforms.ToPILImage(), transforms.TenCrop(opts.imSize),\
	 	transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops]))])

	trainDataset = ConfocalData(root=opts.root, train=True, xtransform=xtrainTransform, ytransform = ytrainTransform)
	trainLoader = torch.utils.data.DataLoader(trainDataset, batch_size=opts.batchSize, shuffle=True)


	xtestTransform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((opts.imSize, opts.imSize)),\
		transforms.Grayscale(3), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

	ytestTransform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((opts.imSize, opts.imSize)), transforms.ToTensor()])
	
	testDataset = ConfocalData(root=opts.root, train=False, xtransform=xtestTransform, ytransform=ytestTransform)
	testLoader = torch.utils.data.DataLoader(testDataset, batch_size=opts.batchSize, shuffle=False)

	print 'Data loaders ready.'

	
	###### Create model ######
	
	net = RefineNet4Cascade(input_shape=(3,opts.imSize), num_classes=1, features=opts.feats, freeze_resnet=True)

	train(net, trainLoader, testLoader, opts)



