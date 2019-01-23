# DCGAN to be trained on UT-Zap-50K SHOES with no labels and have params stored
import sys
import os
import torch
import argparse
import matplotlib 
import scipy.io as sp
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
	parser.add_argument('--batchSize', default=32, type=int)
	parser.add_argument('--maxEpochs', default=30, type=int)
	parser.add_argument('--lr', default=1e-4, type=float)
	parser.add_argument('--smooth', default=0.0, type=float)
	parser.add_argument('--imSize', default=64, type=int)  # 64, 128, or 256.
	parser.add_argument('--feats', default=128, type=int)  #multiple of filters to use
	parser.add_argument('--commit', required=True, type=str)
	parser.add_argument('--gpuNo', default=0, type=int)
	parser.add_argument('--freeze', action='store_false')

	return parser.parse_args()


def make_folder(exDir):
	if not os.path.isdir(exDir):
		os.mkdir(exDir)
	return exDir

def train(net, trainLoader, testLoader, opts):

	####### Define optimizer #######
	opt = optim.Adam(net.parameters(), lr=opts.lr)
	smooth = opts.smooth;

	useCUDA = torch.cuda.is_available()

	if useCUDA:
		torch.cuda.set_device(opts.gpuNo)
		net.cuda()
	
	####### Create a new folder to save results and model info #######
	exDir = make_new_folder(opts.outDir)
	print 'Outputs will be saved to:', exDir
	save_input_args(exDir, opts)

	exDir_params = make_folder(join(exDir, 'params'))
	exDir_train = make_folder(join(exDir, 'plot_train'))
	exDir_eval = make_folder(join(exDir, 'plot_eval'))
	

	####### Start Training #######
	losses = {'train':[], 'test':[]} 
	factor = {'train':1, 'test':100}

	for e in range(opts.maxEpochs):
		T = time()
		for i, (x,y) in enumerate(trainLoader, 0):
			net.train()
			x = Variable(x).cuda() if useCUDA else Variable(x)
			y = y.cuda() if useCUDA else y

			output = net(x)
			
			# calculate the loss. 
			loss = 1-2 * torch.sum(output * y, (1,2,3)) / (torch.sum(output**2 + y**2, (1, 2, 3)) + smooth)
			loss = torch.mean(loss)

			losses['train'].append(loss.item())	
			opt.zero_grad()
			loss.backward() 
			opt.step()


			####### Print info #######
			if i%100==0:
				print '[%d, %d] train: %.5f, time: %.2f' \
					% (e, i, loss.item(), time()-T)

				plot_losses(losses, exDir, factor=factor)

			if (i + e*len(trainLoader)) % 100 == 0:
				u = i + e*len(trainLoader)

				save_image(output, join(exDir_train,'strain_u'+str(u)+'.png'), normalize=True, scale_each=True)
				save_image(x, join(exDir_train,'xtrain_u'+str(u)+'.png'), normalize=True, scale_each=True)
				save_image(y, join(exDir_train,'ytrain_u'+str(u)+'.png'), normalize=True, scale_each=True)
		
				# Test
				net.eval()
				testloss = 0
				plotx = []
				ploty = []
				plots = []
				n = 0
				for j, (x,y) in enumerate(testLoader, 0):

					x = Variable(x).cuda() if useCUDA else Variable(x)
					y = y.cuda() if useCUDA else y

					output = net(x)

					if j == 0 or j == len(testLoader) -1 :
						plotx.append(x)
						ploty.append(y)
						plots.append(output)
					
					# calculate the loss. 
					loss = 1-2 * torch.sum(output * y, (1,2,3)) / (torch.sum(output**2 + y**2, (1, 2, 3)))
					testloss = testloss + torch.sum(loss).item()
					n = n + loss.shape[0]

				testloss = testloss / n
				losses['test'].append(testloss)	
				
				plot_losses(losses, exDir, factor=factor)

				# save the segmentations of the test images... train images too? Few examples? 
				save_image(torch.cat(plots,0), join(exDir_eval,'stest_u'+str(u)+'.png'), normalize=True, scale_each=True)
				save_image(torch.cat(ploty,0), join(exDir_eval,'ytest.png'), normalize=True, scale_each=True)
				save_image(torch.cat(plotx,0), join(exDir_eval,'xtest.png'), normalize=True, scale_each=True)
			

				####### Save params #######
				print 'saving params to: ', exDir_params
				torch.save(net.state_dict(), join(exDir_params, 'params_u'+str(u)))

				# Save losses
				sp.savemat(join(exDir, 'losses.mat'), mdict = {'train': np.array(losses['train'])})
				


if __name__=='__main__':
	opts = get_args()

	####### Data set #######

	xtrainTransform = transforms.Compose([transforms.ToPILImage(), transforms.Grayscale(3),\
		transforms.ColorJitter(brightness=0.5, contrast=0.5), transforms.ToTensor(),\
		transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
	
	xtestTransform = transforms.Compose([transforms.ToPILImage(), transforms.Grayscale(3),\
		transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

	yTransform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])


	trainDataset = ConfocalData(root=opts.root, filename='Train_'+str(opts.imSize), xtransform=xtrainTransform, ytransform = yTransform)
	trainLoader = torch.utils.data.DataLoader(trainDataset, batch_size=opts.batchSize, shuffle=True)
	
	testDataset = ConfocalData(root=opts.root, filename='Val_'+str(opts.imSize), xtransform=xtestTransform, ytransform=yTransform)
	testLoader = torch.utils.data.DataLoader(testDataset, batch_size=opts.batchSize, shuffle=False)

	print 'Data loaders ready.'

	
	###### Create model ######
	
	net = RefineNet4Cascade(input_shape=(3,opts.imSize), num_classes=1, features=opts.feats, freeze_resnet=opts.freeze)

	train(net, trainLoader, testLoader, opts)



