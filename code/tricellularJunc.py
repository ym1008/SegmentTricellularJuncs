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
from refinenet_4cascade import RefineNet4Cascade, RefineNet4CascadePoolingImproved


def get_args(): 
	parser = argparse.ArgumentParser()
	parser.add_argument('--root', default='../Data_experiment/', type=str)
	parser.add_argument('--modelDir', default='../Experiments', type=str)
	parser.add_argument('--imName', default='1.tif', type=str)
	parser.add_argument('--imSize', default=1024, type=int)  # 64, 128, or 256.
	parser.add_argument('--feats', default=128, type=int)  #multiple of filters to use
	parser.add_argument('--freeze', action='store_false')	
	parser.add_argument('--update', default=2000, type=int) 

	return parser.parse_args()



def run(net, x, opts):

	useCUDA = torch.cuda.is_available()

	if useCUDA:
		net.cuda()
	
	net.eval()
	x = Variable(x).cuda() if useCUDA else Variable(x)
	output = net(x)
	output = interpolate(output, scale_factor=(4,4), mode='bilinear')
		
	return output.data


	

if __name__=='__main__':
	opts = get_args()

	net = RefineNet4Cascade(input_shape=(3,opts.imSize), num_classes=1, features=opts.feats, freeze_resnet=opts.freeze)
	net.load_state_dict(torch.load(join(opts.modelDir, 'params_u' + str(opts.update))))

	xTransform = transforms.Compose([transforms.ToPILImage(), transforms.Grayscale(3),\
		transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

	
	im = Image.open(join(opts.root, opts.imName + '.tif'))
	img = np.array(im)
	img = img[:,:,np.newaxis]

	x = xTransform(img)
	x = x.unsqueeze(0)
	y = run(net, x, opts)

	np.save(join(opts.root, 'tricell_'+opts.imName + '.npy'), y.cpu().numpy())
	save_image(y, join(opts.root,'tricell_'+ opts.imName + '.png'), normalize=True)



