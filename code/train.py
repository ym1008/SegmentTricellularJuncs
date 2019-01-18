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
from torch.nn.functional import binary_cross_entropy as bce
import torch.nn.functional as F

sys.path.append('../refinenet/')
from refinenet_4cascade import RefineNet4Cascade



def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--root', default='../data/', type=str)
	parser.add_argument('--outDir', default='../Experiments', type=str)
	parser.add_argument('--batchSize', default=64, type=int)
	parser.add_argument('--maxEpochs', default=150, type=int)
	parser.add_argument('--lr', default=1e-3, type=float)
	parser.add_argument('--imSize', default=256, type=int)  # divisible by 32.
	parser.add_argument('--feats', default=256, type=int)  #multiple of filters to use
	parser.add_argument('--commit', required=True, type=str)
	parser.add_argument('--gpuNo', default=0, type=int)

	return parser.parse_args()


def train_mode(gen, dis, trainLoader, useNoise=False, betas=[0.5,0.5,0.999], c=0.01, k = 1, lam = 10, WGAN=False, WGP=False, clip=True):
	####### Define optimizer #######
	genOptimizer = optim.Adam(gen.parameters(), lr=opts.lr, betas=(betas[0], betas[2]))
	disOptimizer = optim.Adam(dis.parameters(), lr=opts.lr, betas=(betas[1], betas[2]))

	if gen.useCUDA:
		torch.cuda.set_device(opts.gpuNo)
		gen.cuda()
		dis.cuda()
	
	####### Create a new folder to save results and model info #######
	exDir = make_new_folder(opts.outDir)
	print 'Outputs will be saved to:',exDir
	save_input_args(exDir, opts)

	#noise level
	noiseSigma = np.logspace(np.log2(0.5), np.log2(0.001), opts.maxEpochs, base=2)

	####### Start Training #######
	losses = {'gen':[], 'dis':[], 'GP':[]} if WGP else {'gen':[], 'dis':[]}
	       
	for e in range(opts.maxEpochs):
		dis.train()
		gen.train()

		epochLoss_gen = 0
		epochLoss_dis = 0

		noiseLevel = float(noiseSigma[e])

		T = time()
		for i, data in enumerate(trainLoader, 0):

			for _ in range(k):

				# add a small amount of corruption to the data
				xReal = Variable(data)
				xReal = xReal.cuda() if gen.useCUDA else xReal
				xReal = corrupt(xReal, noiseLevel) if useNoise else xReal

				noSamples = xReal.size(0)
				xFake = gen.sample_x(noSamples)
				xFake = corrupt(xFake, noiseLevel) if useNoise else xFake


				####### Calculate discriminator loss #######
				pReal_D = dis.forward(xReal)
				pFake_D = dis.forward(xFake.detach())
				
				real = dis.ones(xReal.size(0))
				fake = dis.zeros(xFake.size(0))

				if WGP:
					eps = torch.rand(noSamples, 1, 1, 1)
					eps.expand_as(xReal)
					eps = eps.cuda() if gen.useCUDA else eps

					xinterp = eps * xReal.data  + (1 - eps) * xFake.data
					xinterp = xinterp.cuda() if gen.useCUDA else xinterp
					xinterp = Variable(xinterp, requires_grad=True)

					pInterp_D = dis.forward(xinterp)
					gradInterp = grad(outputs=pInterp_D, inputs=xinterp, 
									  grad_outputs=torch.ones(pInterp_D.size()).cuda() if gen.useCUDA else torch.ones(pInterp_D.size()), 
									  create_graph=True, retain_graph=True)[0]
					
					gradInterp = gradInterp.view(noSamples, -1)
					normGrad = gradInterp.norm(2, dim=1)
					gpLoss = lam * ((normGrad - 1)**2).mean() 
					disLoss = pFake_D.mean() - pReal_D.mean() 

					losses['GP'].append(gpLoss.item())
					losses['dis'].append(disLoss.item())

					disLoss = disLoss + gpLoss


				else:				
					if WGAN: 
						disLoss = pFake_D.mean() - pReal_D.mean() 
					else:
						disLoss =  opts.pi * F.binary_cross_entropy(pReal_D, real) + \
								(1 - opts.pi) * F.binary_cross_entropy(pFake_D, fake) 

					losses['dis'].append(disLoss.item())	
					


				####### Do DIS updates ####### 
				disOptimizer.zero_grad()
				disLoss.backward() 
				disOptimizer.step()


				#### clip DIS weights #### YM
				if WGAN and clip:
					for p in dis.parameters():
						p.data.clamp_(-c, c)



			####### Calculate generator loss #######
			xFake_ = gen.sample_x(noSamples)  
			if useNoise:
				xFake_ = corrupt(xFake_, noiseLevel) #add a little noise
			pFake_G = dis.forward(xFake_)
			

			if WGAN or WGP:
				genLoss = - pFake_G.mean()
			else:
				genLoss = F.binary_cross_entropy(pFake_G, real)


			####### Do GEN updates #######   
			genOptimizer.zero_grad()
			genLoss.backward()
			genOptimizer.step()

			losses['gen'].append(genLoss.item())

			####### Print info #######
			if i%100==1:
				print '[%d, %d] gen: %.5f, dis: %.5f, time: %.2f' \
					% (e, i, genLoss.item(), disLoss.item(), time()-T)

		####### Tests #######
		gen.eval()
		print 'Outputs will be saved to:',exDir
		#save some samples
		samples = gen.sample_x(48)
		save_image(samples.data, join(exDir,'epoch'+str(e)+'.png'), normalize=True)

		#plot
		plot_losses(losses, exDir, epochs=e+1)

		####### Save params #######
		gen.save_params(exDir, e)
		dis.save_params(exDir, e)

	return gen, dis



if __name__=='__main__':
	opts = get_args()

	####### Data set #######

	IM_SIZE = opts.imSize

	if opts.dataset == 'shoes':

		transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((IM_SIZE, IM_SIZE)), transforms.RandomHorizontalFlip(),\
		 transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

		trainDataset = SHOES(root=opts.root, filename='Shoes_train', train=True, transform=transform)
		trainLoader = torch.utils.data.DataLoader(trainDataset, batch_size=opts.batchSize, shuffle=True)
	
		gen = GEN((IM_SIZE, 3), nz=opts.nz, fSize=opts.fSize, nlayers=opts.conv)
		dis = DIS((IM_SIZE, 3), fSize=opts.fSize, nlayers=opts.conv, WGAN=opts.WGAN, WGP=opts.WGP, nolinear=opts.nolinear)


	elif opts.dataset == 'omniglot':
		transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((IM_SIZE, IM_SIZE)),\
		 transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

		trainDataset = OMNIGLOT(root=opts.root, filename=opts.experiment, train=True, transform=transform)
		trainLoader = torch.utils.data.DataLoader(trainDataset, batch_size=opts.batchSize, shuffle=True)

		gen = GEN((IM_SIZE, 1), nz=opts.nz, fSize=opts.fSize, nlayers=opts.conv)
		dis = DIS((IM_SIZE, 1), fSize=opts.fSize, nlayers=opts.conv, WGAN=opts.WGAN, WGP=opts.WGP, nolinear=opts.nolinear)
	

	print 'Data loaders ready.'


	###### Create model ######

	gen, dis = train_mode(gen, dis, trainLoader, useNoise=opts.useNoise, 
						  betas=[opts.beta1D, opts.beta1G, opts.beta2], c=opts.c, k = opts.k, lam=opts.lam,
						  WGAN=opts.WGAN, WGP= opts.WGP, clip=opts.clip)

