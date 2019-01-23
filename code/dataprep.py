from PIL import Image
import numpy as np
import os
import argparse

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--imSize', default=256, type=int)

	return parser.parse_args()


def chopImage(fname, crops, label, train):
	img = Image.open(fname)
	img = np.array(img)
	img = img[:,:,1] # green channel
		
	splitrows = np.split(img, crops, 0) # split into 16 rows (height = 1024/16 = 64 pixels)
		
	allImg = []
	for eachrow in splitrows: 
		splitcols = np.split(eachrow, crops, 1) # split each row into 16 (width = 1024/16 = 64 pixels)
			
		for subimg in splitcols:
			subimg = subimg[:,:, np.newaxis]

			if label:
				subimg[subimg>100] = 255
				subimg[subimg<101] = 0

			if train:
				allImg.append(subimg)
				allImg.append(np.rot90(subimg,1))
				allImg.append(np.rot90(subimg,2))
				allImg.append(np.rot90(subimg,3))
				
				allImg.append(np.fliplr(subimg))
				allImg.append(np.fliplr(np.rot90(subimg,1)))
				allImg.append(np.fliplr(np.rot90(subimg,2)))
				allImg.append(np.fliplr(np.rot90(subimg,3)))
			else:
				allImg.append(subimg)

	return np.array(allImg)


if __name__=='__main__':
	
	opts = get_args()

	directories = os.listdir('TrainingData/')
	crops = 1024/opts.imSize
	
	allX = []
	allY = []
	for i,d in enumerate(directories): 

		folder = os.path.join('TrainingData',d)
		
		if os.path.isdir(folder):

			xfname = os.path.join(folder,'data.tiff')
			yfname = os.path.join(folder,'label.tiff')

			if i < 8:
				X = chopImage(xfname, crops, False, True)
				Y = chopImage(yfname, crops, True, True)
			else:
				X = chopImage(xfname, crops, False, False)
				Y = chopImage(yfname, crops, True, False)
							
			allX.append(X)
			allY.append(Y)

			np.save(os.path.join(folder, 'x_'+str(opts.imSize)+'.npy'), X)
			np.save(os.path.join(folder, 'y_'+str(opts.imSize)+'.npy'), Y)


	# Train images 
	np.save('/home/ym1008/Documents/Tricellulin/SemSeg/data/xTrain_' + str(opts.imSize)+'.npy', np.concatenate(allX[:8],0))
	np.save('/home/ym1008/Documents/Tricellulin/SemSeg/data/yTrain_' + str(opts.imSize)+'.npy', np.concatenate(allY[:8],0))

	# Test images 
	np.save('/home/ym1008/Documents/Tricellulin/SemSeg/data/xTest_' + str(opts.imSize)+'.npy', np.concatenate(allX[8:10],0))
	np.save('/home/ym1008/Documents/Tricellulin/SemSeg/data/yTest_' + str(opts.imSize)+'.npy', np.concatenate(allY[8:10],0))

	# Validation images 
	np.save('/home/ym1008/Documents/Tricellulin/SemSeg/data/xVal_' + str(opts.imSize)+'.npy', allX[10])
	np.save('/home/ym1008/Documents/Tricellulin/SemSeg/data/yVal_' + str(opts.imSize)+'.npy', allY[10])