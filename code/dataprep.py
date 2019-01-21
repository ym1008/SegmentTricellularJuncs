from PIL import Image
import numpy as np
import os

def upID(i, nx,ny):
	if i == 0:
		nx += 1
		n = nx
	elif i == 1:
		ny += 1
		n = ny
	return n, nx, ny


directories = os.listdir('TrainingData/')
fname = ['data.tiff', 'label.tiff']

nx = 0
ny = 0
allImg = []
allY = []

for d in directories: 
	if os.path.isdir(os.path.join('TrainingData', d)):

		for i, f in enumerate(fname):
			dirname, _ = os.path.splitext(os.path.join('TrainingData', d, f))
			if not os.path.exists(dirname):
				os.mkdir(dirname)
		
			img = Image.open(os.path.join('TrainingData', d, f))
			img = np.array(img)
			img = img[:,:,1] # green channel
		
			splitrows = np.split(img, 4, 0) # split into 16 rows (height = 1024/16 = 64 pixels)
		
			for eachrow in splitrows: 
				splitcols = np.split(eachrow, 4, 1) # split each row into 16 (width = 1024/16 = 64 pixels)
			
				for subimg in splitcols:
					subimg = subimg[:,:, np.newaxis]

					if i == 0:
						allImg.append(subimg)
						n1,nx,ny = upID(i,nx,ny)

						allImg.append(np.rot90(subimg,1))
						n2,nx,ny = upID(i,nx,ny)
						
						allImg.append(np.rot90(subimg,2))
						n3,nx,ny = upID(i,nx,ny)
						
						allImg.append(np.rot90(subimg,3))
						n4,nx,ny = upID(i,nx,ny)
					elif i == 1:
						subimg[subimg>100] = 255
						subimg[subimg<101] = 0
						
						allY.append(subimg)
						n1,nx,ny = upID(i,nx,ny)

						allY.append(np.rot90(subimg,1))
						n2,nx,ny = upID(i,nx,ny)
						
						allY.append(np.rot90(subimg,2))
						n3,nx,ny = upID(i,nx,ny)
						
						allY.append(np.rot90(subimg,3))
						n4,nx,ny = upID(i,nx,ny)
					

					Image.fromarray(subimg[:,:,0]).save(os.path.join(dirname, str(n1) + '.tiff'))
					Image.fromarray(np.rot90(subimg[:,:,0], 1)).save(os.path.join(dirname, str(n2) + '_r90.tiff'))
					Image.fromarray(np.rot90(subimg[:,:,0], 2)).save(os.path.join(dirname, str(n3) + '_r180.tiff'))
					Image.fromarray(np.rot90(subimg[:,:,0], 3)).save(os.path.join(dirname, str(n4) + '_r270.tiff'))		

np.save('/home/ym1008/Documents/Tricellulin/SemSeg/data/xData.npy', np.array(allImg))
np.save('/home/ym1008/Documents/Tricellulin/SemSeg/data/yData.npy', np.array(allY))

