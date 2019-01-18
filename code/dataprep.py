from PIL import Image
import numpy as np
import os

directories = os.listdir('TrainingData/')
fname = ['data.tiff', 'label.tiff']

nx = 0
ny = 0

for d in directories: 
	if os.path.isdir(os.path.join('TrainingData', d)):

		for i, f in enumerate(fname):
			dirname, _ = os.path.splitext(os.path.join('TrainingData', d, f))
			if not os.path.exists(dirname):
				os.mkdir(dirname)
		
			img = Image.open(os.path.join('TrainingData', d, f))
			img = np.array(img)
			img = img[:,:,1] # green channel
		
			splitrows = np.split(img, 16, 0) # split into 16 rows (height = 1024/16 = 64 pixels)
		
			for eachrow in splitrows: 
				splitcols = np.split(eachrow, 16, 1) # split each row into 16 (width = 1024/16 = 64 pixels)
			
				for subimg in splitcols:
					if i == 0:
						nx += 1
						n = nx
					elif i == 1:
						ny += 1
						n = ny

					Image.fromarray(subimg).save(os.path.join(dirname, str(n) + '.tiff'))
					Image.fromarray(np.rot90(subimg, 1)).save(os.path.join(dirname, 'r90_' + str(n) + '.tiff'))
					Image.fromarray(np.rot90(subimg, 2)).save(os.path.join(dirname, 'r180_' + str(n) + '.tiff'))
					Image.fromarray(np.rot90(subimg, 3)).save(os.path.join(dirname, 'r280_' + str(n) + '.tiff'))
					Image.fromarray(np.fliplr(subimg)).save(os.path.join(dirname, 'H_' + str(n) + '.tiff'))	
					Image.fromarray(np.rot90(np.fliplr(subimg),1)).save(os.path.join(dirname, 'H_r90_' + str(n) + '.tiff'))	
					Image.fromarray(np.rot90(np.fliplr(subimg), 2)).save(os.path.join(dirname, 'H_r180_' + str(n) + '.tiff'))	
					Image.fromarray(np.rot90(np.fliplr(subimg), 3)).save(os.path.join(dirname, 'H_r270_' + str(n) + '.tiff'))			
					


