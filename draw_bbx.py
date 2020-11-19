import os
from PIL import Image
from PIL import ImageDraw

img_dir = "/home/monet/research/dataset/ucf24/rgb-images/Biking/v_Biking_g01_c01"
bbx_dir = "/home/monet/research/YOWO/ucf_detections/detections_9"
out_dir = "/home/monet/research/YOWO/examples"
num_images = 10
for i in range(num_images):
	img_path = os.path.join(img_dir, '%05d.jpg' % (i+1))
	img = Image.open(img_path)
	draw = ImageDraw.Draw(img)
	bbx_path = os.path.join(bbx_dir, 'Biking_v_Biking_g01_c01_%05d.txt' % (i+1))
	with open(bbx_path,'r') as f:
		for line in f:
			elem = line.strip().split(' ')
			a_cls = int(elem[0])
			prob = float(elem[1])
			x1 = int(elem[2])
			y1 = int(elem[3])
			x2 = int(elem[4])
			y2 = int(elem[5])
			if a_cls == 3:
				draw.rectangle(((x1, y1), (x2, y2)), outline="green")
			else:
				draw.rectangle(((x1, y1), (x2, y2)), outline="red")
	save_path = os.path.join(out_dir, '%05d.jpg' % (i+1))
	img.save(save_path)