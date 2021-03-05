import cv2
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from video_mAP import simulate, setup_param, setup_opt
from transformer import Transformer
from opts import parse_opts


# setup
range_size = 1 # number of videos we test
video_num = 910
batch_size = 1
num_batch = video_num//(batch_size*range_size)
print_step = 1
eval_step = 1
PATH = 'backup/rsnet.pth'

class RSNet(nn.Module):
	def __init__(self, settings):
		super(RSNet, self).__init__()
		num_layer = len(settings)-1
		self.layers = nn.ModuleList()
		for i in range(num_layer):
			self.layers += [nn.Linear(settings[i], settings[i+1])]

	def forward(self, x):
		for layer in self.layers[:-1]:
			x = F.relu(layer(x))
		x = self.layers[-1](x)
		return x


class C_Generator:
	def __init__(self,opt):
		img_w,img_h,num_w,num_h = opt.tile_settings
		self.tilew,self.tileh = img_w//num_w,img_h//num_h
		self.num_features = 4

	def get(self):
		# the first 6 parameters are the weights of 6 features (0,1)
		# the 7th parameter is max ratio (0,1)
		# the 8th parameter is order [-3,3]
		# dirichlet?
		# can we reject bad config at the begining?
		C_param = self.uniform_init_gen()
		return C_param

	def uniform_init_gen(self):
		output = np.zeros(self.num_features+2,dtype=np.float64)
		output[:self.num_features+1] = np.random.randint(1,10,self.num_features+1)/10
		output[self.num_features+1] = np.random.randint(-2,2)
		return output

	def c_param_to_tilesizes(self,C_param):
		# weight of different features
		weights = C_param[:self.num_features]
		# (0,1) indicating the total quality after compression
		A = C_param[self.num_features]
		# parameter of the function to amplify the score
		# sigma=0,1,...,9; k=-3,...,3: no big difference with larger value
		# k decides the weights should have small or big difference
		# sigma = C_param[num_features+1]
		k = C_param[self.num_features+1]
		normalized_score = counts/(np.sum(counts,axis=0)+1e-6)
		weights /= (np.sum(weights)+1e-6)
		# ws of all tiles sum up to 1
		weighted_scores = np.matmul(normalized_score,weights)
		# the weight is more valuable when its value is higher
		weighted_scores = np.exp((10**k)*weighted_scores) - 1
		weighted_scores /= (np.max(weighted_scores)+1e-6)
		# quality of each tile?
		quality = A*weighted_scores

		tilesizes = [(int(np.rint(self.tilew*q)),int(np.rint(self.tileh*r))) for q in quality]
		return tilesizes


def train(net):
	np.random.seed(123)
	criterion = nn.MSELoss(reduction='sum')
	optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
	log_file = open('training.log', "w", 1)
	log_file.write('Training...\n')

	# setup target network
	# so that we only do this once
	opt = parse_opts()
	setup_opt(opt)
	opt.dataset = 'ucf101-24'
	AD_param = setup_param(opt)

	for epoch in range(1):
		running_loss = 0.0
		cgen = C_Generator(opt)
		TF = Transformer('compression',opt)

		for bi in range(num_batch):
			inputs,labels = [],[]
			for k in range(batch_size):
				di = bi*batch_size + k # data index
				data_range = (di*range_size,di*range_size+range_size)
				fetch_start = time.perf_counter()
				C_param = cgen.get()
				tilesizes = cgen.c_param_to_tilesizes(C_param)
				sim_result = simulate(opt.dataset, data_range=data_range, TF=TF, tilesizes=tilesizes, AD_param=AD_param)
				fetch_end = time.perf_counter()
				result = [(np.sum(APs),AP_new) for APs,AP_new in sim_result]
				print_str = str(data_range)+str(C_param)+str(tilesizes)+' '+str(result)+' '+str(fetch_end-fetch_start)+'\n'
				print(print_str)
				log_file.write(print_str)
				inputs.append(C_param)
				labels.append(sim_result[4][1]) # accuracy of IoU=0.5
			inputs = torch.FloatTensor(inputs).cuda()
			labels = torch.FloatTensor(labels).cuda()

			# zero gradient
			optimizer.zero_grad()

			# forward + backward + optimize
			outputs = net(inputs)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()

			# print statistics
			running_loss += loss.item()
			val_loss = abs(torch.mean(labels.cpu()-outputs.cpu()))
			print_str = '{:d}, {:d}, loss {:.6f}, val loss {:.6f}'.format(epoch + 1, bi + 1, loss.item(), val_loss)
			print(print_str)
			log_file.write(print_str + '\n')
			if bi % print_step == 0 and bi>0:    
				print_str = '{:d}, {:d}, loss {:.6f}'.format(epoch + 1, bi + 1, running_loss / print_step)
				print(print_str)
				log_file.write(print_str + '\n')
				running_loss = 0.0
		torch.save(net.state_dict(), PATH)
		# evaluation
		# if epoch%eval_step==0 and epoch>0:
		# 	validate(net)

	# val_loss = validate(net,log_file)
	# ptr_str = "loss:{:1.6f}\n".format(val_loss)
	# log_file.write(ptr_str)

# load if needed
# net.load_state_dict(torch.load('backup/rsnet.pth'))
def validate(net,log_file):
	np.random.seed(321)
	# setup target network
	# so that we only do this once
	opt = parse_opts()
	setup_opt(opt)
	opt.dataset = 'ucf101-24'
	AD_param = setup_param(opt)
	with torch.no_grad():
		val_loss = 0.0
		val_cnt = 0
		running_loss = 0.0
		cgen = C_Generator()
		TF = Transformer('compression')
		log_file.write('Evaluation...\n')

		for bi in range(num_batch):
			inputs,labels = [],[]
			for k in range(batch_size):
				di = bi*batch_size + k # data index
				data_range = (di*range_size,di*range_size+range_size)
				fetch_start = time.perf_counter()
				C_param = cgen.get()
				sim_result = simulate(opt.dataset, data_range=data_range, TF=TF, C_param=C_param, AD_param=AD_param)
				fetch_end = time.perf_counter()
				print_str = str(data_range)+str(C_param)+'\t'+str(sim_result)+'\t'+str(fetch_end-fetch_start)+'\n'
				# print(print_str)
				log_file.write(print_str)
				inputs.append(C_param)
				labels.append(sim_result[0][1]) # accuracy of IoU=0.5
			inputs = torch.FloatTensor(inputs).cuda()
			labels = torch.FloatTensor(labels).cuda()

			# forward + backward + optimize
			outputs = net(inputs)

			# print statistics
			running_loss += loss.item()
			val_loss += abs(torch.mean(labels.cpu()-outputs.cpu()))
			val_cnt += 1
			if bi % print_step == 0:    # print every 200 mini-batches
				print_str = '{:d}, loss {:.6}\n'.format(bi + 1, running_loss / print_step)
				print(print_str)
				log_file.write(print_str)
				running_loss = 0.0
	return val_loss/val_cnt


if __name__ == "__main__":
	# prepare network
	net = RSNet([6,255,255,1])
	# net.load_state_dict(torch.load('backup/rsnet.pth'))
	net = net.cuda()
	train(net)

	# cgen = C_Generator()
	# TF = Transformer('compression')
	# C_param = cgen.get()
	# sim_result = simulate('ucf101-24', data_range=(0,1), TF=TF, C_param=C_param)
	# print(sim_result)