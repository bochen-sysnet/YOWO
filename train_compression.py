import cv2
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from video_mAP import simulate
from transformer import Transformer


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
	def __init__(self):
		pass

	def get(self):
		# the first 6 parameters are the weights of 6 features (0,1)
		# the 7th parameter is max ratio (0,1)
		# the 8th parameter is multiplier (0,1)
		# the 9th parameter is order [-3,3]
		# dirichlet?
		C_param = self,uniform_init_gen()
		return C_param

	def uniform_init_gen(self):
		output = np.zeros(9,dtype=np.float64)
		output[:8] = np.random.randint(0,10,8)/10
		output[8] = np.random.randint(-3,3)
		return output


def train(net):
	np.random.seed(123)
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

	# setup
	range_size = 10 # number of videos we test
	video_num = 137557 # 9126 for J-HMDB
	batch_size = 4
	num_batch = 2 # video_num//(batch_size*range_size)
	print_step = 1
	eval_step = 1

	for epoch in range(1):
		running_loss = 0.0
		cgen = C_Generator()
		TF = Transformer('compression')

		for bi in range(num_batch):
			inputs,labels = [],[]
			for k in range(batch_size):
				di = bi*batch_size + k # data index
				data_range = (di*range_size,di*range_size+range_size)
				C_param = cgen.get()
				sim_result = simulate('ucf101-24', data_range=data_range, TF=TF, C_param=C_param)
				print(data_range,C_param,sim_result)
				inputs.append(C_param)
				labels.append(sim_result[0][4]) # accuracy of IoU=0.5
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
			if i % print_step == 0 and print_step>0:    
				print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / print_step))
				running_loss = 0.0
		# evaluation
		if epoch%eval_step==0 and epoch>0:
			validate(net)
	PATH = 'backup/rsnet.pth'
	torch.save(net.state_dict(), PATH)

# load if needed
# net.load_state_dict(torch.load(PATH))
def validate(net):
	np.random.seed(321)
	with torch.no_grad():
		# setup
		range_size = 10 # number of videos we test
		video_num = 137557 # 9126 for J-HMDB
		batch_size = 4
		num_batch = 1 # video_num//batch_size
		print_step = 1
		eval_step = 1

		for epoch in range(1):
			running_loss = 0.0
			cgen = C_Generator()
			TF = Transformer('compression')

			for bi in range(num_batch):
				inputs,labels = [],[]
				for di in range(batch_size):
					data_range = (bi*range_size,(bi+1)*range_size)
					C_param = cgen.get()
					sim_result = simulate('ucf101-24', data_range=data_range, TF=TF, C_param=C_param)
					inputs.append(C_param)
					labels.append(sim_result[0][4]) # accuracy of IoU=0.5
				inputs = torch.FloatTensor(inputs).cuda()
				labels = torch.FloatTensor(labels).cuda()

				# zero gradient
				optimizer.zero_grad()

				# forward + backward + optimize
				outputs = net(inputs)
				loss = criterion(outputs, labels)

				# print statistics
				running_loss += loss.item()
				if i % print_step == 0:    # print every 200 mini-batches
					print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / print_step))
					running_loss = 0.0


if __name__ == "__main__":
	# prepare network
	net = RSNet([8,255,255,1])
	net = net.cuda()
	train(net)

	# cgen = C_Generator()
	# TF = Transformer('compression')
	# C_param = cgen.get()
	# sim_result = simulate('ucf101-24', data_range=(0,1), TF=TF, C_param=C_param)
	# print(sim_result)