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
	    # the first 6 parameters are the weights of 6 features (-1,1)
	    # the 7th parameter is max ratio
	    # the 8th parameter is multiplier
	    # the 9th parameter is order
	    # dirichlet?
		return np.array([1,1,2,2,3,3,1,1,-3],dtype=np.float64)

def train(net):
	np.random.seed(123)
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

	# setup
	num_batch = 1
	range_size = 10 # number of videos we test
	batch_size = 4
	print_step = 10
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
			loss.backward()
			optimizer.step()

			# print statistics
			running_loss += loss.item()
			if i % print_step == print_step-1:    # print every 200 mini-batches
				print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / print_step))
				running_loss = 0.0
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
		num_batch = 1
		range_size = 10 # number of videos we test
		batch_size = 4	
		print_step = 10

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
				if i % print_step == print_step-1:    # print every 200 mini-batches
					print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / print_step))
					running_loss = 0.0


if __name__ == "__main__":
	# prepare network
	net = RSNet([8,255,255,1])
	print(net)
	net = net.cuda()
	random_data = torch.rand((8)).cuda()
	print(random_data)
	result = net(random_data)
	print(result)
	exit(0)
	train(net)

	cgen = C_Generator()
	TF = Transformer('compression')
	C_param = cgen.get()
	sim_result = simulate('ucf101-24', data_range=(0,1), TF=TF, C_param=C_param)
	print(sim_result)