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
from DDPG.ddpgtrain import Trainer
from DDPG.ddpgbuffer import MemoryBuffer


# setup
classes_num = 24
batch_size = 1
num_batch = classes_num//batch_size
print_step = 10
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

class ParetoFront:
	def __init__(self):
		# points on pareto front
		# (acc,cr,c_param)
		self.data = {}
		# average compression param of cfgs
		# on and not on pareto front
		self.dominated_c_param = np.zeros(6,dtype=np.float64)
		self.dominated_cnt = 1e-6
		self.dominating_c_param = np.zeros(6,dtype=np.float64)
		self.dominating_cnt = 1e-6

	def add(self, c_param, dp):
		reward = 0
		# check the distance of (accuracy,bandwidth) to the previous Pareto Front
		to_remove = set()
		add_new = False
		for point in self.data:
			# if there is a same point, we dont add this
			if point[:2] == dp: break
			# if a point is dominated
			if point[0] <= dp[0] and point[1] <= dp[1]:
				to_remove.add(point)
				reward = 1
				add_new = True
			# if the new point is dominated
			elif point[0] >= dp[0] and point[1] >= dp[1]:
				reward = -1
				break
			else:
				reward = max(reward,dp[0]-point[0],dp[1]-point[1])
				add_new = True
		if not self.data: add_new = True

		# remove dominated points
		for point in to_remove:
			self.dominated_c_param += self.data[point]
			self.dominated_cnt += 1
			self.dominating_c_param -= self.data[point]
			self.dominating_cnt -= 1
			del self.data[point]

		# update the current Pareto Front
		if add_new:
			self.dominating_c_param += c_param
			self.dominating_cnt += 1
			self.data[dp] = c_param
		else:
			self.dominated_c_param += c_param
			self.dominated_cnt += 1

		return reward


	def get_observation(self):
		return np.concatenate((self.dominating_c_param/self.dominating_cnt,self.dominated_c_param/self.dominated_cnt))

class C_Generator:
	def __init__(self):
		MAX_BUFFER = 1000000
		S_DIM = 12
		A_DIM = 6
		A_MAX = 0.5 #[-.5,.5]

		self.ram = MemoryBuffer(MAX_BUFFER)
		self.trainer = Trainer(S_DIM, A_DIM, A_MAX, self.ram)
		self.paretoFront = ParetoFront()

	def get(self):
		# get an action from the actor
		state = np.float32(self.paretoFront.get_observation())
		self.action = self.trainer.get_exploration_action(state)
		# self.C_param = self.uniform_init_gen()
		return self.action

	def uniform_init_gen(self):
		# 0,1,2:feature weights; 3,4:lower and upper; 5:order
		output = np.zeros(6,dtype=np.float64)
		output[:4] = np.random.randint(1,10,4)
		output[4] = np.random.randint(output[3],11)
		output[:5] /= 10
		output[5] = np.random.randint(0,5) #[1/3,1/2,1,2,3]
		# output = np.array([1,1,1,1,1,2],dtype=np.float64)
		return output   

	def optimize(self, datapoint, done):
		# if one episode ends, do nothing
		if done: return
		# use (accuracy,bandwidth) to update observation
		state = self.paretoFront.get_observation()
		reward = self.paretoFront.add(self.action, datapoint)
		new_state = self.paretoFront.get_observation()
		# add experience to ram
		self.ram.add(state, self.action, reward, new_state)
		print(state, self.action, reward, new_state)
		# optimize the network 
		self.trainer.optimize()


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

	cgen = C_Generator()

	for epoch in range(10):
		running_loss = 0.0
		TF = Transformer('compression')

		for bi in range(num_batch):
			inputs,labels = [],[]
			for k in range(batch_size):
				di = bi*batch_size + k # data index
				class_idx = di%24
				# DDPG-based generator
				C_param = cgen.get()
				# start counting the compressed size
				TF.start()
				# apply the compression param chosen by the generator
				sim_result = simulate(opt.dataset, class_idx=class_idx, TF=TF, C_param=C_param, AD_param=AD_param)
				# get the compression ratio
				cr = TF.get_compression_ratio()
				# optimize generator
				cgen.optimize((sim_result[2][class_idx],cr))

				print_str = str(class_idx)+str(C_param)+'\t'+str(sim_result)
				log_file.write(print_str+'\n')
				inputs.append(C_param)
				labels.append(sim_result[2][class_idx]) # accuracy of IoU=0.5
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
				class_idx = di%24
				fetch_start = time.perf_counter()
				C_param = cgen.get()
				sim_result = simulate(opt.dataset, class_idx=class_idx, TF=TF, C_param=C_param, AD_param=AD_param)
				fetch_end = time.perf_counter()
				print_str = str(class_idx)+str(C_param)+'\t'+str(sim_result)+'\t'+str(fetch_end-fetch_start)
				print(print_str)
				log_file.write(print_str+'\n')
				inputs.append(C_param)
				labels.append(sim_result[2][class_idx]) # accuracy of IoU=0.5
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

