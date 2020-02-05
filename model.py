import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(1)
	
class OracleSelectorModel(nn.Module):
	def __init__(self, input_dim, hidden=512):
		super(OracleSelectorModel, self).__init__()
		self.linear = nn.Linear(input_dim, hidden)
		self.ReLU = nn.ReLU()
		self.linear2 = nn.Linear(hidden, 1)
		self.log_softmax = nn.LogSoftmax(dim=1)

	def forward(self, x, mask=None):
		x = self.linear(x)
		x = self.ReLU(x)
		x = self.linear2(x)
		x[mask] = -float("inf")
		x = self.log_softmax(x)
		return x, torch.argmax(x, 1)
