import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(1)
	
class OracleSelectorModel(nn.Module):
	def __init__(self, input_dim, output_dim):
		super(OracleSelectorModel, self).__init__()
		self.linear = nn.Linear(input_dim, output_dim)
		self.softmax = nn.Softmax(dim=0)

	def forward(self, x):
		x = self.linear(x)
		x = self.softmax(x)
		# Returning softmax would be helpful for calculating loss easily but then can't have fixed output_dim
		return torch.argmax(x)

