import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(1)

class classifier(nn.Module):
  def __init__(self, input_dim, output_dim):
    super(classifier, self).__init__()
    self.linear = nn.Linear(input_dim, output_dim)

  def forward(self, x):
        return torch.argmax(F.log_softmax(self.linear(x)))