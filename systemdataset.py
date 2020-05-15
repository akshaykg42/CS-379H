import torch
import random
import numpy as np
from torch.utils import data
from torch.utils.data import Dataset, DataLoader, Sampler, SubsetRandomSampler
from sklearn.model_selection import train_test_split

def collate_batch(batch):
	batch_inputs = [item for item in batch]
	batch_size = len(batch_inputs)
	lengths = np.array([len(example) for example in batch_inputs])
	max_len = max(lengths)
	num_features = batch_inputs[0].shape[1]
	padded_inputs = np.zeros((batch_size, max_len, num_features))
	for i, example in enumerate(batch_inputs):
		for j, sentence in enumerate(example):
			padded_inputs[i][j] = sentence
	mask = torch.arange(max_len) >= torch.from_numpy(lengths)[:, None]
	padded_inputs = torch.from_numpy(padded_inputs).float().cuda()
	mask = mask.type(torch.uint8).to(torch.bool).cuda()
	return padded_inputs, mask

class SubsetSequentialSampler(Sampler):
    """Samples elements randomly from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (index for index in self.indices)

    def __len__(self):
        return len(self.indices)


class SystemDataset(Dataset):
	def __init__(self, data_dir, indices):
		self.data_dir = data_dir
		self.len = len(indices)

	def __len__(self):
		return self.len

	def __getitem__(self, index):
		features = np.load(self.data_dir + '/processed/test/' + str(index) + '.npy')
		return features

def create_datasets(data_dir, indices, batch_size=1):
	# choose the training and test datasets
	data = SystemDataset(data_dir, indices)
	
	# define samplers for obtaining training and validation batches
	sampler = SubsetSequentialSampler(indices)
	
	# load test data in batches
	loader = torch.utils.data.DataLoader(data,
										batch_size=1,
										sampler=sampler,
										collate_fn=collate_batch)
	
	return loader
