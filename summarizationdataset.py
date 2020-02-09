import torch
import numpy as np
from torch.utils import data
from torch.utils.data import Dataset, DataLoader, Sampler, SubsetRandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split


data_dir = 'pcr_data/'

def collate_batch(batch):
	batch_inputs = [item[0] for item in batch]
	batch_labels = torch.from_numpy(np.array([item[1] for item in batch])).unsqueeze(1).cuda()
	batch_size = len(batch_inputs)
	lengths = np.array([len(example) for example in batch_inputs])
	max_len = max(lengths)
	num_features = batch_inputs[0].shape[1]
	padded_inputs = np.zeros((batch_size, max_len, num_features))
	for i, example in enumerate(batch_inputs):
		for j, sentence in enumerate(example):
			padded_inputs[i][j] = sentence
	mask = np.arange(max_len) < lengths[:, None]
	padded_inputs = torch.from_numpy(padded_inputs).float().cuda()
	mask = (~(torch.from_numpy(mask).byte())).to(torch.bool).cuda()
	return padded_inputs, mask, batch_labels

class SubsetSequentialSampler(Sampler):
    """Samples elements randomly from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)


class SummarizationDataset(Dataset):
	def __init__(self, X_indices, y):
		self.labels = {X_indices[i] : y[i] for i in range(len(y))}

	def __len__(self):
		return len(self.labels)

	def __getitem__(self, index):
		X = np.load(data_dir + 'processed/documents/' + str(index) + '.npy')
		y = self.labels[index]
		return X, y

def create_datasets(oracles, sent_type, batch_size):
	labels, available_indices = [], []
	for i, j in enumerate(oracles):
		try:
			labels.append(j[sent_type])
			available_indices.append(i)
		except:
			pass

	indices_train, indices_test, labels_train, labels_test = train_test_split(available_indices, labels, test_size=0.2)
	indices_train, indices_val, labels_train, labels_val = train_test_split(indices_train, labels_train, test_size=0.25)
	
	# choose the training and test datasets
	train_data = SummarizationDataset(indices_train, labels_train)
	valid_data = SummarizationDataset(indices_val, labels_val)
	test_data = SummarizationDataset(indices_test, labels_test)
	
	# define samplers for obtaining training and validation batches
	train_sampler = SubsetRandomSampler(indices_train)
	valid_sampler = SubsetRandomSampler(indices_val)
	test_sampler = SubsetSequentialSampler(indices_test)
	
	# load training data in batches
	train_loader = torch.utils.data.DataLoader(train_data,
												batch_size=batch_size,
												sampler=train_sampler,
												collate_fn=collate_batch)
	
	# load validation data in batches
	valid_loader = torch.utils.data.DataLoader(valid_data,
												batch_size=len(indices_val),
												sampler=valid_sampler,
												collate_fn=collate_batch)
	
	# load test data in batches
	test_loader = torch.utils.data.DataLoader(test_data,
												batch_size=len(indices_test),
												sampler=test_sampler,
												collate_fn=collate_batch)
	
	return train_loader, test_loader, valid_loader, set(available_indices), indices_test
