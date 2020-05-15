import os
import torch
import random
import numpy as np
from torch.utils import data
from torch.utils.data import Dataset, DataLoader, Sampler, SubsetRandomSampler
from sklearn.model_selection import train_test_split

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
	mask = torch.arange(max_len) >= torch.from_numpy(lengths)[:, None]
	padded_inputs = torch.from_numpy(padded_inputs).float().cuda()
	mask = mask.type(torch.uint8).to(torch.bool).cuda()
	# mask = np.arange(max_len) < lengths[:, None]
	# padded_inputs = torch.from_numpy(padded_inputs).float().cuda()
	# mask = (~(torch.from_numpy(mask).byte())).to(torch.bool).cuda()
	return padded_inputs, mask, batch_labels

def get_mini_indices(old_len, new_len, label):
	if(new_len >= old_len):
		options = list(range(old_len))
		random.shuffle(options)
		new_label = options.index(label)
		return options, new_label
	else:
		options = list(range(old_len))
		options.pop(label)
		random.shuffle(options)
		indices = options[:new_len]
		new_label = random.randint(0, new_len - 1)
		indices[new_label] = label
		return indices, new_label

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


class SummarizationDataset(Dataset):
	def __init__(self, data_dir, indices, labels, typ):
		self.data_dir = data_dir
		self.labels = {indices[i] : labels[i] for i in range(len(labels))}
		self.typ = typ

	def __len__(self):
		return len(self.labels)

	def __getitem__(self, index):
		features = np.load(self.data_dir + '/processed/{}/'.format(self.typ) + str(index) + '.npy')
		label = self.labels[index]
		return features, label

class MiniSummarizationDataset(Dataset):
	def __init__(self, data_dir, indices, labels, typ, minidoc_size=5):
		self.data_dir = data_dir
		self.minidoc_size = minidoc_size
		self.labels = {indices[i] : labels[i] for i in range(len(labels))}
		self.typ = typ

	def __len__(self):
		return len(self.labels)

	def __getitem__(self, index):
		features = np.load(self.data_dir + '/processed/{}/'.format(self.typ) + str(index) + '.npy', allow_pickle=True)
		label = self.labels[index]
		indices, label = get_mini_indices(len(features), self.minidoc_size, label)
		features = np.array([features[i] for i in indices])
		return features, label

def get_indices(data_dir):
	train_files = os.listdir(data_dir + '/processed/train/')
	val_files = os.listdir(data_dir + '/processed/val/')
	test_files = os.listdir(data_dir + '/processed/test/')
	train_indices = [int(file[:-4]) for file in train_files if file.endswith('.npy')]
	val_indices = [int(file[:-4]) for file in val_files if file.endswith('.npy')]
	test_indices = [int(file[:-4]) for file in test_files if file.endswith('.npy')]
	return train_indices, val_indices, test_indices

def create_datasets(data_dir, oracles, types, sent_type, batch_size, mini=False):
	labels, available_indices = [], []
	indices_train, indices_val, indices_test = get_indices(data_dir)
	labels_train, labels_val, labels_test = [], [], []
	indices_train = [i for i in indices_train if any([sent_type in t_ for t_ in types[i]])]
	indices_val = [i for i in indices_val if any([sent_type in t_ for t_ in types[i]])]
	indices_test = [i for i in indices_test if any([sent_type in t_ for t_ in types[i]])]
	available_indices = sorted(indices_train + indices_test + indices_val)
	summary_indices_test = []

	for i in indices_train:
		t, o = types[i], oracles[i]
		for j, t_ in enumerate(t):
			if(sent_type in t_):
				labels_train.append(o[j])
				break
	for i in indices_val:
		t, o = types[i], oracles[i]
		for j, t_ in enumerate(t):
			if(sent_type in t_):
				labels_val.append(o[j])
				break
	for i in indices_test:
		t, o = types[i], oracles[i]
		for j, t_ in enumerate(t):
			if(sent_type in t_):
				labels_test.append(o[j])
				summary_indices_test.append(j)
				break

	DatasetType = MiniSummarizationDataset if mini else SummarizationDataset

	# choose the training and test datasets
	train_data = DatasetType(data_dir, indices_train, labels_train, 'train')
	valid_data = DatasetType(data_dir, indices_val, labels_val, 'val')
	test_data = DatasetType(data_dir, indices_test, labels_test, 'test')
	
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
	
	return train_loader, test_loader, valid_loader, available_indices, indices_test, labels_test, summary_indices_test
