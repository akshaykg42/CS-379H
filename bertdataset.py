import torch
import random
import numpy as np
from torch.utils import data
from torch.utils.data import Dataset, DataLoader, Sampler, SubsetRandomSampler
from sklearn.model_selection import train_test_split

def collate_batch(batch):
	batch_inputs = [item[0] for item in batch]
	batch_labels = [item[1] for item in batch]
	batch_size = len(batch_inputs)
	sent_lens = np.array([np.array([len(sent) for sent in example]) for example in batch_inputs])
	max_sent_len = max(np.array([max(lens) for lens in sent_lens]))
	doc_lens = np.array([len(example) for example in batch_inputs])
	max_doc_len = max(doc_lens)
	padded_inputs = np.zeros((batch_size, max_doc_len, max_sent_len))
	mask = np.zeros((batch_size, max_doc_len, max_sent_len))
	for i, example in enumerate(batch_inputs):
		for j, sentence in enumerate(example):
			for k, token in enumerate(sentence):
				if(k < max_sent_len):
					padded_inputs[i][j][k] = token
					mask[i][j][k] = 1
	padded_inputs = np.vstack(padded_inputs)
	mask = np.vstack(mask)
	batch_labels = torch.from_numpy(np.array(batch_labels)).unsqueeze(1)
	padded_inputs = torch.from_numpy(padded_inputs).long()
	mask = torch.from_numpy(mask).long()
	doc_lens = torch.from_numpy(doc_lens)
	return padded_inputs, mask, batch_labels, doc_lens

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

class BertDataset(Dataset):
	def __init__(self, data_dir, indices, labels):
		self.data_dir = data_dir
		self.labels = {indices[i] : labels[i] for i in range(len(labels))}

	def __len__(self):
		return len(self.labels)

	def __getitem__(self, index):
		features = np.load(self.data_dir + '/bert_processed/documents/' + str(index) + '.npy', allow_pickle=True)
		label = self.labels[index]
		return features, label

class BertMiniDataset(Dataset):
	def __init__(self, data_dir, indices, labels, minidoc_size=10):
		self.data_dir = data_dir
		self.minidoc_size = minidoc_size
		self.labels = {indices[i] : labels[i] for i in range(len(labels))}

	def __len__(self):
		return len(self.labels)

	def __getitem__(self, index):
		features = np.load(self.data_dir + '/bert_processed/documents/' + str(index) + '.npy', allow_pickle=True)
		label = self.labels[index]
		indices, label = get_mini_indices(len(features), self.minidoc_size, label)
		features = np.array([features[i] for i in indices])
		return features, label

def create_datasets(data_dir, oracles, sent_type, batch_size, mini=False):
	labels, available_indices = [], []
	for i, j in enumerate(oracles):
		try:
			labels.append(j[sent_type])
			available_indices.append(i)
		except:
			pass

	indices_train, indices_test, labels_train, labels_test = train_test_split(available_indices, labels, test_size=0.2)
	indices_train, indices_val, labels_train, labels_val = train_test_split(indices_train, labels_train, test_size=0.25)
	
	BertDatasetType = BertMiniDataset if mini else BertDataset

	# choose the training and test datasets
	train_data = BertDatasetType(data_dir, indices_train, labels_train)
	valid_data = BertDatasetType(data_dir, indices_val, labels_val)
	test_data = BertDatasetType(data_dir, indices_test, labels_test)
	
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
												batch_size=batch_size,
												sampler=valid_sampler,
												collate_fn=collate_batch)
	
	# load test data in batches
	test_loader = torch.utils.data.DataLoader(test_data,
												batch_size=batch_size,
												sampler=test_sampler,
												collate_fn=collate_batch)
	
	return train_loader, test_loader, valid_loader, set(available_indices), indices_test
