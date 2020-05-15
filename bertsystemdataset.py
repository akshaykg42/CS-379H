import torch
import random
import numpy as np
from torch.utils import data
from torch.utils.data import Dataset, DataLoader, Sampler, SubsetRandomSampler
from sklearn.model_selection import train_test_split

def collate_batch(batch):
	batch_inputs = [item for item in batch]
	batch_size = len(batch_inputs)
	sent_lens = np.array([np.array([len(sent) for sent in example]) for example in batch_inputs])
	max_sent_len = min(512, max(np.array([max(lens) for lens in sent_lens])))
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
	padded_inputs = torch.from_numpy(padded_inputs).long()
	mask = torch.from_numpy(mask).long()
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


class BertSystemDataset(Dataset):
	def __init__(self, data_dir, indices):
		self.data_dir = data_dir
		self.len = len(indices)

	def __len__(self):
		return self.len

	def __getitem__(self, index):
		features = np.load(self.data_dir + '/bert_processed/test/' + str(index) + '.npy', allow_pickle=True)
		return features

def create_datasets(data_dir, indices, batch_size=1):
	# choose the training and test datasets
	data = BertSystemDataset(data_dir, indices)
	
	# define samplers for obtaining training and validation batches
	sampler = SubsetSequentialSampler(indices)
	
	# load test data in batches
	loader = torch.utils.data.DataLoader(data,
										batch_size=1,
										sampler=sampler,
										collate_fn=collate_batch)
	
	return loader
