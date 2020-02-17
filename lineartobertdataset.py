import torch
import random
import numpy as np
from torch.utils import data
from bertdataset import SubsetSequentialSampler, collate_batch
from torch.utils.data import Dataset, DataLoader, Sampler, SubsetRandomSampler
from sklearn.model_selection import train_test_split

class LinearToBERTDataset(Dataset):
	def __init__(self, data_dir, indices, labels, top_k):
		self.data_dir = data_dir
		self.top_k = {indices[i] : top_k[i] for i in range(len(top_k))}
		self.labels = {indices[i] : labels[i] for i in range(len(labels))}

	def __len__(self):
		return len(self.labels)

	def __getitem__(self, index):
		features = np.load(self.data_dir + '/bert_processed/documents/' + str(index) + '.npy', allow_pickle=True)
		label = self.labels[index]
		label = np.where(self.top_k[index] == label)[0][0]
		features = np.array([features[i] for i in self.top_k[index]])
		return features, label

def create_lineartobert_datasets(data_dir, available_indices, labels, top_k, batch_size):
	updated_available_indices = [i for i, j in enumerate(available_indices) if labels[i] in top_k[i]]

	available_indices = [available_indices[i] for i in updated_available_indices]
	labels = [labels[i] for i in updated_available_indices]
	top_k = [top_k[i] for i in updated_available_indices]

	indices_train, indices_test, labels_train, labels_test, top_k_train, top_k_test = train_test_split(available_indices, labels, top_k, test_size=0.2)
	indices_train, indices_val, labels_train, labels_val, top_k_train, top_k_val = train_test_split(indices_train, labels_train, top_k_train, test_size=0.25)
	
	# choose the training and test datasets
	train_data = LinearToBERTDataset(data_dir, indices_train, labels_train, top_k_train)
	valid_data = LinearToBERTDataset(data_dir, indices_val, labels_val, top_k_val)
	test_data = LinearToBERTDataset(data_dir, indices_test, labels_test, top_k_test)
	
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
	
	return train_loader, test_loader, valid_loader, indices_test