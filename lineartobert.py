from utils import *
from model import *
from trainbert import *
from train import *
from trainbert import train as trainbert
from testbert import *
from test import *
from testbert import test as testbert
from summarizationdataset import *
import math
import argparse

class LinearToBERTDataset(Dataset):
	def __init__(self, data_dir, indices, labels, top_k):
		self.data_dir = data_dir
		self.top_k = top_k
		self.labels = {indices[i] : labels[i] for i in range(len(labels))}

	def __len__(self):
		return len(self.labels)

	def __getitem__(self, index):
		features = np.load(self.data_dir + '/bert_processed/documents/' + str(index) + '.npy', allow_pickle=True)
		label = self.labels[index]
		label = self.top_k.index(label)
		features = np.array([features[i] for i in self.top_k])
		return features, label

def create_lineartobert_datasets(data_dir, available_indices, labels, top_k, batch_size):
	updated_available_indices = [i for i, j in enumerate(available_indices) if labels[i] in top_k[i]]

	available_indices = [available_indices[i] for i in updated_available_indices]
	labels = [labels[i] for i in updated_available_indices]
	top_k = [top_k[i] for i in updated_available_indices]

	indices_train, indices_test, labels_train, labels_test, top_k_train, top_k_test = train_test_split(available_indices, labels, top_k, test_size=0.2)
	indices_train, indices_val, labels_train, labels_val, top_k_train, top_k_val = train_test_split(indices_train, labels_train, top_k test_size=0.25)
	
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

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-t', '--type', type=int, default=0)
	parser.add_argument('-b', '--batchsize', type=int, default=16)
	parser.add_argument('-e', '--epochs', type=int, default=100)
	parser.add_argument('-f', '--features', type=int, default=None)
	parser.add_argument('-d', '--datadir', default='pcr_data')
	parser.add_argument('-m', '--mini', action='store_true', default=False)
	parser.add_argument('-k', type=int, default=10)

	args = parser.parse_args()

	DATA_DIR, TYPE, BATCH_SIZE, EPOCHS, FEATURES, MINI, K = \
		args.datadir, args.type, args.batchsize, args.epochs, args.features, args.mini, args.k

	print('Loading data...')
	documents, summaries, oracles = load(DATA_DIR)
	train_loader, test_loader, valid_loader, available_indices, indices_test = create_datasets(DATA_DIR, oracles, TYPE, BATCH_SIZE, MINI)
	for inputs, mask, targets in train_loader:
		FEATURES = inputs[0].shape[1]
		break
	train(train_loader, valid_loader, num_features=FEATURES, n_epochs=EPOCHS, batch_size=BATCH_SIZE)

	labels = [oracles[i][TYPE] for i in available_indices]
	dataset = SummarizationDataset(DATA_DIR, available_indices, labels)
	sampler = SubsetSequentialSampler(available_indices)

	dataloader = torch.utils.data.DataLoader(dataset,
												batch_size=1,
												sampler=sampler,
												collate_fn=collate_batch)

	test_scores = [logits.cpu().detach().numpy() for logits in test(dataloader, num_features=FEATURES)]

	top_k = [test_scores[i].flatten().argsort()[-K:][::-1] for i, j in enumerate(available_indices)]
	
	train_loader, test_loader, valid_loader, indices_test = create_lineartobert_datasets(DATA_DIR, available_indices, labels, top_k, batch_size=1)

	trainbert(train_loader, valid_loader, n_epochs=4, batch_size=1)

	test_scores = testbert(test_loader)
