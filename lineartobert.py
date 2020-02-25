from utils import *
from model import *
from trainbert import *
from train import *
from trainbert import train as trainbert
from testbert import *
from test import *
from testbert import test as testbert
from lineartobertdataset import *
from summarizationdataset import *
import math
import argparse

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-k', type=int, default=10)
	parser.add_argument('-t', '--type', type=int, default=-1)
	parser.add_argument('-f', '--features', type=int, default=None)
	parser.add_argument('-d', '--datadir', default='pcr_data')
	parser.add_argument('--linear_batchsize', type=int, default=16)
	parser.add_argument('--linear_epochs', type=int, default=100)
	parser.add_argument('--bert_batchsize', type=int, default=1)
	parser.add_argument('--bert_epochs', type=int, default=4)

	args = parser.parse_args()

	DATA_DIR, TYPE, BATCH_SIZE, EPOCHS, FEATURES, K, BERT_BATCH_SIZE, BERT_EPOCHS = \
		args.datadir, args.type, args.linear_batchsize, args.linear_epochs, args.features, args.k, args.bert_batchsize, args.bert_epochs

	print('Loading data...')

	documents, summaries, oracles, types = load(DATA_DIR)
	train_loader, test_loader, valid_loader, available_indices, indices_test = create_datasets(DATA_DIR, oracles, types, TYPE, BATCH_SIZE)
	
	for inputs, mask, targets in train_loader:
		FEATURES = inputs[0].shape[1]
		break

	train(train_loader, valid_loader, num_features=FEATURES, n_epochs=EPOCHS, batch_size=BATCH_SIZE)
	test(test_loader, num_features=FEATURES)

	labels = [oracles[i][TYPE] for i in available_indices]
	dataset = SummarizationDataset(DATA_DIR, available_indices, labels)
	sampler = SubsetSequentialSampler(available_indices)

	dataloader = torch.utils.data.DataLoader(dataset,
												batch_size=1,
												sampler=sampler,
												collate_fn=collate_batch)

	print('Ignore the testing accuracy below this, it is testing on all data including the train set')
	test_scores = [logits.cpu().detach().numpy() for logits in test(dataloader, num_features=FEATURES)]

	top_k = [test_scores[i].flatten().argsort()[-K:][::-1] for i, j in enumerate(available_indices)]
	
	train_loader, test_loader, valid_loader, indices_test = create_lineartobert_datasets(DATA_DIR, available_indices, labels, top_k, batch_size=BERT_BATCH_SIZE)

	trainbert(train_loader, valid_loader, n_epochs=BERT_EPOCHS, batch_size=BERT_BATCH_SIZE)

	test_scores = testbert(test_loader)
