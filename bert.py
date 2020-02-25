from utils import *
from trainbert import *
from testbert import *
from bertdataset import *
import math
import argparse

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-t', '--type', type=int, default=0)
	parser.add_argument('-b', '--batchsize', type=int, default=1)
	parser.add_argument('-e', '--epochs', type=int, default=4)
	parser.add_argument('-d', '--datadir', default='pcr_data')
	parser.add_argument('-m', '--mini', action='store_true', default=False)

	args = parser.parse_args()

	DATA_DIR, TYPE, BATCH_SIZE, EPOCHS, MINI = \
		args.datadir, args.type, args.batchsize, args.epochs, args.mini

	print('Loading data...')
	documents, summaries, oracles, types = load(DATA_DIR)
	train_loader, test_loader, valid_loader, available_indices, indices_test = create_datasets(DATA_DIR, oracles, types, TYPE, BATCH_SIZE, mini=MINI)
	train(train_loader, valid_loader, n_epochs=EPOCHS, batch_size=BATCH_SIZE)
	test_scores = test(test_loader)
