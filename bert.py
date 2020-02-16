from utils import *
from trainbert import *
from testbert import *
from bertdataset import *
import math

data_dir = 'pcr_data/'
sent_type = -1
BATCH_SIZE = 1
EPOCHS = 4
MINI = True

if __name__ == '__main__':
	print('Loading data...')
	documents, summaries, oracles = load(data_dir)
	train_loader, test_loader, valid_loader, available_indices, indices_test = create_datasets(data_dir, oracles, sent_type, BATCH_SIZE, mini=MINI)
	train(train_loader, valid_loader, n_epochs=EPOCHS, batch_size=BATCH_SIZE)
	test_scores = test(test_loader)
