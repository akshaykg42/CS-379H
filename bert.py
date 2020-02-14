from utils import *
from trainbert import *
from bertdataset import *
import math

data_dir = 'pcr_data/'
sent_type = 0
BATCH_SIZE = 1
EPOCHS = 4
FEATURES = None

if __name__ == '__main__':
	print('Loading data...')
	documents, summaries, oracles = load(data_dir)
	train_loader, test_loader, valid_loader, available_indices, indices_test = create_datasets(data_dir, oracles, sent_type, BATCH_SIZE)
	train(train_loader, valid_loader, n_epochs=EPOCHS, batch_size=BATCH_SIZE)

