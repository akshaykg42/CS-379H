from utils import *
from model import *
from train import *
from test import *
from sklearn.model_selection import train_test_split

sent_type = 0

if __name__ == '__main__':
	print('Loading data...')
	X, y = load('pcr_documents.pkl', 'pcr_summaries.pkl', 'pcr_oracles.pkl', sent_type)
	X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(X, y, np.arange(len(X)))
	train(X_train, y_train)
	test(X_test, y_test)
