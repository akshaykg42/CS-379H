from utils import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

def tokenize_only(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens

tfidf_vectorizer = TfidfVectorizer(max_df=0.95, max_features=10000,
                                 min_df=2, stop_words='english',
                                 use_idf=True, lowercase=True, 
                                 tokenizer=tokenize_only, ngram_range=(1,3))

summary_sentences = pickle.load(open('earthquake_summary_sentences.pkl', 'rb'))
summaries = pickle.load(open('earthquake_data/raw/summaries.pkl', 'rb'))
documents = pickle.load(open('earthquake_data/raw/documents.pkl', 'rb'))
oracles = pickle.load(open('earthquake_data/raw/oracles.pkl', 'rb'))

tfidf_matrix = tfidf_vectorizer.fit_transform(summary_sentences)

type_samples = [[4, 12, 1012, 1014, 897, 795, 770, 1408, 754, 808, 819, 900, 909, 952, 1215],[1423, 1521, 1633, 1649, 1705, 1821, 1909, 49, 171],[377, 401, 512, 536, 624, 889, 1396, 1518, 552, 1312, 592],[839, 892, 1170, 1175, 1242, 1279, 1287, 1314, 1458, 1482, 1642, 1680, 1475, 1670, 1691, 1889],[1065, 1234, 1372, 1532, 1872, 334, 1147, 1253, 1491, 1679],[1822, 1846, 1511, 1631, 1838, 1910, 41, 348, 388, 415, 546, 646, 670, 419, 787, 1560, 1704, 138, 9, 81],[686, 674, 721, 783, 872, 874, 1124, 1132, 1138, 1194, 1332, 1338, 1340, 167, 237, 427]]

type_seed_centroids = []
for type_sample in type_samples:
	type_vectors = [tfidf_matrix[i] for i in type_sample]
	type_vectors = np.array([i.toarray() for i in type_vectors])
	type_vectors = type_vectors.reshape(type_vectors.shape[0], type_vectors.shape[2])
	km = KMeans(n_clusters=1)
	km.fit(type_vectors)
	seed_centroid = km.cluster_centers_[0]
	type_seed_centroids.append(seed_centroid)

type_seed_centroids = np.array(type_seed_centroids)
km = KMeans(n_clusters=len(type_samples), init=type_seed_centroids)
km.fit(tfidf_matrix)

sents_to_types = {}

sents_by_type = []

for i in range(len(type_samples)):
	indices = [j for j, k in enumerate(km.labels_) if k == i]
	sentences = []
	for index in indices:
		sents_to_types[summary_sentences[index]] = i
		sentences.append(summary_sentences[index])
	sents_by_type.append(sentences)

for i in range(len(sents_by_type)):
	with open('quakes type {}.txt'.format(i), 'w') as f:
		f.write('\n\n'.join(sents_by_type[i]))

summaries_as_types = [[[sents_to_types[sent]] for sent in summary] for summary in summaries]

pickle.dump(summaries_as_types, open('earthquake_data/raw/types.pkl', 'wb'))

rouges = [rouge.get_scores(' '.join(summaries[i]), ' '.join([documents[i][j] for j in oracles[i]]))[0]['rouge-1']['f'] for i in range(len(oracles))]