import utils

from talkpages import WikiCorpusReader, WikiCorpus
from alignment import Alignment

corpus_reader = WikiCorpusReader('../../data/controversial/')

topics = ['sports', 'law', 'linguistics', 'politics', 'sexuality', 'history', 
		  'people', 'science', 'religion', 'media', 'entertainment', 'environment',
		  'technology', 'philosophy', 'psychiatry', 'all']

# for topic in topics:
# 	print( corpus_reader.json_to_tsv('./', topic_list=[topic]) )





markers, marker_types = utils.read_convokit_markers('../coord-liwc-patterns.txt')
categories = list(markers.keys())

for topic in topics:
	print('*'*20, topic, '*'*20)

	tsv_filename = 'tsv/WikiControversial-{}.tsv'.format(topic)
	corpus = WikiCorpus(tsv_filename)

	corpus.tokenize_posts()
	# corpus.count_marker_categories(markers)
	# corpus.count_marker_tokens(marker_types)

	corpus.save('./tokenized_posts/{}_posts.csv'.format(topic))