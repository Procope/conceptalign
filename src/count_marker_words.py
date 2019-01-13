import utils
import pickle

from talkpages import WikiCorpusReader, WikiCorpus
from alignment import Alignment
from matplotlib import pyplot as plt
from collections import defaultdict, Counter

import numpy as np
import networkx as nx


CNW = False

TOPICS = ['environment', 'history', 'linguistics',
              'people', 'philosophy', 'politics', 'psychiatry',
              'religion', 'science', 'sexuality', 'sports']


with open('../../data/liwc/final.dict', 'rb') as f:
    marker_dict = pickle.load(f)

if CNW:
	marker_words = []
	for cat in marker_dict:
		for m in marker_dict[cat]:
			marker_words.append(m)

	marker_words = list(set(marker_words))




for TOPIC in TOPICS:

    corpus = WikiCorpus('./with_counts/{}_wcounts.csv'.format(TOPIC))    

    if CNW:
    	corpus.count_marker_tokens_efficient(marker_words)
    	corpus.save('./with_counts/{}_cnw_counts.csv'.format(TOPIC))
    else:
    	corpus.count_marker_categories(marker_dict)
        corpus.save('./with_counts/{}_category_counts.csv'.format(TOPIC))
    
    