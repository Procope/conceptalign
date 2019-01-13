import pickle
import dill

import numpy as np
import pandas as pd
import pymc3 as pymc 

from collections import defaultdict, Counter

import utils
from talkpages import WikiCorpusReader, WikiCorpus
from alignment import Alignment


# CNW = False

TOPICS = ['sports', 'religion',
         'science', 
         'politics', 
         'history', 
         'people',
         'philosophy',
         'linguistics', 
         'psychiatry',
         'environment',
         'sexuality']

META_CATEGORIES = {'stylistic': [
                        'articles',
                        'negations',
                        'prepositions',
                        'numbers',
                        'pronouns'
                    ], 
                    'rhetoric': [
                        'tentative',   
                        'certainty',
                        'discrepancy',
                        'inclusive',
                        'exclusive'
                    ],
                    'discursive': [
                        'causation',
                        'insight',
                        'inhibition',
                        'communication',
                        'cognitive process',
                        'sensory process',
                        'motion'
                    ],
                    'stance': [
                        'optimism',
                        'anger',
                        'anxiety',
                        'sadness'
                    ]}


# Keep a list of category names for convenience.
CATEGORY_LIST = []
for cats in META_CATEGORIES.values():
    CATEGORY_LIST.extend(cats)


for CNW in [True, False]:
    # Load the filtered lists of markers. 
    with open('../../data/liwc/final.dict', 'rb') as f:
        MARKER_DICT = pickle.load(f)
        
        if CNW:
            marker_list = []
            for markers in MARKER_DICT.values():
                marker_list.extend(markers)
            MARKER_LIST = list(set(marker_list))



    for TOPIC in TOPICS:
        
        print('{}\n{}'.format(TOPIC, '*'*15))
        
        # Load dataframes with precomputed marker counts
        if CNW:
            csv_filename = './with_counts/{}_fullcounts.csv'.format(TOPIC)
        else:
            csv_filename = './with_counts/{}_category_counts.csv'.format(TOPIC)

        corpus = WikiCorpus(csv_filename)
        
        # Obtain dataframe of conversational turns
        turns = corpus.reply_pairs()
        
        # Generate network of TalkPages users 
        # (bool) prune: prune to the largest connected component?
        users = corpus.get_users()
        net = corpus.social_network(prune=False)
        
        # Compute centrality for each user and include into the dataframe of reply pairs
        corpus.assign_centrality('eigenvector')
        corpus.assign_centrality('betweenness')
        corpus.assign_tie_strength()
        
        # Initialise alignment tracker
        al = Alignment(corpus, MARKER_DICT)
        
        if CNW:
            # Compute category-not-word alignment counts (Doyle & Frank, 2016)
            N_base_all, N_align_all, C_base_all, C_align_all, dyad2eigen, dyad2betw, dyad2strength, dyad2admin = al.counts(mode='category-not-word', all_info=True)
            out_filename = './counts-cnw/{}.dill'.format(TOPIC)
        else:
            N_base_all, N_align_all, C_base_all, C_align_all, dyad2eigen, dyad2betw, dyad2strength, dyad2admin = al.counts(mode='categorical', all_info=True)
            out_filename = './counts-category/{}.dill'.format(TOPIC)

    with open(out_filename, 'wb') as f:
        dill.dump((N_base_all, N_align_all, C_base_all, C_align_all, dyad2eigen, dyad2betw, dyad2strength, dyad2admin), f)


