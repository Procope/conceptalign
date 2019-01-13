import json
import pickle
import warnings
import os

import numpy as np
import pandas as pd
import networkx as nx

from math import ceil
from collections import defaultdict, Counter
from tqdm import tqdm
from matplotlib import pyplot as plt

import pystan
from convokit import Utterance, Corpus, User, Coordination, download

from utils import *
from copy import copy

import spacy
from spacy.tokenizer import Tokenizer
tokenized = spacy.load('en')


class WikiCorpusReader(object):
    """
    Reader object for the Controversial TalkPages Corpus.
    """
    def __init__(self, directory_path):
        super(WikiCorpusReader, self).__init__()

        self.directory_path = directory_path
        self.topics_path = directory_path + 'ControversialTopics.txt'
        self.threads_path = directory_path + 'WikipediaControversial.json'

        self.FIELDS = ['topics', 'forum', 'author_name', 'target_name', 'utt_id',
                       'target_id', 'author_admin', 'target_admin', 
                       'author_registered', 'author_gender', 'text']

        self.ROOT = '<<ROOT>>'
        self.TOPIC_SEPARATOR = '|||'
        
        with open(self.topics_path, 'r') as f:
          lines = f.readlines()
        
        self.forum_to_topic = defaultdict(list)
        for line in tqdm(lines):
            
            line = line.strip()
            if not line:
                continue

            # Antisemitism    Politics and economics|||History|||People
            forum_and_topics = line.split()
            forum = forum_and_topics[0].replace('_', ' ')
            topics_tmp = forum_and_topics[1].split(self.TOPIC_SEPARATOR)

            for t in topics_tmp:
                if t[-1] == ',': t = t[:-1]
                self.forum_to_topic[forum].append(t.lower())




    def json_to_tsv(self, tsv_path, topic_list=None):
    
        print('Loading threads from original json file...')

        with open(self.threads_path, 'r') as f:
            threads = json.loads('\n'.join(f.readlines()))
            print('{} threads loaded.'.format(len(threads)))
        

        print('Process threads.')
        
        utterances = []
        for post_dict in tqdm(threads):

            forum = post_dict["forum"]
            forum_name = forum[5:].split('/Archive')[0]

            try:
                topics = self.forum_to_topic[forum_name]
            except KeyError:
                continue

            if topic_list:
                relevant_topics = set(topic_list) & set(topics)
                if not relevant_topics:
                    continue
            else:
                relevant_topics = topics
            
            # (speaker_name, is_speaker_admin)
            previous_speakers = defaultdict(lambda: [(self.ROOT, -1, False)])  

            for comment_dict in post_dict["z_comments"]:

                comment_id = comment_dict["comment_id"]  # int
                author_name = comment_dict["user_id"] # string
                text = comment_dict["comment"]
                author_registered = comment_dict["registered"]  # boolean

                try:
                    author_gender = comment_dict["gender"]  # string
                except KeyError:
                    author_gender = 'unknown'
                try:
                    is_author_admin = comment_dict["admin_post"]  # boolean
                except KeyError:
                    is_author_admin = False
            
                
                for topic in relevant_topics:
                    previous_speaker = previous_speakers[topic][comment_dict["parent_id"]]
                    target_name, target_utt_id, is_target_admin = previous_speaker
                    
                    utterance = '\t'.join([topic, 
                                          forum_name,
                                          author_name, target_name,
                                          str(len(utterances)), str(target_utt_id),
                                          str(is_author_admin), str(is_target_admin),
                                          str(author_registered), author_gender, 
                                          text])
                    
                    previous_speakers[topic].append((author_name, len(utterances), is_author_admin))
                    utterances.append(utterance)

        print('{} valid utterances found.'.format(len(utterances)))
        

        if topic_list:
            tsv_filepath = '{}WikiControversial-{}.tsv'.format(tsv_path, ':'.join(topic_list))
        else:
            tsv_filepath = '{}WikiControversial-all.tsv'.format(tsv_path)   
            
    
        with open(tsv_filepath, 'w') as f:
            for utterance in tqdm(utterances):
                f.write("{}\n".format(utterance))
        print('Utterances written to tab-separated file {}'.format(tsv_filepath))

        return tsv_filepath


class WikiCorpus(object):
    """docstring for WikiCorpus"""
    def __init__(self, filename):
        super(WikiCorpus, self).__init__()
        
        self.FIELDS = ['topics', 'forum', 'author_name', 'target_name', 'utt_id',
                       'target_id', 'author_admin', 'target_admin', 
                       'author_registered', 'author_gender', 'text']

        # self.ROOT = '<<ROOT>>'
        if filename[-3:] == 'tsv':
            self.posts = pd.read_csv(filename, sep='\t', names=self.FIELDS)
            # try:
            #     os.remove(filename)
            # except OSError:
            #     pass

        elif filename[-3:] == 'csv':
            self.posts = pd.read_csv(filename, header=0, index_col=0)
        else:
            raise ValueError('Use a tsv / csv file.')

        try:
            self.posts.set_index('utt_id', inplace=True)
        except:
            pass

        self.pairs = None
        self.users = None
        self.network = None
        

    def tokenize_posts(self):
        """ 
        Add a 'tokens' column to the posts dataframe.
        """
        if 'tokens' in self.posts:
            warnings.warn("Posts are already tokenized. Skipping tokenization.")
            return 
        
        tokens_lens_types = []
        for text in tqdm(self.posts['text'], desc="Tokenizing posts."):
            toks = [t.text.lower() for t in tokenized(str(text))]
            tokens_lens_types.append((toks, len(toks), len(set(toks))))
        
        tokens, lengths, types = zip(*tokens_lens_types)
        self.posts = self.posts.assign(tokens=tokens)
        self.posts = self.posts.assign(n_tokens=lengths)
        self.posts = self.posts.assign(n_types=types)

        n = len(self.posts)
        self.posts = self.posts[self.posts.n_tokens > 0]
        
        print("Filtered {} posts with 0-length utterances".format(n - len(self.posts)))
        return


    def count_marker_categories(self, markers):
        """ Add feature columns for marker counts.
        Markers is a dictionary where the keys are the marker types and
        the values are a list of markers.
        """
        if not 'tokens' in self.posts:
            raise ValueError("Corpus must be tokenized for marker detection.")

        if all(m in self.posts for m in markers):
            warnings.warn("All marker columns already exist. Skipping marker detection.")
            return
        
        for m in markers: 
            counts = []
            for tokens in tqdm(self.posts['tokens'], desc="Detecting {}.".format(m)):
                m_lemmas = markers[m]
                m_cnt = 0
                for m_lemma in m_lemmas:
                    m_cnt += count(m_lemma, tokens)
                counts.append(m_cnt)
            self.posts[m] = counts        


    def count_marker_categories_efficient(self, markers):
        """ Add feature columns for marker counts.
        Markers is a dictionary where the keys are the marker types and
        the values are a list of markers.
        """
        if not 'tokens' in self.posts:
            raise ValueError("Corpus must be tokenized for marker detection.")

        if all(m in self.posts for m in markers):
            warnings.warn("All marker columns already exist. Skipping marker detection.")
            return
        

        counts_tmp = {}
        m2m = {}
        m2cat = {}
        for cat, marker_words in markers.items():
            for m in marker_words:
                
                if m[-1] == '*':
                    # counts_tmp[m[:-1]] = 0
                    m2m[m] = m[-1]

                m2cat[m] = cat
            counts_tmp[cat] = 0

        counts = []
        for i in range(len(self.posts['tokens'])):
            counts.append(dict(counts_tmp))
            # counts.append({k:v for k,v in counts_tmp.items()})

        
        for i, tokens in enumerate(tqdm(self.posts['tokens'], desc="Detecting markers.")):
            for token in tokens:
                try:
                    counts[i][m2cat[token]] += 1
                except KeyError:
                    for m, m_ in m2m.items():  # contains *
                        if token.startswith(m_):
                            counts[i][m2cat[m]] += 1

        for cat in markers.keys():
            self.posts[cat] = [count[cat] for count in counts]          


    def count_marker_tokens(self, marker_words):
        """ Add feature columns for marker counts.
        Markers is a list of word markers.
        """

        if not 'tokens' in self.posts:
            raise ValueError("Corpus must be tokenized for marker detection.")

        if all(m in self.posts for m in marker_words):
            warnings.warn("All marker columns already exist. Skipping marker detection.")
            return
        
        for m in marker_words:
            counts = []
            for tokens in tqdm(self.posts['tokens'], desc="Detecting {}.".format(m)):
                counts.append(count(m, tokens))
        
            self.posts[m] = counts


    def count_marker_tokens_efficient(self, marker_words):
        """ Add feature columns for marker counts.
        Markers is a list of markers.
        """

        if not 'tokens' in self.posts:
            raise ValueError("Corpus must be tokenized for marker detection.")

        if all(m in self.posts for m in marker_words):
            warnings.warn("All marker columns already exist. Skipping marker detection.")
            return

        
        counts_tmp = {}
        m2m = {}
        for m in marker_words:
            if m[-1] == '*':
                # counts_tmp[m[:-1]] = 0
                m2m[m] = m[-1]
            counts_tmp[m] = 0

        counts = []
        for i in range(len(self.posts['tokens'])):
            counts.append({k:v for k,v in counts_tmp.items()})

        
        for i, tokens in enumerate(tqdm(self.posts['tokens'], desc="Detecting markers.")):
            for token in tokens:
                try:
                    counts[i][token] += 1
                except KeyError:

                    for m, m_ in m2m.items():  # contains *

                        if token.startswith(m_):

                            counts[i][m] += 1

        for m in marker_words:
            self.posts[m] = [count[m] for count in counts]                   


    def save(self, filename):
        self.posts.to_csv(filename)


    def reply_pairs(self, suffixes=['_a', '_b'], filter_self_replies=True):
        """ 
        View the posts dataframe as reply pairs 
        """
        if self.pairs is not None:
            return self.pairs

        pairs = pd.merge(self.posts, self.posts, 
                         how='inner', left_index=True, 
                         right_on='target_id', left_on='utt_id', 
                         suffixes=suffixes)
        
        if filter_self_replies:
            self_replies = (pairs['author_name' + suffixes[0]] != pairs['author_name' + suffixes[1]])
            pairs = pairs[self_replies]
            
        self.pairs = pairs

        return self.pairs


    def social_network(self, prune=False):

        if self.network is not None:
            return self.network

        if self.pairs is None:
            print("The list of reply pairs has not been constructed yet.")
            print('Build reply pairs.')
            self.reply_pairs()

        print('Build network.')

        self.network = nx.Graph()
        for i, pair in tqdm(self.pairs.iterrows(), total=len(self.pairs)):
            user_a, user_b = pair['author_name_a'], pair['author_name_b']
            
            # ignore self-talk
            if user_a == user_b or '' in (user_a, user_b):
                continue 
                
            if self.network.has_edge(user_a, user_b):
                self.network[user_a][user_b]['weight'] += 1
            else:
                self.network.add_edge(user_a, user_b, weight=1)

            if self.network.has_edge(user_b, user_a):
                self.network[user_b][user_a]['weight'] += 1
            else:
                self.network.add_edge(user_b, user_a, weight=1)
                
        print("The unpruned network has {} nodes (users).".format(len(self.network.nodes())))

        if prune:
            # pruning the network to it's largest component:
            minor_components = list(nx.connected_components(self.network))[1:]
            disconnected_users = [user for com in minor_components for user in com]
            self.network.remove_nodes_from(disconnected_users)

            print("Removed {} users from {} disconnected components.".format(
                len(disconnected_users), len(minor_components)))

        return self.network


    def assign_tie_strength(self):

        tie_strengths = []
        for _, pair in self.pairs.iterrows():
            user_a, user_b = pair['author_name_a'], pair['author_name_b']
            tie_strengths.append(self.network[user_a][user_b]['weight'])

        tot_exchanges = np.sum(tie_strengths)
        tie_strengths = [x / tot_exchanges for x in tie_strengths] 

        self.pairs['tie_strength'] = tie_strengths
        print('Tie strength information has been assigned to all pairs.')


    def assign_centrality(self, centrality='eigenvector'):
        if centrality == 'eigenvector':
            centrality_scores = nx.eigenvector_centrality_numpy(self.network)
        elif centrality == 'betweenness':
            centrality_scores = nx.eigenvector_centrality_numpy(self.network)
        else:
            raise ValueError('Choose "eigenvector" or "betweenness" centrality.')

        # centrality_scores = pd.DataFrame(list(centrality_scores.items()), 
        #                                  columns=['author_name', centrality])

        
        # centrality_scores.set_index('author_name', inplace=True)

        # self.users[centrality] = centrality_scores

        centrality_values = []
        for i, pair in self.pairs.iterrows():
            try:
                centrality_a = centrality_scores[pair['author_name_a']]
                centrality_b = centrality_scores[pair['author_name_b']]
            except KeyError:
                centrality_a, centrality_b = (-1, -1)

            centrality_values.append((centrality_a, centrality_b))

        a_values, b_values = zip(*centrality_values)
        self.pairs[centrality + '_a'] = a_values
        self.pairs[centrality + '_b'] = b_values
        print('Centrality information has been assigned to all pairs.')


    def get_users(self):

        if self.users is not None:
            return self.users

        users = {'author_name': [],
                 'author_admin': [],
                 'author_gender': []}

        seen = {}

        for _, pair in tqdm(self.pairs.iterrows(), total=len(self.pairs)):
            for suffix in ['_a', '_b']:
                try:
                    seen[pair['author_name' + suffix]]
                except KeyError:
                    for field, values in users.items():
                        users[field].append(pair[field + suffix])

        self.users = pd.DataFrame(data=users, columns=list(users.keys()))
        self.users.set_index('author_name', inplace=True)

        return self.users