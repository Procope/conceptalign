import json
import pickle
import warnings

import numpy as np
import pandas as pd

from math import ceil
from collections import defaultdict, Counter
from tqdm import tqdm
from matplotlib import pyplot as plt

import pystan
from convokit import Utterance, Corpus, User, Coordination, download

import utils

import spacy
from spacy.tokenizer import Tokenizer
tokenized = spacy.load('en')


class Alignment(object):
    """docstring for Alignment"""
    def __init__(self, corpus, markers, marker_tokens=None):
        super(Alignment, self).__init__()
        self.corpus = corpus
        self.marker_categories = markers
        self.marker_tokens = marker_tokens


    def counts(self, 
               mode='categorical',
               group_filter=lambda pair: True,
               centrality=None,
               tie_strength=False,
               all_info=False):

        assert mode in ['categorical', 'categorical_wide_baseline', 'category-not-word']

        pairs = self.corpus.reply_pairs()
        num_categories = len(self.marker_categories)

        N_base  = defaultdict(lambda: np.zeros(num_categories, dtype='int32'))  # N_base_ab_m
        N_align = defaultdict(lambda: np.zeros(num_categories, dtype='int32'))  # N_align_ab_m
        C_base  = defaultdict(lambda: np.zeros(num_categories, dtype='int32'))  # C_base_ab_m
        C_align = defaultdict(lambda: np.zeros(num_categories, dtype='int32'))  # C_align_ab_m

        if all_info:
            eigenvector_dict = {}
            betweenness_dict = {}
            strength_dict = {}
            admin_dict = {}
        elif centrality:
            centrality_dict = {}
        elif tie_strength:
            strength_dict = {}


        for _, pair in tqdm(pairs.iterrows(), total=len(pairs)):

            if pair['author_name_a'] != pair['target_name_b']:
                continue  # 'a != parent(b)'
            if pair['target_admin_b'] != pair['author_admin_a']:
                continue  # 'admin(a) != admin(parent(b))'
            
            if not group_filter(pair):
                continue


            if mode == 'categorical':
                if centrality:
                    N_base, N_align, C_base, C_align, centrality_dict = self.category_counts_with_centrality(pair, 
                        centrality, centrality_dict, N_base, N_align, C_base, C_align)
                if all_info:
                    N_base, N_align, C_base, C_align, eigenvector_dict, betweenness_dict, strength_dict, admin_dict = self.category_counts_allinfo(pair, 
                        eigenvector_dict, betweenness_dict, strength_dict, admin_dict, N_base, N_align, C_base, C_align)
                else:
                    N_base, N_align, C_base, C_align = self.category_counts(pair, 
                        N_base, N_align, C_base, C_align)

            elif mode == 'category-not-word':
                if centrality:
                    N_base, N_align, C_base, C_align, centrality_dict = self.category_not_word_counts_with_centrality(pair, 
                        centrality, centrality_dict, N_base, N_align, C_base, C_align)

                if all_info:
                    N_base, N_align, C_base, C_align, eigenvector_dict, betweenness_dict, strength_dict, admin_dict = self.category_not_word_counts_allinfo(pair, 
                        eigenvector_dict, betweenness_dict, strength_dict, admin_dict, N_base, N_align, C_base, C_align)

                else:
                    N_base, N_align, C_base, C_align = self.category_not_word_counts(pair, 
                        N_base, N_align, C_base, C_align)

            elif mode == 'categorical_wide_baseline':
                if centrality:
                    raise NotImplementedError()

                N_base, N_align, C_base, C_align = self.category_counts_overall_baseline(pair, 
                    N_base, N_align, C_base, C_align)
                
                N_base_tmp, C_base_tmp = {}, {}
                for (a, b) in N_align:
                    N_base_tmp[(a, b)] = N_base[b]
                    C_base_tmp[(a, b)] = C_base[b]
                N_base, C_base = N_base_tmp, C_base_tmp
            #
            # else: change initial assertion for new modes
        if all_info:
            return N_base, N_align, C_base, C_align, eigenvector_dict, betweenness_dict, strength_dict, admin_dict
        elif centrality:
            return N_base, N_align, C_base, C_align, centrality_dict
        elif tie_strength:
            return N_base, N_align, C_base, C_align, strength_dict
        else:
            return N_base, N_align, C_base, C_align


    def category_counts(self, pair, N_base, N_align, C_base, C_align):
        # The dyad
        a, b = pair['author_name_a'], pair['author_name_b']
        # Total token count in reply
        num_tokens = int(pair['n_tokens_b'])

        for cat_idx, cat in enumerate(self.marker_categories):
            marker_count_b = pair[cat + '_b']
            marker_in_a    = pair[cat + '_a'] > 0
            
            N_base [(a,b)][cat_idx] += (not marker_in_a) * num_tokens
            C_base [(a,b)][cat_idx] += (not marker_in_a) * marker_count_b
            
            N_align[(a,b)][cat_idx] += marker_in_a * num_tokens
            C_align[(a,b)][cat_idx] += marker_in_a * marker_count_b

        return N_base, N_align, C_base, C_align


    def category_counts_allinfo(self,
                                pair,
                                eigenvector_dict,
                                betweenness_dict,
                                tie_strength_dict,
                                admin_dict,
                                N_base,
                                N_align,
                                C_base,
                                C_align):

        # The dyad
        a, b = pair['author_name_a'], pair['author_name_b']
        # Total token count in reply
        num_tokens = int(pair['n_tokens_b'])

        try:
            tie_strength_dict[(a,b)]  
        except KeyError:
            tie_strength_dict[(a,b)] = pair['tie_strength']

        try:
            eigenvector_dict[(a,b)]  
        except KeyError:
            eigenvector_dict[(a,b)] = (pair['eigenvector_a'], pair['eigenvector_b'])

        try:
            betweenness_dict[(a,b)]  
        except KeyError:
            betweenness_dict[(a,b)] = (pair['betweenness_a'], pair['betweenness_b'])

        try:
            admin_dict[(a,b)]  
        except KeyError:
            admin_dict[(a,b)] = (pair['author_admin_a'], pair['author_admin_b'])
        

        for cat_idx, cat in enumerate(self.marker_categories):
            marker_count_b = pair[cat + '_b']
            marker_in_a    = pair[cat + '_a'] > 0
            
            N_base [(a,b)][cat_idx] += (not marker_in_a) * num_tokens
            C_base [(a,b)][cat_idx] += (not marker_in_a) * marker_count_b
            
            N_align[(a,b)][cat_idx] += marker_in_a * num_tokens
            C_align[(a,b)][cat_idx] += marker_in_a * marker_count_b

        return N_base, N_align, C_base, C_align, eigenvector_dict, betweenness_dict, tie_strength_dict, admin_dict


    def category_counts_with_centrality(self, 
                                        pair,
                                        centrality,
                                        centrality_dict,
                                        N_base,
                                        N_align,
                                        C_base,
                                        C_align):

        
        # The dyad
        a, b = pair['author_name_a'], pair['author_name_b']
        # Total token count in reply
        num_tokens = int(pair['n_tokens_b'])

        try:
            centrality_dict[(a,b)]  
        except KeyError:
            centrality_dict[(a,b)] = (pair[centrality + '_a'], pair[centrality + '_b'])

        for cat_idx, cat in enumerate(self.marker_categories):
            marker_count_b = pair[cat + '_b']
            marker_in_a    = pair[cat + '_a'] > 0
            
            N_base [(a,b)][cat_idx] += (not marker_in_a) * num_tokens
            C_base [(a,b)][cat_idx] += (not marker_in_a) * marker_count_b
            
            N_align[(a,b)][cat_idx] += marker_in_a * num_tokens
            C_align[(a,b)][cat_idx] += marker_in_a * marker_count_b


        return N_base, N_align, C_base, C_align, centrality_dict



    def category_counts_overall_baseline(self, pair, N_base, N_align, C_base, C_align):
        # The dyad
        a, b = pair['author_name_a'], pair['author_name_b']
        # Total token count in reply
        num_tokens = int(pair['n_tokens_b'])


        for cat_idx, cat in enumerate(self.marker_categories):
            marker_count_b = pair[cat + '_b']
            marker_in_a    = pair[cat + '_a'] > 0
            
            try:
                N_base [b]
            except KeyError:
                N_base [b] = np.zeros(len(self.marker_categories), dtype='int32')
                C_base [b] = np.zeros(len(self.marker_categories), dtype='int32')
            
            N_base [b][cat_idx] += (not marker_in_a) * num_tokens
            C_base [b][cat_idx] += (not marker_in_a) * marker_count_b

            N_align[(a,b)][cat_idx] += marker_in_a * num_tokens
            C_align[(a,b)][cat_idx] += marker_in_a * marker_count_b

        return N_base, N_align, C_base, C_align


    def category_not_word_counts(self, pair, N_base, N_align, C_base, C_align):
        # The dyad
        a, b = pair['author_name_a'], pair['author_name_b']
        # Total token count in reply
        num_tokens = int(pair['n_tokens_b'])

        for cat_idx, cat in enumerate(self.marker_categories):
            
            tot_marker_count = 0
            marker_in_a = []
            
            for m in self.marker_categories[cat]:
                tot_marker_count += pair[m + '_b']
                marker_in_a.append(pair[m + '_a'] > 0)
            
            baseline_case  = tot_marker_count == 0
            alignment_case = sum(marker_in_a) == 1
            
            # not C
            if baseline_case:
                N_base [(a,b)][cat_idx] += num_tokens
                C_base [(a,b)][cat_idx] += tot_marker_count
            
            # w and not C \ w    
            elif alignment_case:
                w = self.marker_categories[cat][np.argmax(marker_in_a)]
                cat_alignment_count = tot_marker_count - pair[w + '_b']
                
                N_align[(a,b)][cat_idx] += num_tokens
                C_align[(a,b)][cat_idx] += cat_alignment_count
            
            # else: continue  -->  multiple w's from C in target

        return N_base, N_align, C_base, C_align


    def category_not_word_counts_with_centrality(self, 
                                                 pair,
                                                 centrality,
                                                 centrality_dict,
                                                 N_base,
                                                 N_align,
                                                 C_base,
                                                 C_align):

        # The dyad
        a, b = pair['author_name_a'], pair['author_name_b']
        # Total token count in reply
        num_tokens = int(pair['n_tokens_b'])

        try:
            centrality_dict[(a,b)]  
        except KeyError:
            centrality_dict[(a,b)] = (pair[centrality + '_a'], pair[centrality + '_b'])

        for cat_idx, cat in enumerate(self.marker_categories):
            
            tot_marker_count = 0
            marker_in_a = []
            
            for m in self.marker_categories[cat]:
                tot_marker_count += pair[m + '_b']
                marker_in_a.append(pair[m + '_a'] > 0)
            
            baseline_case  = tot_marker_count == 0
            alignment_case = sum(marker_in_a) == 1
            
            # not C
            if baseline_case:
                N_base [(a,b)][cat_idx] += num_tokens
                C_base [(a,b)][cat_idx] += tot_marker_count
            
            # w and not C \ w    
            elif alignment_case:
                w = self.marker_categories[cat][np.argmax(marker_in_a)]
                cat_alignment_count = tot_marker_count - pair[w + '_b']
                
                N_align[(a,b)][cat_idx] += num_tokens
                C_align[(a,b)][cat_idx] += cat_alignment_count
            
            # else: continue  -->  multiple w's from C in target

        return N_base, N_align, C_base, C_align, centrality_dict

    def category_not_word_counts_allinfo(self,
                                        pair,
                                        eigenvector_dict,
                                        betweenness_dict,
                                        tie_strength_dict,
                                        admin_dict,
                                        N_base,
                                        N_align,
                                        C_base,
                                        C_align):
        
        # The dyad
        a, b = pair['author_name_a'], pair['author_name_b']
        # Total token count in reply
        num_tokens = int(pair['n_tokens_b'])

        try:
            tie_strength_dict[(a,b)]  
        except KeyError:
            tie_strength_dict[(a,b)] = pair['tie_strength']

        try:
            eigenvector_dict[(a,b)]  
        except KeyError:
            eigenvector_dict[(a,b)] = (pair['eigenvector_a'], pair['eigenvector_b'])

        try:
            betweenness_dict[(a,b)]  
        except KeyError:
            betweenness_dict[(a,b)] = (pair['betweenness_a'], pair['betweenness_b'])

        try:
            admin_dict[(a,b)]  
        except KeyError:
            admin_dict[(a,b)] = (pair['author_admin_a'], pair['author_admin_b'])

        for cat_idx, cat in enumerate(self.marker_categories):
            
            tot_marker_count = 0
            marker_in_a = []
            
            for m in self.marker_categories[cat]:
                tot_marker_count += pair[m + '_b']
                marker_in_a.append(pair[m + '_a'] > 0)
            
            baseline_case  = sum(marker_in_a) == 0
            alignment_case = sum(marker_in_a) == 1
            
            # not C
            if baseline_case:
                N_base [(a,b)][cat_idx] += num_tokens  # int(pair['n_tokens_b'])
                C_base [(a,b)][cat_idx] += tot_marker_count
            
            # w and not C \ w    
            elif alignment_case:
                w = self.marker_categories[cat][np.argmax(marker_in_a)]
                
                N_align[(a,b)][cat_idx] += num_tokens
                C_align[(a,b)][cat_idx] += tot_marker_count - pair[w + '_b']
            
            # else: continue  -->  multiple w's from C in target

        return N_base, N_align, C_base, C_align, eigenvector_dict, betweenness_dict, tie_strength_dict, admin_dict

    def category_not_word_counts_with_tie_strength(self, 
                                                 pair,
                                                 tie_strength_dict,
                                                 N_base,
                                                 N_align,
                                                 C_base,
                                                 C_align):

        # The dyad
        a, b = pair['author_name_a'], pair['author_name_b']
        # Total token count in reply
        num_tokens = int(pair['n_tokens_b'])

        try:
            tie_strength_dict[(a,b)]  
        except KeyError:
            tie_strength_dict[(a,b)] = pair['tie_strength']


        for cat_idx, cat in enumerate(self.marker_categories):
            
            tot_marker_count = 0
            marker_in_a = []
            
            for m in self.marker_categories[cat]:
                tot_marker_count += pair[m + '_b']
                marker_in_a.append(pair[m + '_a'] > 0)
            
            baseline_case  = tot_marker_count == 0
            alignment_case = sum(marker_in_a) == 1
            
            # not C
            if baseline_case:
                N_base [(a,b)][cat_idx] += num_tokens
                C_base [(a,b)][cat_idx] += tot_marker_count
            
            # w and not C \ w    
            elif alignment_case:
                w = self.marker_categories[cat][np.argmax(marker_in_a)]
                cat_alignment_count = tot_marker_count - pair[w + '_b']
                
                N_align[(a,b)][cat_idx] += num_tokens
                C_align[(a,b)][cat_idx] += cat_alignment_count
            
            # else: continue  -->  multiple w's from C in target

        return N_base, N_align, C_base, C_align, tie_strength_dict



    def swam(self, N_base, N_align, C_base, C_align, verbose=True):

        num_categories = len(self.marker_categories)

        eta_align_means = np.zeros((num_categories))
        eta_align_intervals = np.zeros((2, num_categories))

        eta_base_means = np.zeros((num_categories))
        eta_base_intervals = np.zeros((2, num_categories))

        for cat, category in enumerate(self.marker_categories):
        # for cat, category in enumerate(list(self.marker_categories.keys())[:1]):

            N_b = [N_base[dyad][cat] for dyad in N_base]
            N_a = [N_align[dyad][cat] for dyad in N_align]
            C_b = [C_base[dyad][cat] for dyad in C_base]
            C_a = [C_align[dyad][cat] for dyad in C_align]

            data = {
                "NumObservations": len(N_b),
                "NumUtterancesAB": N_a,
                "NumUtterancesNotAB": N_b,
                "CountsAB": C_a,
                "CountsNotAB": C_b,
                "StdDev": .25
            }

            sm = pystan.StanModel(file='../swam.stan', verbose=True)
            fit = sm.sampling(data=data, iter=500, chains=4, pars=['eta_ab_pop', 'eta_pop'])

            if verbose:
                print(fit)

            eta_align_vec = fit.extract(pars='eta_ab_pop', permuted=True)['eta_ab_pop']
            eta_base_vec = fit.extract(pars='eta_pop', permuted=True)['eta_pop']

            eta_align_mean, hdi_align_interval = utils.computeHDI(eta_align_vec)
            eta_base_mean, hdi_base_interval = utils.computeHDI(eta_base_vec)

            eta_align_means[cat] = eta_align_mean
            eta_base_means[cat] = eta_base_mean

            eta_align_intervals[:, cat] = hdi_align_interval[:, 0]
            eta_base_intervals[:, cat] = hdi_base_interval[:, 0]

        return (eta_base_means, eta_align_means), (eta_base_intervals, eta_align_intervals)


    def swam2(self, N_base, N_align, C_base, C_align, verbose=True):

        num_categories = len(self.marker_categories)

        eta_align_means = np.zeros((num_categories))
        eta_align_intervals = np.zeros((2, num_categories))

        eta_base_means = np.zeros((num_categories))
        eta_base_intervals = np.zeros((2, num_categories))

        for cat, category in enumerate(self.marker_categories):
        # for cat, category in enumerate(list(self.marker_categories.keys())[:1]):

            N_b = [N_base[dyad][cat] for dyad in N_base]
            N_a = [N_align[dyad][cat] for dyad in N_align]
            C_b = [C_base[dyad][cat] for dyad in C_base]
            C_a = [C_align[dyad][cat] for dyad in C_align]

            data = {
                "NumObservations": len(N_b),
                "NumUtterancesAB": N_a,
                "NumUtterancesNotAB": N_b,
                "CountsAB": C_a,
                "CountsNotAB": C_b,
                "StdDev": .25
            }

            sm = pystan.StanModel(file='../swam2.stan', verbose=True)
            fit = sm.sampling(data=data, iter=500, chains=4, pars=['eta_ab_pop', 'eta_pop'])

            if verbose:
                print(fit)

            eta_align_vec = fit.extract(pars='eta_ab_pop', permuted=True)['eta_ab_pop']
            eta_base_vec = fit.extract(pars='eta_pop', permuted=True)['eta_pop']

            eta_align_mean, hdi_align_interval = utils.computeHDI(eta_align_vec)
            eta_base_mean, hdi_base_interval = utils.computeHDI(eta_base_vec)

            eta_align_means[cat] = eta_align_mean
            eta_base_means[cat] = eta_base_mean

            eta_align_intervals[:, cat] = hdi_align_interval[:, 0]
            eta_base_intervals[:, cat] = hdi_base_interval[:, 0]

        return (eta_base_means, eta_align_means), (eta_base_intervals, eta_align_intervals)


    def wham(self, N_base, N_align, C_base, C_align, verbose=True):
        marker_type = []
        N_b, N_a = [], []
        C_b, C_a = [], []

        n_dyads = len(N_base.keys())
        num_categories = len(self.marker_categories)

        for cat, category in enumerate(self.marker_categories):
            N_b += [ N_base[dyad][cat] for dyad in N_base]
            N_a += [N_align[dyad][cat] for dyad in N_align]
            C_b += [ C_base[dyad][cat] for dyad in C_base]
            C_a += [C_align[dyad][cat] for dyad in C_align]
            marker_type += [cat + 1] * n_dyads  # +1 to meet pystan's taste

        data = {
            "NumMarkers": num_categories,
            "MarkerType": marker_type,
            "NumObservations": len(marker_type),
            "NumUtterancesAB": N_a,
            "NumUtterancesNotAB": N_b,
            "CountsAB": C_a,
            "CountsNotAB": C_b,
            "StdDev": .25
        }

        sm = pystan.StanModel(file='../alignment.cauchy.nosubpop.stan', verbose=True)
        fit = sm.sampling(data=data, iter=500, chains=4, pars=['eta_ab_pop', 'eta_pop'])

        if verbose:
            print(fit)

        eta_align_vecs = fit.extract(pars='eta_ab_pop', permuted=True)['eta_ab_pop']
        eta_base_vecs = fit.extract(pars='eta_pop', permuted=True)['eta_pop']

        eta_align_means, eta_align_intervals = utils.computeHDI(eta_align_vecs)
        eta_base_means, eta_base_intervals = utils.computeHDI(eta_base_vecs)

        return (eta_base_means, eta_align_means), (eta_base_intervals, eta_align_intervals)