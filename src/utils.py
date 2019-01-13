import json
import pickle

import numpy as np
import pandas as pd

from math import ceil
from collections import defaultdict, Counter
from tqdm import tqdm
from matplotlib import pyplot as plt

import pystan
from convokit import Utterance, Corpus, User, Coordination, download

import spacy
from spacy.tokenizer import Tokenizer
tokenized = spacy.load('en')

    

def count(m, tokens):
    if m[-1] != '*':
        return tokens.count(m)
    else:
        m_count = 0
        for token in tokens:
            token = token.lower()
            if token[:len(m)-1] == m[:-1]:
                m_count += 1
        return m_count


def read_convokit_markers(filename):
    markers = {}
    lexemes = []

    with open(filename, 'r') as f:
        for line in f:
            category, pattern = line.strip().split("\t")
            category_markers = pattern.replace("\\b", "").split("|")
            
            markers[category] = category_markers
            lexemes.extend(category_markers)

    marker_words = list(set(lexemes))

    return markers, marker_words


def read_liwc_markers(directory_path, category_list):
    markers = {cat: [] for cat in category_list}
    lexemes = []

    for cat in category_list:
        with open(directory_path + cat + '.txt') as f:
            feature_lexemes = [word.rstrip('\n') for word in f.readlines()]
            markers[cat] = feature_lexemes
            lexemes.extend(feature_lexemes)

    marker_words = list(set(lexemes))

    return markers, marker_words


def standardise(x):
    if type(x) is list:
        x = np.array(x)

    return (x - np.mean(x)) / np.std(x)


def computeHDI(chains, interval = .95):
    """
    From
    stats.stackexchange.com/questions/252988/highest-density-interval-in-stan/253032
    """
    
    intervals = []

    if chains.ndim == 1:
        chain = chains[:]
        chain.sort()  # sort chain using the first axis which is the chain

        nSample = chain.size  # how many samples did you generate?  
        nSampleCred = int(ceil(nSample * interval))   # how many samples must go in the HDI?
        nCI = nSample - nSampleCred  # number of intervals to be compared

        # width of every proposed interval
        width = np.array([chain[i+nSampleCred] - chain[i] for  i in range(nCI)])

        best = width.argmin()  # index of lower bound of shortest interval (which is the HDI) 

        intervals.append([chain[best], chain[best + nSampleCred]])
        means = np.mean(chain)
    else:
        for idx in range(chains.shape[1]):
        
            chain = chains[:, idx]
            chain.sort()  # sort chain using the first axis which is the chain

            nSample = chain.size  # how many samples did you generate?  
            nSampleCred = int(ceil(nSample * interval))   # how many samples must go in the HDI?
            nCI = nSample - nSampleCred  # number of intervals to be compared

            # width of every proposed interval
            width = np.array([chain[i+nSampleCred] - chain[i] for i in range(nCI)])

            best = width.argmin()  # index of lower bound of shortest interval (which is the HDI) 

            intervals.append([chain[best], chain[best + nSampleCred]])
        means = np.mean(chains, axis=0)

    intervals = np.array(intervals).T

    intervals[0] = means - intervals[0]
    intervals[1] = intervals[1] - means

    return means, intervals


def make_chart(a_scores, b_scores, a_description, b_description, a_color="b", b_color="g", filename=None, title=None):
    """
    Plot two coordination scores against each other as a chart, both by coordination marker and on aggregate.
    """
    
    # get scores by marker and on aggregate
    _, a_score_by_marker, a_agg1, a_agg2, a_agg3 = coord.score_report(a_scores)
    _, b_score_by_marker, b_agg1, b_agg2, b_agg3 = coord.score_report(b_scores)

    # the rest plots this data as a double bar graph
    a_data_points = sorted(a_score_by_marker.items())
    b_data_points = sorted(b_score_by_marker.items())
    a_data_points, b_data_points = zip(*sorted(zip(a_data_points, b_data_points),
        key=lambda x: x[0][1], reverse=True))
    labels, a_data_points = zip(*a_data_points)
    _, b_data_points = zip(*b_data_points)

    labels = ["aggregate 1", "aggregate 2", "aggregate 3"] + list(labels)
    a_data_points = [a_agg1, a_agg2, a_agg3] + list(a_data_points)
    b_data_points = [b_agg1, b_agg2, b_agg3] + list(b_data_points)

    fig, ax = plt.subplots()
    ax.set_xticks(np.arange(len(a_data_points)) + 0.35)
    ax.set_xticklabels(labels, rotation="vertical")

    ax.bar(np.arange(len(a_data_points)), a_data_points, 0.35, color=a_color)
    ax.bar(np.arange(len(b_data_points)) + 0.35, b_data_points, 0.35, color=b_color)

#     a_scores_a1 = [s for s in a_scores if len(a_scores[s]) == 8]
#     b_scores_a1 = [s for s in b_scores if len(b_scores[s]) == 8]
    
    a_scores_a1 = [s for s in a_scores]
    b_scores_a1 = [s for s in b_scores]
    
    b_patch = mpatches.Patch(color="b",
                             label=a_description + " (total: " +
                             str(len(a_scores_a1)) + ", " +
                             str(len(a_scores)) + ")")
    g_patch = mpatches.Patch(color="g",
                             label=b_description + " (total: "  +
                             str(len(b_scores_a1)) + ", " +
                             str(len(b_scores)) + ")")
    plt.legend(handles=[b_patch, g_patch])
    
    plt.title(title)
    
    if filename:
        plt.savefig(filename)
    else:
        plt.show()


def plot_baseline_and_alignment(categories, 
                                base_means, align_means, 
                                base_intervals, align_intervals,
                                filename=None):

    fig, ax = plt.subplots(figsize=(25, 7))
    ax.scatter(categories, align_means)
    ax.errorbar(categories, align_means, align_intervals)
    ax.axhline(y=0, linestyle='--', color='black', linewidth=1)
    
    if filename:
        plt.savefig('{}-alignment.png'.format(filename))
    else:
        plt.show()
    # plt.savefig('wiki-{}-swam-alignment.png'.format(TOPIC))

    fig, ax = plt.subplots(figsize=(35, 7))
    ax.scatter(categories, base_means)
    ax.errorbar(categories, base_means, base_intervals)
    
    if filename:
        plt.savefig('{}-baseline.png'.format(filename))
    else:
        plt.show()