import utils

from talkpages import WikiCorpusReader, WikiCorpus
from alignment import Alignment

TOPIC = 'sports'

corpus_reader = WikiCorpusReader('../../data/controversial/')
tsv_filename = corpus_reader.json_to_tsv('tokenized_posts/', topic_list=[TOPIC])

# tsv_filename = './tsv/WikiControversial-{}.tsv'.format(TOPIC)
corpus = WikiCorpus(tsv_filename)

# corpus = WikiCorpus('tokenized_posts/{}_posts.csv'.format(TOPIC))
corpus.tokenize_posts()

markers, marker_words = utils.read_convokit_markers('../coord-liwc-patterns.txt')
categories = list(markers.keys())

corpus.count_marker_categories(markers)
print(corpus.posts.describe())
# corpus.count_marker_tokens(marker_words)
# corpus.save('with_counts/{}_posts.csv'.format(TOPIC))

# corpus = WikiCorpus('with_counts/{}_posts.csv'.format(TOPIC))
pairs = corpus.reply_pairs()
al = Alignment(corpus, markers)


N_base, N_align, C_base, C_align = al.counts(mode='categorical')

means, intervals = al.swam(N_base, N_align, C_base, C_align, verbose=True)
base_means, align_means = means 
base_intervals, align_intervals = intervals

utils.plot_baseline_and_alignment(categories, 
                                base_means, align_means, 
                                base_intervals, align_intervals,
                                filename='plots/swam-{}'.format(TOPIC))

# users = corpus.get_users()
# # print(users.head())

# net = corpus.social_network(prune=False)
# corpus.assign_centrality('eigenvector')
# # print(users.head())

# tie_strengths = []
# for _, pair in pairs.iterrows():
# 	user_a, user_b = pair['author_name_a'], pair['author_name_b']
# 	tie_strengths.append(net[user_a][user_b]['weight'])

# pairs['tie_strength'] = tie_strengths

# strong_dyad_filter = ('strong-tie', lambda pair: (pair['tie_strength'] >= 2))
# weak_dyad_filter = ('weak-tie', lambda pair: (pair['tie_strength'] < 2))

# print(len(pairs[(pairs['tie_strength'] < 2)]), len(pairs[(pairs['tie_strength'] >= 2)]))
# for filter_str, group_filter in [strong_dyad_filter, weak_dyad_filter]:

#     N_base, N_align, C_base, C_align = al.counts(mode='categorical', group_filter=group_filter)

#     means, intervals = al.swam(N_base, N_align, C_base, C_align, verbose=True)
#     base_means, align_means = means 
#     base_intervals, align_intervals = intervals

#     utils.plot_baseline_and_alignment(categories, 
#                                     base_means, align_means, 
#                                     base_intervals, align_intervals,
#                                     filename='plots/swam-{}-{}'.format(TOPIC, filter_str))



# print(pairs.head())
# print(corpus)



# corpus.assign_centrality('betweenness')

# print(users.head())