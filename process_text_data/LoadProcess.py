import os
import re
import sys
import time
# import enchant
import itertools
import numpy as np
import pandas as pd
import pickle
import string

from collections import Counter

# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from nltk.stem import PorterStemmer
# from nltk.stem import WordNetLemmatizer
# stemmer= PorterStemmer()
# lemmatizer=WordNetLemmatizer()
# stop_words = set(stopwords.words('english'))
# WORD = re.compile(r'\w+')
# d = enchant.Dict("en_US")


# def regTokenize(text):
#     words = WORD.findall(text)
#     return words


class ShowProcess:
    i = 0
    max_steps = 0
    max_arrow = 50

    def __init__(self, max_steps):
        self.max_steps = max_steps
        self.i = 0

    def show_process(self, i=None):
        if i is not None:
            self.i = i
        else:
            self.i += 1
        num_arrow = int(self.i * self.max_arrow / self.max_steps)
        num_line = self.max_arrow - num_arrow
        percent = self.i * 100.0 / self.max_steps
        process_bar = '[' + '>' * num_arrow + '-' * num_line + ']' + '%.3f' % percent + '%' + '\r'
        sys.stdout.write(process_bar)
        sys.stdout.flush()

    def close(self, words=''):
        # print(words)
        self.i = 0


def create_parameters_dict(**param):
    return param


def clean_str(text):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    text = re.sub(r"[^A-Za-z]", " ", text)
    text = re.sub(r"\'s", " \'s", text)
    text = re.sub(r"\'ve", " \'ve", text)
    text = re.sub(r"n\'t", " n\'t", text)
    text = re.sub(r"\'re", " \'re", text)
    text = re.sub(r"\'d", " \'d", text)
    text = re.sub(r"\'ll", " \'ll", text)
    text = re.sub(r",", " , ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\(", " \( ", text)
    text = re.sub(r"\)", " \) ", text)
    text = re.sub(r"\?", " \? ", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = text.lower()

    # text = regTokenize(text)
    # text = ' '.join([w for w in text if w not in stop_words])
    # text = text.translate(None, string.punctuation)

    return text.strip()


def pad_sentences(_text, _len, padding_word="<PAD/>"):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    sequence_length = _len
    u_text2 = {}
    print len(_text)
    for i in _text.keys():
        # print i
        sentence = _text[i]
        if sequence_length > len(sentence):

            num_padding = sequence_length - len(sentence)
            new_sentence = sentence + [padding_word] * num_padding
            u_text2[i] = new_sentence
        else:
            new_sentence = sentence[:sequence_length]
            u_text2[i] = new_sentence

    return u_text2


def build_vocab(sentences1, sentences2):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    word_counts1 = Counter(itertools.chain(*sentences1))
    # Mapping from index to word
    vocabulary_inv1 = [x[0] for x in word_counts1.most_common()]
    # vocabulary_inv1 = list(sorted(vocabulary_inv1))
    # Mapping from word to index
    vocabulary1 = {x: i for i, x in enumerate(vocabulary_inv1)}
    print('done vocab 1')

    word_counts2 = Counter(itertools.chain(*sentences2))
    # Mapping from index to word
    vocabulary_inv2 = [x[0] for x in word_counts2.most_common()]

    # vocabulary_inv2 = list(sorted(vocabulary_inv2))
    # Mapping from word to index
    vocabulary2 = {x: i for i, x in enumerate(vocabulary_inv2)}
    print('done vocab 2')
    return [vocabulary1, vocabulary_inv1, vocabulary2, vocabulary_inv2]


# def build_vocab(sentences1, sentences2):
#     """
#     Builds a vocabulary mapping from word to index based on the sentences.
#     Returns vocabulary mapping and inverse vocabulary mapping.
#     """
#     # Build vocabulary
#     word_counts1 = dict(Counter(itertools.chain(*sentences1))).keys()
#     # words to stemm words
#     _1_word_to_stemm_lemm = {w: lemmatizer.lemmatize(stemmer.stem(w)) for w in word_counts1 if
#                              w != "<PAD/>" and d.check(w)}
#     _1_word_to_stemm_lemm["<PAD/>"] = "<PAD/>"
#     # stemm words to index
#     _1_stemm_lemm_to_index = {x: i for i, x in enumerate(_1_word_to_stemm_lemm.values())}
#     print('done vocab 1')
#
#     # Build vocabulary
#     word_counts2 = dict(Counter(itertools.chain(*sentences2))).keys()
#     # words to stemm words
#     _2_word_to_stemm_lemm = {w: lemmatizer.lemmatize(stemmer.stem(w)) for w in word_counts2 if
#                              w != "<PAD/>" and d.check(w)}
#     _2_word_to_stemm_lemm["<PAD/>"] = "<PAD/>"
#     # stemm words to index
#     _2_stemm_lemm_to_index = {x: i for i, x in enumerate(_2_word_to_stemm_lemm.values())}
#     print('done vocab 1')
#     return _1_word_to_stemm_lemm, _1_stemm_lemm_to_index, _2_word_to_stemm_lemm, _2_stemm_lemm_to_index

# def build_input_data(u_text, i_text, user_word_to_sl, user_sl_to_index, item_word_to_sl, item_sl_to_index):
#     """
#     Maps sentencs and labels to vectors based on a vocabulary.
#     """
#     u_text2 = {}
#     for i in u_text.keys():
#         u_reviews = u_text[i]
#         temp = [user_sl_to_index.get(user_word_to_sl.get(word)) for word in u_reviews if word in user_word_to_sl]
#         u_text2[i] = list(set(temp))
#     print('done build in data user')
#
#     i_text2 = {}
#     for j in i_text.keys():
#         i_reviews = i_text[j]
#         temp = [item_sl_to_index.get(item_word_to_sl.get(word)) for word in i_reviews if word in item_word_to_sl]
#         i_text2[j] = list(set(temp))
#     print('done build in data item')
#
#     return u_text2, i_text2


def build_input_data(u_text, i_text, vocabulary_u, vocabulary_i):
    """
    Maps sentencs and labels to vectors based on a vocabulary.
    """
    u_text2 = {}
    for i in u_text.keys():
        u_reviews = u_text[i]
        u = np.array([vocabulary_u.get(word) for word in u_reviews])
        u_text2[i] = u
    i_text2 = {}
    for j in i_text.keys():
        i_reviews = i_text[j]
        i = np.array([vocabulary_i.get(word) for word in i_reviews])
        i_text2[j] = i
    return u_text2, i_text2


class LoadData(object):
    def __init__(self, param):
        self.path = param['path']

        self.train_df = pd.read_csv(os.path.join(self.path, param['train_name']))
        self.validate_df = pd.read_csv(os.path.join(self.path, param['validate_name']))
        self.test_df = pd.read_csv(os.path.join(self.path, param['test_name']))

        self.train_df = self.train_df.dropna()
        self.validate_df = self.validate_df.dropna()
        self.test_df = self.test_df.dropna()

        print('done reading data')

    def _process_train_test_pivot_dict(self, data, data_2):
        self.user_rated_reviews = {}  # user: reviews
        self.item_rated_reviews = {}  # item : reviews
        self.user_rated_items = {}  # user: item
        self.item_rated_users = {}  # item: user

        for (u, i, r, text) in data:
            if u in self.user_rated_reviews:
                self.user_rated_reviews[u].append(text)
                self.user_rated_items[u].append(i)
            else:
                self.user_rated_reviews[u] = [text]
                self.user_rated_items[u] = [i]

            if i in self.item_rated_reviews:
                self.item_rated_reviews[i].append(text)
                self.item_rated_users[i].append(u)
            else:
                self.item_rated_reviews[i] = [text]
                self.item_rated_users[i] = [u]
        print('done data')

        for (u, i, r, text) in data_2:
            if u not in self.user_rated_reviews:
                self.user_rated_items[u] = ['0']
                self.user_rated_reviews[u] = [0]

            if i not in self.item_rated_reviews:
                self.item_rated_users[i] = ['0']
                self.item_rated_reviews[i] = [0]
        print('done data_2')

    def _process_label(self, train, validate, test):
        self.u_text, self.i_text = {}, {}

        # train
        self.uid_train, self.iid_train, self.y_train = [], [], []
        sp = ShowProcess(len(train))
        for (u, i, r, _) in train:
            self.uid_train.append(u)
            self.iid_train.append(i)
            self.y_train.append(r)
            if u not in self.u_text:
                self.u_text[u] = '<PAD/>'
                for review in self.user_rated_reviews[u]:
                    self.u_text[u] = self.u_text[u] + ' ' + review.strip()
                self.u_text[u] = clean_str(self.u_text[u])
                self.u_text[u] = self.u_text[u].split(' ')

            if i not in self.i_text:
                self.i_text[i] = '<PAD/>'
                for review in self.item_rated_reviews[i]:
                    self.i_text[i] = self.i_text[i] + ' ' + review.strip()
                self.i_text[i] = clean_str(self.i_text[i])
                self.i_text[i] = self.i_text[i].split(' ')
            sp.show_process()
        sp.close()
        print('finish train reviews')

        # validate
        self.uid_validate, self.iid_validate, self.y_validate = [], [], []
        sp = ShowProcess(len(validate))
        for (u, i, r, _) in validate:
            self.uid_validate.append(u)
            self.iid_validate.append(i)
            self.y_validate.append(r)
            if u not in self.u_text:
                self.u_text[u] = '<PAD/>'
                self.u_text[u] = clean_str(self.u_text[u])
                self.u_text[u] = self.u_text[u].split(' ')

            if i not in self.i_text:
                self.i_text[i] = '<PAD/>'
                self.i_text[i] = clean_str(self.i_text[i])
                self.i_text[i] = self.i_text[i].split(' ')
            sp.show_process()
        sp.close()
        print('finish validate reviews')

        # test
        self.uid_test, self.iid_test, self.y_test = [], [], []
        sp = ShowProcess(len(test))
        for (u, i, r, _) in test:
            self.uid_test.append(u)
            self.iid_test.append(i)
            self.y_test.append(r)
            if u not in self.u_text:
                self.u_text[u] = '<PAD/>'
                self.u_text[u] = clean_str(self.u_text[u])
                self.u_text[u] = self.u_text[u].split(' ')

            if i not in self.i_text:
                self.i_text[i] = '<PAD/>'
                self.i_text[i] = clean_str(self.i_text[i])
                self.i_text[i] = self.i_text[i].split(' ')
            sp.show_process()
        sp.close()
        print('finish test reviews')

        # choose text length
        temp = np.sort([len(x) for x in self.u_text.values()])
        # self.u_len = temp[int(0.85 * len(temp)) - 1]
        self.u_len = 100
        temp = np.sort([len(x) for x in self.i_text.values()])
        # self.i_len = temp[int(0.85 * len(temp)) - 1]
        self.i_len = 100
        print('review length (user/item) {}/{}'.format(self.u_len, self.i_len))

        self.num_user, self.num_item = len(self.u_text), len(self.i_text)
        print('number of user/item {}/{}'.format(self.num_user, self.num_item))


class LoadProcessBeer(LoadData):
    def __init__(self, param):
        super(LoadProcessBeer, self).__init__(param)

        # unique users and beers
        print('process user and beer')
        unique_users = pd.concat([self.train_df, self.validate_df, self.test_df])['review_profileName'].unique()
        unique_beers = pd.concat([self.train_df, self.validate_df, self.test_df])['beer_beerId'].unique()
        unique_users_to_index = {k: v for v, k in enumerate(unique_users)}
        unique_beers_to_index = {k: v for v, k in enumerate(unique_beers)}

        # user -> user id
        # beer -> beer id
        # calcualte -> ratings
        print('process train data')
        train_users = self.train_df['review_profileName'].apply(lambda x: unique_users_to_index.get(x)).values
        train_beers = self.train_df['beer_beerId'].apply(lambda x: unique_beers_to_index.get(x)).values
        train_ratings = self.train_df['review_overall'].apply(lambda x: x * 2 - 1).values
        train_reviews = self.train_df['review_text'].values
        train = np.array([train_users, train_beers, train_ratings, train_reviews]).T

        validate_users = self.validate_df['review_profileName'].apply(lambda x: unique_users_to_index.get(x)).values
        validate_beers = self.validate_df['beer_beerId'].apply(lambda x: unique_beers_to_index.get(x)).values
        validate_ratings = self.validate_df['review_overall'].apply(lambda x: x * 2 - 1).values
        validate_reviews = self.validate_df['review_text'].values
        validate = np.array([validate_users, validate_beers, validate_ratings, validate_reviews]).T

        test_users = self.test_df['review_profileName'].apply(lambda x: unique_users_to_index.get(x)).values
        test_beers = self.test_df['beer_beerId'].apply(lambda x: unique_beers_to_index.get(x)).values
        test_ratings = self.test_df['review_overall'].apply(lambda x: x * 2 - 1).values
        test_reviews = self.test_df['review_text'].values
        test = np.array([test_users, test_beers, test_ratings, test_reviews]).T

        data = np.concatenate((train, validate))
        data_2 = test

        """ 1. user -> reviews, item -> reviews, user -> items, item -> users """
        self._process_train_test_pivot_dict(data=data, data_2=data_2)

        """ 2. user -> text, item -> text """
        self._process_label(train=train, validate=validate, test=test)

        """ 3. pad text """
        self.u_text = pad_sentences(_text=self.u_text, _len=self.u_len)
        self.i_text = pad_sentences(_text=self.i_text, _len=self.i_len)

        # """ 4. user item vocabulary """
        # self.user_word_to_sl, self.user_sl_to_index, self.item_word_to_sl, self.item_sl_to_index = build_vocab(
        #     self.u_text.values(), self.i_text.values())
        #
        # """ 5. build input data """
        # self.u_text, self.i_text = build_input_data(
        #     u_text=self.u_text, i_text=self.i_text,
        #     user_word_to_sl=self.user_word_to_sl, user_sl_to_index=self.user_sl_to_index,
        #     item_word_to_sl=self.item_word_to_sl, item_sl_to_index=self.item_sl_to_index)
        #
        # """ 6. save """
        # self.data_parameters = create_parameters_dict(
        #     num_user=self.num_user, num_item=self.num_item,
        #     user_review_length=self.u_len, item_review_length=self.i_len,
        #     user_word_to_sl=self.user_word_to_sl, item_word_to_sl=self.item_word_to_sl,
        #     user_sl_to_index=self.user_sl_to_index, item_sl_to_index=self.item_sl_to_index,
        #     train_length=len(self.y_train), validate_length=len(self.y_validate), test_length=len(self.y_test),
        #     u_text=self.u_text, i_text=self.i_text, train_data=train, validate_data=validate, test_data=test)

        """ 4. user item vocabulary """
        self.user_word_index_dict, self.user_word_list, self.item_word_index_dict, self.item_word_list = build_vocab(
            self.u_text.values(), self.i_text.values())

        """ 5. build input data """
        self.u_text, self.i_text = build_input_data(
            u_text=self.u_text, i_text=self.i_text,
            vocabulary_u=self.user_word_index_dict, vocabulary_i=self.item_word_index_dict)

        """ 6. save """
        self.data_parameters = create_parameters_dict(
            num_user=self.num_user, num_item=self.num_item,
            user_review_length=self.u_len, item_review_length=self.i_len,
            user_word_index_dict=self.user_word_index_dict, item_word_index_dict=self.item_word_index_dict,
            user_word_list=self.user_word_list, item_word_list=self.item_word_list,
            train_length=len(self.y_train), validate_length=len(self.y_validate), test_length=len(self.y_test),
            u_text=self.u_text, i_text=self.i_text, train_data=train, validate_data=validate, test_data=test)

        output = open(os.path.join(param['save_path'], 'data_param.pkl'), 'wb')
        pickle.dump(self.data_parameters, output)


class LoadProcessAmazon(LoadData):
    def __init__(self, param):
        super(LoadProcessAmazon, self).__init__(param)

        # unique users and items
        print('process user and beer')
        unique_users = pd.concat([self.train_df, self.validate_df, self.test_df])['reviewerID'].unique()
        unique_items = pd.concat([self.train_df, self.validate_df, self.test_df])['asin'].unique()
        unique_users_to_index = {k: v for v, k in enumerate(unique_users)}
        unique_items_to_index = {k: v for v, k in enumerate(unique_items)}

        # user -> user id
        # item -> item id
        # calcualte -> ratings
        print('process train data')
        train_users = self.train_df['reviewerID'].apply(lambda x: unique_users_to_index.get(x)).values
        train_items = self.train_df['asin'].apply(lambda x: unique_items_to_index.get(x)).values
        train_ratings = self.train_df['overall'].values
        train_reviews = self.train_df['reviewText'].values
        train = np.array([train_users, train_items, train_ratings, train_reviews]).T

        validate_users = self.validate_df['reviewerID'].apply(lambda x: unique_users_to_index.get(x)).values
        validate_items = self.validate_df['asin'].apply(lambda x: unique_items_to_index.get(x)).values
        validate_ratings = self.validate_df['overall'].values
        validate_reviews = self.validate_df['reviewText'].values
        validate = np.array([validate_users, validate_items, validate_ratings, validate_reviews]).T

        test_users = self.test_df['reviewerID'].apply(lambda x: unique_users_to_index.get(x)).values
        test_items = self.test_df['asin'].apply(lambda x: unique_items_to_index.get(x)).values
        test_ratings = self.test_df['overall'].values
        test_reviews = self.test_df['reviewText'].values
        test = np.array([test_users, test_items, test_ratings, test_reviews]).T

        data = np.concatenate((train, validate))
        data_2 = test

        """ 1. user -> reviews, item -> reviews, user -> items, item -> users """
        self._process_train_test_pivot_dict(data=data, data_2=data_2)

        """ 2. user -> text, item -> text """
        self._process_label(train=train, validate=validate, test=test)

        """ 3. pad text """
        self.u_text = pad_sentences(_text=self.u_text, _len=self.u_len)
        self.i_text = pad_sentences(_text=self.i_text, _len=self.i_len)

        # """ 4. user item vocabulary """
        # self.user_word_to_sl, self.user_sl_to_index, self.item_word_to_sl, self.item_sl_to_index = build_vocab(
        #     self.u_text.values(), self.i_text.values())
        #
        # """ 5. build input data """
        # self.u_text, self.i_text = build_input_data(
        #     u_text=self.u_text, i_text=self.i_text,
        #     user_word_to_sl=self.user_word_to_sl, user_sl_to_index=self.user_sl_to_index,
        #     item_word_to_sl=self.item_word_to_sl, item_sl_to_index=self.item_sl_to_index)
        #
        # """ 6. save """
        # self.data_parameters = create_parameters_dict(
        #     num_user=self.num_user, num_item=self.num_item,
        #     user_review_length=self.u_len, item_review_length=self.i_len,
        #     user_word_to_sl=self.user_word_to_sl, item_word_to_sl=self.item_word_to_sl,
        #     user_sl_to_index=self.user_sl_to_index, item_sl_to_index=self.item_sl_to_index,
        #     train_length=len(self.y_train), validate_length=len(self.y_validate), test_length=len(self.y_test),
        #     u_text=self.u_text, i_text=self.i_text, train_data=train, validate_data=validate, test_data=test)

        """ 4. user item vocabulary """
        self.user_word_index_dict, self.user_word_list, self.item_word_index_dict, self.item_word_list = build_vocab(
            self.u_text.values(), self.i_text.values())

        """ 5. build input data """
        self.u_text, self.i_text = build_input_data(
            u_text=self.u_text, i_text=self.i_text,
            vocabulary_u=self.user_word_index_dict, vocabulary_i=self.item_word_index_dict)

        """ 6. save """
        self.data_parameters = create_parameters_dict(
            num_user=self.num_user, num_item=self.num_item,
            user_review_length=self.u_len, item_review_length=self.i_len,
            user_word_index_dict=self.user_word_index_dict, item_word_index_dict=self.item_word_index_dict,
            user_word_list=self.user_word_list, item_word_list=self.item_word_list,
            train_length=len(self.y_train), validate_length=len(self.y_validate), test_length=len(self.y_test),
            u_text=self.u_text, i_text=self.i_text, train_data=train, validate_data=validate, test_data=test)

        output = open(os.path.join(param['save_path'], 'data_param.pkl'), 'wb')
        pickle.dump(self.data_parameters, output)


class LoadProcessYelp(LoadData):
    def __init__(self, param):
        super(LoadProcessYelp, self).__init__(param)

        # unique users and items
        print('process user and beer')
        unique_users = pd.concat([self.train_df, self.validate_df, self.test_df])['user'].unique()
        unique_items = pd.concat([self.train_df, self.validate_df, self.test_df])['item'].unique()
        unique_users_to_index = {k: v for v, k in enumerate(unique_users)}
        unique_items_to_index = {k: v for v, k in enumerate(unique_items)}

        # user -> user id
        # item -> item id
        # calcualte -> ratings
        print('process train data')
        train_users = self.train_df['user'].apply(lambda x: unique_users_to_index.get(x)).values
        train_items = self.train_df['item'].apply(lambda x: unique_items_to_index.get(x)).values
        train_ratings = self.train_df['rating'].values
        train_reviews = self.train_df['review'].values
        train = np.array([train_users, train_items, train_ratings, train_reviews]).T

        validate_users = self.validate_df['user'].apply(lambda x: unique_users_to_index.get(x)).values
        validate_items = self.validate_df['item'].apply(lambda x: unique_items_to_index.get(x)).values
        validate_ratings = self.validate_df['rating'].values
        validate_reviews = self.validate_df['review'].values
        validate = np.array([validate_users, validate_items, validate_ratings, validate_reviews]).T

        test_users = self.test_df['user'].apply(lambda x: unique_users_to_index.get(x)).values
        test_items = self.test_df['item'].apply(lambda x: unique_items_to_index.get(x)).values
        test_ratings = self.test_df['rating'].values
        test_reviews = self.test_df['review'].values
        test = np.array([test_users, test_items, test_ratings, test_reviews]).T

        data = np.concatenate((train, validate))
        data_2 = test

        """ 1. user -> reviews, item -> reviews, user -> items, item -> users """
        self._process_train_test_pivot_dict(data=data, data_2=data_2)

        """ 2. user -> text, item -> text """
        self._process_label(train=train, validate=validate, test=test)

        """ 3. pad text """
        self.u_text = pad_sentences(_text=self.u_text, _len=self.u_len)
        self.i_text = pad_sentences(_text=self.i_text, _len=self.i_len)

        # """ 4. user item vocabulary """
        # self.user_word_to_sl, self.user_sl_to_index, self.item_word_to_sl, self.item_sl_to_index = build_vocab(
        #     self.u_text.values(), self.i_text.values())
        #
        # """ 5. build input data """
        # self.u_text, self.i_text = build_input_data(
        #     u_text=self.u_text, i_text=self.i_text,
        #     user_word_to_sl=self.user_word_to_sl, user_sl_to_index=self.user_sl_to_index,
        #     item_word_to_sl=self.item_word_to_sl, item_sl_to_index=self.item_sl_to_index)
        #
        # """ 6. save """
        # self.data_parameters = create_parameters_dict(
        #     num_user=self.num_user, num_item=self.num_item,
        #     user_review_length=self.u_len, item_review_length=self.i_len,
        #     user_word_to_sl=self.user_word_to_sl, item_word_to_sl=self.item_word_to_sl,
        #     user_sl_to_index=self.user_sl_to_index, item_sl_to_index=self.item_sl_to_index,
        #     train_length=len(self.y_train), validate_length=len(self.y_validate), test_length=len(self.y_test),
        #     u_text=self.u_text, i_text=self.i_text, train_data=train, validate_data=validate, test_data=test)

        """ 4. user item vocabulary """
        self.user_word_index_dict, self.user_word_list, self.item_word_index_dict, self.item_word_list = build_vocab(
            self.u_text.values(), self.i_text.values())

        """ 5. build input data """
        self.u_text, self.i_text = build_input_data(
            u_text=self.u_text, i_text=self.i_text,
            vocabulary_u=self.user_word_index_dict, vocabulary_i=self.item_word_index_dict)

        """ 6. save """
        self.data_parameters = create_parameters_dict(
            num_user=self.num_user, num_item=self.num_item,
            user_review_length=self.u_len, item_review_length=self.i_len,
            user_word_index_dict=self.user_word_index_dict, item_word_index_dict=self.item_word_index_dict,
            user_word_list=self.user_word_list, item_word_list=self.item_word_list,
            train_length=len(self.y_train), validate_length=len(self.y_validate), test_length=len(self.y_test),
            u_text=self.u_text, i_text=self.i_text, train_data=train, validate_data=validate, test_data=test)

        output = open(os.path.join(param['save_path'], 'data_param.pkl'), 'wb')
        pickle.dump(self.data_parameters, output)


if __name__ == '__main__':
    start_time = time.time()

    param_dict = dict()
    param_dict['path'] = '/home/sixun/KDD_2019_beer/processed_data'
    param_dict['train_name'] = 'beer_train.csv'
    param_dict['validate_name'] = 'beer_validate.csv'
    param_dict['test_name'] = 'beer_test.csv'
    param_dict['save_path'] = '../data'

    load_beer = LoadProcessBeer(param=param_dict)

    print('cost time', time.time() - start_time)




