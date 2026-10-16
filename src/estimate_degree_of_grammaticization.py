import sys, codecs
import argparse
import gensim
import numpy as np

import random


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from pulearn import WeightedElkanotoPuClassifier
from pulearn import ElkanotoPuClassifier


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('corpus_file')
    parser.add_argument('-f', '--function_word_pos_dict', default='function_word_pos.dict')
    parser.add_argument('-c', '--content_word_pos_dict', default='content_word_pos.dict')
    parser.add_argument('-t', '--freq_threshold', default=100, type=int)
    parser.add_argument('-C', '--C', default=1.0, type=float)
    parser.add_argument('-r', '--hold_out_ratio', default=0.2, type=float)
    parser.add_argument('-n', '--num_classifiers', default=10, type=int)
    parser.add_argument('-e', '--embedding_file', default='sample.vec')
    parser.add_argument('-p', '--pu_learning', action='store_true')
    parser.add_argument('-z', '--to_standardize', action='store_true')
    parser.add_argument('-s', '--seed', default=1, type=int)
   
    args = parser.parse_args()

    return args


def set_random_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)


def load_word_set(filename, delim=' '):
    words = []
    with codecs.open(filename,  'r', 'utf-8', 'ignore') as fp:
        for word in fp:
            word = word.rstrip()
            words.append(word)

    return set(words)


def check_if_target_type(total_freq,
                         pos2freq,
                         word_set,
                         ratio_threshold=0.95):

    is_target = False
    target_freq = 0
    for pos, freq in pos2freq.items():
        if pos in word_set:
            target_freq += freq


    ratio = target_freq/float(total_freq)
    if ratio>ratio_threshold:
        is_target = True 
       
    return is_target


def standardize(x, axis=1):
    xmean = x.mean(axis=axis, keepdims=True)
    xstd  = np.std(x, axis=axis, keepdims=True)
    zscore = (x-xmean)/xstd

    return zscore


def load_word_embeddings(filename, to_standardize=False):
    w2v = gensim.models.Word2Vec.load(filename)

    if to_standardize:
        vs = w2v.wv.vectors
        vs = standardize(vs)
        vs = vs/np.linalg.norm(vs, axis=1, keepdims=True)
        w2v.wv.vectors = vs

    return w2v


def separate_func_content_words(word2freq,
                                word2pos_freq,
                                func_word_set,
                                content_word_set,
                                freq_threshold=1000):
    function_words = []
    content_words = []
    other_words = []

    for word, freq in word2freq.items():
        if freq >= freq_threshold:
            if check_if_target_type(freq, word2pos_freq[word], func_word_set):
                function_words.append(word)
            elif check_if_target_type(freq, word2pos_freq[word], content_word_set):
                content_words.append(word)
            else:
                other_words.append(word)


    return function_words, content_words, other_words


def create_data_set(func_words, content_words, w2v,
                    func_word_label=1,
                    content_word_label=0):

    func_words_ = [ ]
    func_word_labels = [ ]
    func_word_vecs = [ ]
    for w in func_words:
        if w in w2v.wv:
            func_words_.append(w)
            func_word_labels.append(func_word_label)
            func_word_vecs.append(w2v.wv.get_vector(w))
    func_words = func_words_ 

    content_words_ = []
    content_word_labels = [ ]
    content_word_vecs = [ ]
    for word in content_words:
        if word in w2v.wv:
            content_words_.append(word)
            content_word_labels.append(content_word_label)
            content_word_vecs.append(w2v.wv.get_vector(word))
    content_words = content_words_

    return func_words, func_word_vecs, func_word_labels, content_words, content_word_vecs, content_word_labels


def split_data(data, n_split=5):

    num_instances = len(data)
    sub_data_size = int(num_instances/n_split)
    splitted_data = []
    for i in range(n_split):
        if i < n_split -1:
            splitted_data.append(data[i*sub_data_size:(i+1)*sub_data_size])
        else:
            splitted_data.append(data[i*sub_data_size:])

    return splitted_data
        

def pu_train_model(function_words,
                   func_word_vecs,
                   func_word_labels,
                   content_words,
                   content_word_vecs,
                   content_word_labels,
                   other_words,
                   other_word_vecs,
                   seed=7,
                   C=1.0,
                   hold_out_ratio=0.2):

    model = LogisticRegression(random_state=seed, C=C)

    other_word_labels = [ 0.0 ]*len(other_word_vecs)
    train_x = np.array(func_word_vecs + content_word_vecs + other_word_vecs)
    train_y = np.array(func_word_labels + content_word_labels + other_word_labels)

    pu_estimator = WeightedElkanotoPuClassifier(estimator=model,
                                                labeled=10,
                                                unlabeled=20,
                                                hold_out_ratio=hold_out_ratio)
    pu_estimator.fit(train_x, train_y)

    content_word2func_word_prob = {}
    content_word_probs = pu_estimator.predict_proba(content_word_vecs)
    for word, probs in zip(content_words, content_word_probs):
        content_word2func_word_prob[word] = probs[1]

    other_word2func_word_prob = {}
    other_word_probs = pu_estimator.predict_proba(other_word_vecs)
    for word, probs in zip(other_words, other_word_probs):
        other_word2func_word_prob[word] = probs[1]

    func_word2func_prob = {}
    func_pred_probs = pu_estimator.predict_proba(func_word_vecs)
    for word, probs in zip(function_words, func_pred_probs):
        func_word2func_prob[word] = probs[1]

    return pu_estimator, content_word2func_word_prob, other_word2func_word_prob, func_word2func_prob


def train_models_by_cv(function_words,
                       func_word_vecs,
                       func_word_labels,
                       content_words,
                       content_word_vecs,
                       content_word_labels,
                       other_words,
                       other_vecs,
                       seed=7,
                       C=1.0,
                       num_cv=10):

    dev_pred_probs = []
    other_pred_probs = []
    models = []
    word2func_word_prob = {}
    for i, (words, vecs, labels) in enumerate(zip(content_words, content_word_vecs, content_word_labels)):
        train_x = np.array(func_word_vecs + vecs)
        train_y = np.array(func_word_labels + labels)
        model = LogisticRegression(random_state=seed, C=C)
        #model = SVC(C=C, kernel='rbf', gamma=0.3, probability=True)
        model.fit(train_x, train_y)
        models.append(model)
        for j in range(num_cv):
            if i != j:
                dev_pred_probs = model.predict_proba(content_word_vecs[j])
                for word, probs in zip(content_words[j], dev_pred_probs):
                    word2func_word_prob[word] =\
                        word2func_word_prob.get(word, 0.0) + probs[1]
    content_word2func_word_prob =\
        { w: p/float(num_cv) for w, p in word2func_word_prob.items() }


    other_pred_probs = []
    for model in models:
        other_pred_probs.append(model.predict_proba(other_vecs))
    other_pred_probs = np.sum(other_pred_probs, axis=0)/float(num_cv)
    other_word2func_prob = {}
    for word, probs in zip(other_words, other_pred_probs):
        other_word2func_prob[word] = probs[1]

    func_pred_probs = []
    for model in models:
        func_pred_probs.append(model.predict_proba(func_word_vecs))
    func_pred_probs = np.sum(func_pred_probs, axis=0)/float(num_cv)
    func_word2func_prob = {}
    for word, probs in zip(function_words, func_pred_probs):
        func_word2func_prob[word] = probs[1]

    return models, content_word2func_word_prob, other_word2func_prob, func_word2func_prob


def main():
    args = parse_args()

    set_random_seed(args.seed)

    function_word_pos_set = load_word_set(args.function_word_pos_dict)
    content_word_set = load_word_set(args.content_word_pos_dict)
    w2v = load_word_embeddings(args.embedding_file,
                               to_standardize=args.to_standardize)

    word2freq = {}
    word2pos_freq = {}
    with codecs.open(args.corpus_file,  'r', 'utf-8', 'ignore') as fp:
        for line in fp:
            line = line.rstrip()
            word_poss = line.split(' ')
            for word_pos in word_poss:
                elem = word_pos.split('\t')
                if len(elem) == 2:
                    word = elem[0]
                    pos = elem[1]
                    word = word.lower()
                    word2freq[word] = word2freq.get(word, 0) + 1
                    pos2freq = word2pos_freq.get(word, {})
                    pos2freq[pos] = pos2freq.get(pos, 0) + 1
                    word2pos_freq[word] = pos2freq

    function_words, content_words, other_words =\
        separate_func_content_words(word2freq,
                                    word2pos_freq,
                                    function_word_pos_set,
                                    content_word_set,
                                    freq_threshold=args.freq_threshold)
    function_words, func_word_vecs, func_word_labels, content_words, content_word_vecs, content_word_labels =\
        create_data_set(function_words, content_words, w2v)

    other_vecs = [ ]
    other_words_ = [ ]
    for word in other_words:
        if word in w2v.wv:
            other_words_.append(word)
            other_vecs.append(w2v.wv.get_vector(word))
    other_words = other_words_

    if args.pu_learning == True:
        models, content_word2func_word_prob, other_word2func_word_prob, func_word2func_word_prob =\
            pu_train_model(function_words,
                           func_word_vecs,
                           func_word_labels,
                           content_words,
                           content_word_vecs,
                           content_word_labels,
                           other_words,
                           other_vecs,
                           hold_out_ratio=args.hold_out_ratio,
                           C=args.C,
                           seed=args.seed)
    else:
        # Note that content words are splited into N sets for 
        # cross-validation hereafter
        content_words = split_data(content_words, n_split=args.num_classifiers)
        content_word_vecs =\
            split_data(content_word_vecs, n_split=args.num_classifiers)
        content_word_labels =\
            split_data(content_word_labels, n_split=args.num_classifiers)

        models, content_word2func_word_prob, other_word2func_word_prob, func_word2func_word_prob =\
            train_models_by_cv(function_words,
                               func_word_vecs,
                               func_word_labels,
                               content_words,
                               content_word_vecs,
                               content_word_labels,
                               other_words,
                               other_vecs,
                               seed=args.seed,
                               C=args.C,
                               num_cv=args.num_classifiers)

    word_prob = []
    for word, prob in content_word2func_word_prob.items():
        word_prob.append((word, prob))
    for word, prob in other_word2func_word_prob.items():
        word_prob.append((word, prob))
    for word, prob in func_word2func_word_prob.items():
        word_prob.append((word, prob))
    for word, prob in sorted(word_prob, key=lambda x:x[1], reverse=True):
        print(word, prob)


if __name__ == '__main__':
    main()
