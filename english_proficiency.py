import os
from statsmodels.robust import mad
from planck.db.helpers import dict_cache
from planck.predictors.base_predictor import GraphDependency
from collections import defaultdict
import numpy as np
from nltk.stem.porter import PorterStemmer
from itertools import chain, groupby
from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from planck.predictors.base_predictor import depends_on
import re
from nltk import pos_tag

here = os.path.dirname(os.path.abspath(__file__))
stopwords = set(stopwords.words('english'))
non_word_p = re.compile('\W+')
punctuation = set(",.;:' ")
sent_stop_p = re.compile('([a-z]\.)([A-Z])')


def get_features(text):
    target_funcs = {
        n_sents,
        max_sent_length,
        min_sent_length,
        avg_sent_length,
        in_vocab_words,
        mad_word_prob,
        min_word_prob,
        max_word_prob,
        avg_connectors,
        yule,
        avg_non_words,
        avg_n_punct_per_sent,
        max_n_punct_per_sent,
        avg_verbs,
        type_token_ratio,
    }

    res = compute(target_funcs, {base_case: text})
    return {func.__name__: v for func, v in res.iteritems()}


def compute(target_funcs, base_cases):
    outputs = {}
    vector = {}

    for func in GraphDependency.get_dependency_clausure(*target_funcs, include_requested=True):
        dependencies = GraphDependency.dependency_graph.get(func)
        if dependencies:
            outputs[func] = func(outputs[dependencies[0]])
        else:
            outputs[func] = func(base_cases[func])

        if func in target_funcs:
            vector[func] = outputs[func]
    return vector


def base_case(txt):
    return txt


@depends_on(base_case)
def tokenize(txt):
    txt = sent_stop_p.sub('\g<1> \g<2>', txt)
    return sent_tokenize(txt)


@depends_on(tokenize)
def n_sents(sents):
    return len(sents)


@depends_on(tokenize)
def tokenize_sents(sents):
    res = []
    for sent in sents:
        new_sent = []
        for word in word_tokenize(sent):
            word = word.lower()
            if word == "n't": word = 'not'
            new_sent.append(word)
        res.append(new_sent)
    return res


@depends_on(base_case)
def avg_non_words(txt):
    l = [e.strip() for e in non_word_p.findall(txt)]
    l = [e for e in l if e and e not in punctuation]
    return len(l) / float(len(word_tokenize(txt)))


@depends_on(tokenize)
def n_punct_per_sent(sents):
    l = []
    for sent in sents:
        n_puncts = len([e for e in sent if e in ',;:'])
        l.append(n_puncts)
    return l


@depends_on(n_punct_per_sent)
def avg_n_punct_per_sent(puncts):
    return np.mean(puncts)


@depends_on(n_punct_per_sent)
def max_n_punct_per_sent(puncts):
    return max(puncts)


@depends_on(tokenize_sents)
def postaged_sents(tok_sents):
    return map(pos_tag, tok_sents)


@depends_on(postaged_sents)
def avg_verbs(sents):
    verbs = []
    for sent in sents:
        n_verbs = len([e for e, tag in sent if tag[0] == 'V'])
        verbs.append(n_verbs / float(len(sent)))
    return np.mean(verbs)


@depends_on(tokenize_sents)
def sent_lengths(sents):
    return map(len, sents)


@depends_on(tokenize_sents)
def yule(tok_sents):
    # yule's I measure (the inverse of yule's K measure)
    # higher number is higher diversity - richer vocabulary
    d = defaultdict(int)
    stemmer = PorterStemmer()
    for w in chain(*tok_sents):
        w = stemmer.stem(w).lower()
        d[w] += 1

    M1 = float(len(d))
    M2 = sum([len(list(g)) * (freq ** 2) for freq, g in groupby(sorted(d.values()))])

    if M1 == M2:
        return 0
    else:
        return (M1 * M1) / (M2 - M1)


@depends_on(tokenize_sents)
def type_token_ratio(tok_sents):
    n_tokens = float(sum(map(len, tok_sents)))
    res = len(set(chain(*tok_sents))) / n_tokens
    normalization_constant = 10 / np.log(n_tokens * 4)
    return res / normalization_constant


@depends_on(sent_lengths)
def max_sent_length(lengths):
    return max(lengths)


@depends_on(sent_lengths)
def min_sent_length(lengths):
    return min(lengths)


@depends_on(sent_lengths)
def avg_sent_length(lengths):
    return np.mean(lengths)


@depends_on(tokenize_sents)
def remove_stopwords(tok_sents):
    res = []
    for sent in tok_sents:
        res.append([e for e in sent if e not in stopwords])
    return res


@dict_cache
def load_vocab():
    vocab = {}
    # from http://norvig.com/ngrams/count_1w.txt
    with open(os.path.join(here, 'count_1w.txt')) as f:
        for line in f:
            w, cnt = line.split('\t')
            vocab[w] = int(cnt)
    s = float(sum(vocab.values()))
    res = {k: v / s for k, v in vocab.iteritems()}
    min_vocab_prob = min(res.itervalues()) / 2
    return res, min_vocab_prob


@depends_on(remove_stopwords)
def in_vocab_words(tok_sents):
    # if you want to use this, need to download this file #http://norvig.com/ngrams/count_1w.txt
    words = list(chain(*tok_sents))
    vocab, _ = load_vocab()
    n = len([e for e in words if e in vocab])
    return float(n) / len(words)


@depends_on(remove_stopwords)
def word_probs(tok_sents):
    vocab, _ = load_vocab()
    return [np.log(vocab[word]) for word in chain(*tok_sents) if word in vocab]


@depends_on(word_probs)
def min_word_prob(word_probs):
    _, min_vocab_prob = load_vocab()
    return min(word_probs + [min_vocab_prob])


@depends_on(word_probs)
def max_word_prob(word_probs):
    return np.mean(word_probs)


@depends_on(word_probs)
def mad_word_prob(word_probs):
    return mad(word_probs)


@depends_on(word_probs)
def avg_word_prob(word_probs):
    return np.mean(word_probs)


def load_connectors():
    with open(os.path.join(here, 'connectors.txt')) as f:
        connectors = [e.strip().lower() for e in f.read().split('\n') if e]

    return {k: re.compile('(\W|^)%s(\W|$)' % k, re.I) for k in connectors}


connectors = load_connectors()


@depends_on(tokenize)
def avg_connectors(sents):
    l = []
    for sent in sents:
        has_any = any([p.search(sent) is not None for p in connectors.itervalues()])
        l.append(has_any)
    return np.mean(l)
