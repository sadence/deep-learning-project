'''
code adapted from https://github.com/vikasnar/Bleu/blob/master/calculatebleu.py
'''
import sys
import codecs
import os
import math
import operator
import json
from functools import reduce


def fetch_data(fics, generated_text):
    """ Store each reference and candidate sentences as a list """
    references = []
    for fic in fics:
        references.append(fic.body)
    candidate = generated_text
    return candidate, references


def count_ngram(candidate, references, n, character_level):
    print(f'char level: {character_level}')
    clipped_count = 0
    count = 0
    r = 0
    c = 0
    # print(len(candidate))
    # Calculate precision for each sentence
    ref_counts = []
    ref_lengths = []
    # Build dictionary of ngram counts
    for reference in references:
        ngram_d = {}
        if character_level:
            ref_sentence = reference
            words = list(ref_sentence.strip())
        else:
            ref_sentence = reference
            words = ref_sentence.strip().split()
        ref_lengths.append(len(words))
        limits = len(words) - n + 1
        # loop through the sentance consider the ngram length
        for i in range(limits):
            if character_level:
                ngram = ''.join(words[i:i+n]).lower()
            else:
                ngram = ' '.join(words[i:i+n]).lower()
            if ngram in ngram_d.keys():
                ngram_d[ngram] += 1
            else:
                ngram_d[ngram] = 1
        ref_counts.append(ngram_d)
        # candidate
        cand_sentence = candidate
        cand_dict = {}
        if character_level:
            words = list(cand_sentence.strip())
        else:
            words = cand_sentence.strip().split()
        print(f'cand sentence: {cand_sentence}')
        print(f'words: {words}')
        limits = len(words) - n + 1
        for i in range(0, limits):
            if character_level:
                ngram = ''.join(words[i:i + n]).lower()
            else:
                ngram = ' '.join(words[i:i + n]).lower()
            if ngram in cand_dict:
                cand_dict[ngram] += 1
            else:
                cand_dict[ngram] = 1
        clipped_count += clip_count(cand_dict, ref_counts)
        # print(ref_counts)
        count += limits
        r += best_length_match(ref_lengths, len(words))
        c += len(words)
        print(cand_dict)
    if clipped_count == 0:
        pr = 0
    else:
        pr = float(clipped_count) / count
    # bp = brevity_penalty(c, r)
    return pr, None


def clip_count(cand_d, ref_ds):
    """Count the clip count for each ngram considering all references"""
    count = 0
    for m in cand_d.keys():
        m_w = cand_d[m]
        m_max = 0
        for ref in ref_ds:
            if m in ref:
                m_max = max(m_max, ref[m])
        m_w = min(m_w, m_max)
        count += m_w
    return count


def best_length_match(ref_l, cand_l):
    """Find the closest length of reference to that of candidate"""
    least_diff = abs(cand_l-ref_l[0])
    best = ref_l[0]
    for ref in ref_l:
        if abs(cand_l-ref) < least_diff:
            least_diff = abs(cand_l-ref)
            best = ref
    return best


# def brevity_penalty(c, r):
#     if c > r:
#         bp = 1
#     else:
#         bp = math.exp(1-(float(r)/c))
#     return bp


def geometric_mean(precisions):
    return (reduce(operator.mul, precisions)) ** (1.0 / len(precisions))


def BLEU(candidate, references, character_level):
    precisions = []
    max_ngrams = 4  # normally 4
    for i in range(max_ngrams):
        pr, bp = count_ngram(candidate, references, i+1, character_level)
        precisions.append(pr)
    # We do not take into account the brevity precision
    print(precisions)
    bleu = geometric_mean(precisions)
    # is 0.0: Warum ?
    return bleu


def compute_bleu(fics, generated_text, character_level=True):
    '''Compute BLEU score.

    Arguments:
      fics {[type]} -- [description]
      generated_text {string} -- Generated text

    Returns:
      float -- BLEU score
    '''

    candidate, references = fetch_data(fics, generated_text)
    bleu = BLEU(candidate, references, character_level)
    return bleu


if __name__ == "__main__":
    # main deps
    import pickle
    from parse_fics import Fanfic

    with open("./fics.pkl", 'rb') as file:
        fics = pickle.load(file)
        fics = fics[:1]  # begin with only this much
    result = fics[0].body[:500]
    if len(sys.argv) > 1:
        compute_bleu(fics, result, bool(sys.argv[1]))
    else:
        compute_bleu(fics, result)
