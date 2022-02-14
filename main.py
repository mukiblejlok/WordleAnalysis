import sys
import time
from typing import Set
from string import ascii_lowercase
from itertools import product
import random
import plotly.graph_objects as go

import numpy as np
from tqdm import tqdm


def load_words(path: str) -> Set[str]:
    with open(path, "r") as f:
        return set(w.strip() for w in f.readlines())


def is_at_postion(words, letter, position):
    return set(w for w in words if w[position] == letter)


def is_at_other_postion(words, letter, position):
    return set(w for w in words if (letter in w) and (w[position] != letter))


def is_not_in_word(words, letter):
    return set(w for w in words if letter not in w)


def entropy(p: np.ndarray) -> float:
    e = 1e-8
    p = p[p > e]
    return np.sum(-p * np.log2(p))


def get_p_vector(word, lookup, all_words):
    full_len = len(all_words)
    p_v = []
    for variant in product((1, 2, 3), repeat=5):
        s = all_words.copy()
        for i, (vl, wl) in enumerate(zip(variant, word)):
            k = (vl, wl, i if vl != 3 else None)
            # print(k, len(s), s)
            s &= lookup[k]
        p_v.append(len(s) / full_len)
    return np.asarray(p_v)


if __name__ == '__main__':
    sl = load_words(r"data\shortlist.txt")
    fl = load_words(r"data\fulllist.txt")

    lookup_dict = {}
    all_keys = []
    for letter in ascii_lowercase:
        for pos in range(5):
            # In position, M = 1
            lookup_dict[(1, letter, pos)] = is_at_postion(words=fl, letter=letter, position=pos)
            # In other position, M = 2
            lookup_dict[(2, letter, pos)] = is_at_other_postion(words=fl, letter=letter, position=pos)
        # Not in word
        lookup_dict[(3, letter, None)] = is_not_in_word(words=fl, letter=letter)

    pbar = tqdm(sl)
    for word in pbar:
        pv = get_p_vector(word=word, lookup=lookup_dict, all_words=fl)
        pbar.set_description(f"{word} {entropy(pv):1.3f}")


    # print(pv)
    # fig = go.Figure()
    # fig.add_trace(go.Scatter(x=np.arange(len(pv)), y=sorted(pv, reverse=True)))
    # fig.show()
    #
