import codecs
import math
import numpy as np
import os
import time
import torch
import torch.nn as nn
from typing import Tuple, List, Dict, TypeVar
T = TypeVar('T')


def update_tag_scheme(sentences: List[List[List[str]]], tag_scheme: str):
    """
    Check and update sentences tagging scheme to IOB2.
    Only IOB1 and IOB2 schemes are accepted.
    """
    for i, s in enumerate(sentences):
        tags = [w[-1] for w in s]
        # Check that tags are given in the IOB format
        if not iob2(tags):
            s_str = '\n'.join(' '.join(w) for w in s)
            raise Exception('Sentences should be given in IOB format! Please check sentence %i:\n%s' % (i, s_str))
        if tag_scheme == 'iob':
            # If format was IOB1, we convert to IOB2
            for word, new_tag in zip(s, tags):
                word[-1] = new_tag
        elif tag_scheme == 'iobes':
            new_tags = iob_iobes(tags)
            for word, new_tag in zip(s, new_tags):
                word[-1] = new_tag
        else:
            raise Exception('Unknown tagging scheme!')


def iob2(tags):
    """
    Check that tags have a valid IOB format.
    Tags in IOB1 format are converted to IOB2.
    """
    for i, tag in enumerate(tags):
        if tag == 'O':
            continue
        split = tag.split('-')
        if len(split) != 2 or split[0] not in ['I', 'B']:
            return False
        if split[0] == 'B':
            continue
        elif i == 0 or tags[i - 1] == 'O':  # conversion IOB1 to IOB2
            tags[i] = 'B' + tag[1:]
        elif tags[i - 1][1:] == tag[1:]:
            continue
        else:  # conversion IOB1 to IOB2
            tags[i] = 'B' + tag[1:]
    return True


def iob_iobes(tags):
    """
    IOB -> IOBES
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag == 'O':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'B':
            if i + 1 != len(tags) and \
               tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('B-', 'S-'))
        elif tag.split('-')[0] == 'I':
            if i + 1 < len(tags) and \
                    tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('I-', 'E-'))
        else:
            raise Exception('Invalid IOB format!')
    return new_tags


def create_dico(item_list: List[List[T]]) -> Dict[T, int]:
    dico: Dict[T, int] = {}
    for items in item_list:
        for item in items:
            if item not in dico:
                dico[item] = 1
            else:
                dico[item] += 1
    return dico


def create_desc_mapping(dico: Dict[T, int]) -> Tuple[Dict[T, int], Dict[int, T]]:
    sorted_items = sorted(dico.items(), key=lambda x: (-x[1], x[0]))  # List[Tuple[T, int]]
    id_to_item = {i: v[0] for i, v in enumerate(sorted_items)}
    item_to_id = {v: k for k, v in id_to_item.items()}
    return item_to_id, id_to_item


def get_affix_dict_list(sentences: List[List[str]], threshold: int) -> Tuple[List[Dict[str, int]], List[Dict[str, int]]]:
    prefixes_list = [[], [], [], []]
    suffixes_list = [[], [], [], []]
    prefix_counter_list = [{}, {}, {}, {}]
    suffix_counter_list = [{}, {}, {}, {}]
    prefix_id_dicts = [{}, {}, {}, {}]
    suffix_id_dicts = [{}, {}, {}, {}]
    for sentence in sentences:
        for word in sentence:
            if len(word) == 1:
                prefixes_list[0].append(word[:1])
                suffixes_list[0].append(word[-1:])
            elif len(word) == 2:
                prefixes_list[0].append(word[:1])
                prefixes_list[1].append(word[:2])
                suffixes_list[0].append(word[-1:])
                suffixes_list[1].append(word[-2:])
            elif len(word) == 3:
                prefixes_list[0].append(word[:1])
                prefixes_list[1].append(word[:2])
                prefixes_list[2].append(word[:3])
                suffixes_list[0].append(word[-1:])
                suffixes_list[1].append(word[-2:])
                suffixes_list[2].append(word[-3:])
            elif len(word) >= 4:
                prefixes_list[0].append(word[:1])
                prefixes_list[1].append(word[:2])
                prefixes_list[2].append(word[:3])
                prefixes_list[3].append(word[:4])
                suffixes_list[0].append(word[-1:])
                suffixes_list[1].append(word[-2:])
                suffixes_list[2].append(word[-3:])
                suffixes_list[3].append(word[-4:])

    for n in range(0, 4):
        for prefix in prefixes_list[n]:
            if prefix not in prefix_counter_list[n]:
                prefix_counter_list[n][prefix] = 0
            else:
                prefix_counter_list[n][prefix] += 1
        for suffix in suffixes_list[n]:
            if suffix not in suffix_counter_list[n]:
                suffix_counter_list[n][suffix] = 0
            else:
                suffix_counter_list[n][suffix] += 1

        prefixes_list[n] = sorted(filter(lambda p: prefix_counter_list[n][p] >= threshold, list(set(prefixes_list[n]))), key=lambda p: prefix_counter_list[n][p], reverse=True)
        suffixes_list[n] = sorted(filter(lambda s: suffix_counter_list[n][s] >= threshold, list(set(suffixes_list[n]))), key=lambda s: suffix_counter_list[n][s], reverse=True)
        for idx, prefix in enumerate(prefixes_list[n]):
            prefix_id_dicts[n][prefix] = idx + 1
        for idx, suffix in enumerate(suffixes_list[n]):
            suffix_id_dicts[n][suffix] = idx + 1

    return prefix_id_dicts, suffix_id_dicts


def get_words_affix_ids(sentence_words: List[str], prefix_dicts: List[Dict[str, int]], suffix_dicts: List[Dict[str, int]]) -> Tuple[List[List[int]], List[List[int]]]:
    def get_prefixes_suffixes(word: str) -> Tuple[List[str], List[str]]:
        prefixes = ['', '', '', '']
        suffixes = ['', '', '', '']
        if len(word) == 1:
            prefixes[0] = word[:1]
            suffixes[0] = word[-1:]
        elif len(word) == 2:
            prefixes[0] = word[:1]
            suffixes[0] = word[-1:]
            prefixes[1] = word[:2]
            suffixes[1] = word[-2:]
        elif len(word) == 3:
            prefixes[0] = word[:1]
            suffixes[0] = word[-1:]
            prefixes[1] = word[:2]
            suffixes[1] = word[-2:]
            prefixes[2] = word[:3]
            suffixes[2] = word[-3:]
        elif len(word) > 3:
            prefixes[0] = word[:1]
            suffixes[0] = word[-1:]
            prefixes[1] = word[:2]
            suffixes[1] = word[-2:]
            prefixes[2] = word[:3]
            suffixes[2] = word[-3:]
            prefixes[3] = word[:4]
            suffixes[3] = word[-4:]
        return prefixes, suffixes

    words_prefix_ids, words_suffix_ids = [], []
    for word in sentence_words:
        prefix_ids, suffix_ids = [], []
        prefixes, suffixes = get_prefixes_suffixes(word)
        for n, prefix in enumerate(prefixes):
            prefix_ids.append(prefix_dicts[n][prefix] if prefix in prefix_dicts[n] else 0)
        for n, suffix in enumerate(suffixes):
            suffix_ids.append(suffix_dicts[n][suffix] if suffix in suffix_dicts[n] else 0)
        words_prefix_ids.append(prefix_ids)
        words_suffix_ids.append(suffix_ids)
    return words_prefix_ids, words_suffix_ids
