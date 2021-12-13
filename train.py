import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import pickle
import time
import torch
from torch import optim
from typing import Tuple, List, Dict

import dataset
from model import BiLSTM_CRF
import utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_train',     type=str, default='./data/eng.train')
    parser.add_argument('--dataset_val',       type=str, default='./data/eng.testa')
    parser.add_argument('--dataset_test',      type=str, default='./data/eng.testb')
    parser.add_argument('--pretrained_glove',  type=str, default='./data/glove.6B.100d.txt')
    parser.add_argument('--output_mapping',    type=str, default='./output/mapping.pkl')
    parser.add_argument('--output_affix_list', type=str, default='./output/affix_list.json')

    parser.add_argument('--use_gpu',     type=bool, default=True)
    parser.add_argument('--model_path',  type=str, default='./model')

    args = parser.parse_args()
    args.use_gpu = args.use_gpu and torch.cuda.is_available()
    return args


def load_datasets(train_path: str, val_path: str, test_path: str, pretrained_glove: str, output_mapping: str, output_affix_list: str):
    train_sentences = dataset.load_sentences(train_path)
    val_sentences = dataset.load_sentences(val_path)[:1000]
    test_sentences = dataset.load_sentences(test_path)[:1000]

    dico_words, _, _ = dataset.word_mapping(train_sentences)
    _, char_to_id, _ = dataset.char_mapping(train_sentences)
    _, tag_to_id, id_to_tag = dataset.tag_mapping(train_sentences)
    _, word_to_id, _, word_embedding = dataset.load_pretrained_embedding(dico_words.copy(), pretrained_glove, word_dim=100)

    train_data = dataset.prepare_dataset(train_sentences, word_to_id, char_to_id, tag_to_id)
    val_data = dataset.prepare_dataset(val_sentences, word_to_id, char_to_id, tag_to_id)
    test_data = dataset.prepare_dataset(test_sentences, word_to_id, char_to_id, tag_to_id)
    prefix_dicts, suffix_dicts = dataset.add_affix_to_datasets(train_data, val_data, test_data)

    with open(output_mapping, 'wb') as f:
        mappings = {'word_to_id': word_to_id, 'tag_to_id': tag_to_id, 'char_to_id': char_to_id, 'word_embedding': word_embedding}
        pickle.dump(mappings, f)
    with open(output_affix_list, 'w') as f:
        json.dump([prefix_dicts, suffix_dicts], f, indent=2)

    print('Datasets status:')
    print('#train_data: {} / #val_data: {} / #test_data: {}'.format(len(train_data), len(val_data), len(test_data)))
    print('#word_to_id: {}, #char_to_id: {}, #tag_to_id: {}, #prefix_dicts: {}, #suffix_dicts: {}, '.format(len(word_to_id), len(char_to_id), len(tag_to_id), len(prefix_dicts), len(suffix_dicts)))
    print('#prefixes_2/3/4: [{}, {}, {}], #suffixes_2/3/4: [{}, {}, {}]'.format(len(prefix_dicts[1]), len(prefix_dicts[2]), len(prefix_dicts[3]), len(suffix_dicts[1]), len(suffix_dicts[2]), len(suffix_dicts[3])))
    return (train_data, val_data, test_data), (word_to_id, char_to_id, tag_to_id, id_to_tag), word_embedding, (prefix_dicts, suffix_dicts)


def train():
    pass


def evaluate():
    pass


def main():
    args = parse_args()
    device = 'cuda' if args.use_gpu else 'cpu'

    (train_data, val_data, test_data), (word_to_id, char_to_id, tag_to_id, id_to_tag), word_embedding, (prefix_dicts, suffix_dicts) = load_datasets(
        train_path=args.dataset_train,
        val_path=args.dataset_val,
        test_path=args.dataset_test,
        pretrained_glove=args.pretrained_glove,
        output_mapping=args.output_mapping,
        output_affix_list=args.output_affix_list,
    )

    pass


if __name__ == '__main__':
    main()
