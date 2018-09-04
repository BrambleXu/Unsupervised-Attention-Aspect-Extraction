import re
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from pathlib import Path


def soup2dict(sentence_nodes):
    """
    Input: a soup object, e.g. soup.find_all("sentence")
    Output: a list of dictionaries, contains id, text, aspect terms
    """
    sentences = []
    i = 0
    for n in sentence_nodes:
        i += 1
        aspect_term = []
        sentence['text'] = n.find('text').string
        if n.find('Opinions'):
            category_term = []
            for c in n.find('Opinions').contents:
                if c.name == 'Opinion':
                    sentence['polarity'] = c['polarity']
                    if c['category'] not in category_term:
                        category_term.append(c['category'])
                    if c['target'] not in aspect_term:
                        aspect_term.append(c['target'])

        sentence['category'] = category_term
        sentences.append(sentence)

    return sentences


def dict2labels(sentences):
    """
    Input:
        sentences: a list of dictionaries
    Output:
        label_df: a dataframe contains multiple labels
        label_kinds: contain all labels categories

    """
    # all layer to a list of label
    label_for_sentences = []
    for s in sentences:
        label_for_sentences.append(s['category'])

    # get all categories
    label_kinds = set()
    for s in label_for_sentences:
        for c in s:
            label_kinds.add(c)

            # convert all labels to one-hot format
    labels = []
    for sentence_labels in label_for_sentences:
        if len(sentence_labels) > 0:
            row = {}
        for k in label_kinds:
            if k in sentence_labels:
                row[k] = 1
            else:
                row[k] = 0
        labels.append(row)

    labels_df = pd.DataFrame(labels)

    return labels_df


def get_labels(file_path):
    # Get soup object
    soup = None
    with file_path.open(encoding="utf-8") as f:
        soup = BeautifulSoup(f.read().strip(), "lxml-xml")
    if soup is None:
        raise Exception("Can't read xml file")
    sentence_nodes = soup.find_all("sentence")

    # soup obejct to a list of dictionaries
    sentences = soup2dict(sentence_nodes)

    # a list of dictionaries to dataframe
    label_df = dict2labels(sentences)

    return label_df


if __name__ == "__main__":

    # Train and Test Path
    train_path = Path.cwd().parent.joinpath('datasets/raw-semeval-2016/train.xml')
    test_path = Path.cwd().parent.joinpath('datasets/raw-semeval-2016/test.xml')

    # Get labels
    train_label_df = get_labels(train_path)
    test_label_df = get_labels(test_path)

    # Save data
    train_path = '../datasets/semeval-2016/slot1/train_label_df.csv'
    test_path = '../datasets/semeval-2016/slot1/test_label_df.csv'

    train_label_df.to_csv(train_path, encoding='utf-8', index=False)
    test_label_df.to_csv(test_path, encoding='utf-8', index=False)

    # Load data
    # train_path = '../datasets/semeval-2016/slot1/train_label_df.csv'
    # test_path = '../datasets/semeval-2016/slot1/test_label_df.csv'
    #
    # train_labels = pd.read_csv(train_path)
    # test_labels = pd.read_csv(test_path)


