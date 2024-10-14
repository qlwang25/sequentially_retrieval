# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import, division, print_function

import csv
import os
import sys
sys.path.append("../")
sys.path.append("../../")
import json
import re
import random
import pickle
from collections import defaultdict
import numpy as np
from random import shuffle

import torch
from torch.utils.data import TensorDataset
from transformer_utils.models.bert.tokenization_bert import BertTokenizer


from colorlogging import getLogger
logger = getLogger(__name__)


class ABSATokenizer(object):
    def __init__(self, plm_name, archive=None):
        if 'bert_' in plm_name:
            self.tokenizer = BertTokenizer.from_pretrained(archive)
            self.name = 'bert'
        else:
            raise ValueError


class InputExample(object):
    def __init__(self, guid, text, label=None):
        self.guid = guid
        self.text = text
        self.label = label


class InputFeatures(object):
    def __init__(self, input_ids=None, input_masks=None, input_seg_ids=None, label_id=None,):
        self.input_ids = input_ids
        self.input_masks = input_masks


class ABSAProcessor(object):
    def get_train_examples(self, data_dir):
        logger.info("***** Running training *****")
        return self._create_examples(os.path.join(data_dir, "train.json"), set_type="train")

    def get_test_examples(self, data_dir):
        logger.info("***** Running evaluation *****")
        return self._create_examples(os.path.join(data_dir, "test.json"), set_type="test")

    def _create_examples(self, file_path, set_type):
        with open(file_path, 'r') as json_file:
            datas = json.load(json_file)
        
        examples = []
        for k in range(len(datas)):
            guid = "%s-%s" % (set_type, k)
            examples.append(InputExample(guid=guid, text=datas[str(k)]['tokens'], label=datas[str(k)]['tag']))
        
        logger.info("***** example format *****")
        print(examples[0].text)
        print(examples[0].label)
        return examples


def convert_examples_to_features(examples, 
    max_seq_length, 
    tokenizer,
    mask_padding_with_zero=True,
    printN=1,
    ):
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    pad_token = tokenizer.pad_token
    mask_token = tokenizer.mask_token
    unk_token = tokenizer.unk_token
    cls_token_id = tokenizer.convert_tokens_to_ids([cls_token])[0]
    sep_token_id = tokenizer.convert_tokens_to_ids([sep_token])[0]
    pad_token_id = tokenizer.convert_tokens_to_ids([pad_token])[0]

    def inputIdMaskSegment(tmp_text=None):
        tokens = []
        for tok in tmp_text.split():
            tokens.extend(tokenizer.wordpiece_tokenizer.tokenize(tok))
        if len(tokens) > max_seq_length - 2:
            tokens = tokens[:(max_seq_length - 2)]

        tokens = [cls_token] + tokens + [sep_token]        
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        padding_length = max_seq_length - len(input_ids)
        input_ids += ([pad_token_id] * padding_length)
        input_mask += ([0] * padding_length)
        assert len(input_ids) == len(input_mask) == max_seq_length
        return input_ids, input_mask

    features = []
    for (ex_index, example) in enumerate(examples):

        input_ids, input_masks = inputIdMaskSegment(tmp_text=example.text.lower())
        features.append(InputFeatures(input_ids=input_ids, input_masks=input_masks,))

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_masks = torch.tensor([f.input_masks for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_input_masks)
    return dataset


