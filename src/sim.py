# coding=utf-8

from __future__ import absolute_import, division, print_function

import argparse
import logging
import codecs
import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"

import random
import sys
sys.path.append('../')
sys.path.append("../../")
import csv

import numpy as np
import torch.nn.functional as F
import torch

from transformer_utils.models.bert.modeling_bert import BertPreTrainedModel, BertModel
from transformer_utils.models.bert.tokenization_bert import BertTokenizer


logger = logging.getLogger(__name__)
torch.set_num_threads(12)


class BertForSequence(BertPreTrainedModel):
    def __init__(self, config, tokenizer, max_seq_length, clue_num=0,):
        super(BertForSequence, self).__init__(config)
        config.clue_num = clue_num
        self.config = config
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.ex_index = 0
        self.bert = BertModel(config, output_attentions=False)
        self.apply(self.init_bert_weights)

    def convert_text_to_feature(self, text):
        cls_token = '[CLS]'
        sep_token = '[SEP]'
        pad_token_id = 0

        tokens = []
        for tok in text.split(" "):
            tokens.extend(self.tokenizer.wordpiece_tokenizer.tokenize(tok))
        if len(tokens) > self.max_seq_length - 2:
            tokens = tokens[:(self.max_seq_length - 2)]

        tokens = [cls_token] + tokens + [sep_token]        
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        padding_length = self.max_seq_length - len(input_ids)
        input_ids += ([pad_token_id] * padding_length)
        input_mask += ([0] * padding_length)
        assert len(input_ids) == len(input_mask) == self.max_seq_length
        
        if self.ex_index < 1:
            logger.info("tokens: %s" % text)
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            self.ex_index += 1
        return input_ids, input_mask

    def get_representations(self, examples):
        all_input_ids, all_input_masks = [], []
        for text in examples:
            input_id, input_mask = self.convert_text_to_feature(text)
            input_id = torch.tensor(input_id).unsqueeze(0).cuda()
            input_mask = torch.tensor(input_mask).unsqueeze(0).cuda()

            all_input_ids.append(input_id)
            all_input_masks.append(input_mask)

        input_ids = torch.cat(all_input_ids, dim=0)
        input_masks = torch.cat(all_input_masks, dim=0)
        _, pool_output = self.bert(input_ids=input_ids, attention_mask=input_masks, output_all_encoded_layers=False)
        return pool_output


def main():
    parser = argparse.ArgumentParser()

    ## Other parameters
    parser.add_argument("--max_seq_length", default=128, type=int, help="The maximum total input sequence length after WordPiece tokenization.")
    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")

    parser.add_argument("--pad_token_label_id", default=-1, type=int, help="id of pad token .")
    parser.add_argument("--logging_global_step", default=50, type=int)
    args = parser.parse_args()

    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt = '%m/%d/%Y %H:%M:%S', level = logging.INFO)

    model_path = "../../pre-trained_models/bert_uncased_L-12_H-768_A-12"
    bert_tokenizer = BertTokenizer(os.path.join(model_path, "vocab.txt"), do_lower_case=args.do_lower_case)
    bert_model = BertForSequence.from_pretrained(model_path, tokenizer=bert_tokenizer, max_seq_length=args.max_seq_length)
    bert_model.cuda()
    
    files = ["../llama-13b_result/result_100_rest14_4/similarity_results_100_llama-13b_rest14_random_each_4.csv"]
    for csv_path in files:
        print(csv_path)
        file_inter_sims, file_intra_sims = [], []
        with open(csv_path, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.reader(file)
            next(reader)
            for i, row in enumerate(reader):
                ipt = row[1].lower().strip().split("\n")
                assert len(ipt) == 11
                query = [ipt[9][8:]]
                query_embed = bert_model.get_representations(query)

                examples = [ipt[k][8:] for k in [1, 3, 5, 7]]
                example_embeds = bert_model.get_representations(examples)

                query_expanded = query_embed.expand_as(example_embeds)
                cos_similarities = F.cosine_similarity(query_expanded, example_embeds, dim=1)
                average_tmp = torch.mean(cos_similarities)
                file_inter_sims.append(average_tmp.item())

                cos_similarities = []
                for ii in range(4):
                    for jj in range(ii+1, 4):
                        cos_sim = F.cosine_similarity(example_embeds[ii].unsqueeze(0), example_embeds[jj].unsqueeze(0))
                        cos_similarities.append(cos_sim)
                average_tmp = torch.mean(torch.tensor(cos_similarities))
                file_intra_sims.append(average_tmp.item())


        mean_inter = sum(file_inter_sims) / len(file_inter_sims)
        mean_intra = sum(file_intra_sims) / len(file_intra_sims)
        print("mean_inter:", mean_inter, "mean_intra:", mean_intra)


if __name__ == "__main__":
    main()


