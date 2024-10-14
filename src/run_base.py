# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team and authors from University of Illinois at Chicago.
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



import os
import sys
sys.path.append('../')
sys.path.append('../../')
import logging
import argparse
import random
import json
import csv

import torch
import numpy as np
from torch.autograd import Function
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from transformer_utils.models.bert.modeling_bert import BertPreTrainedModel, BertModel
from utils import ABSATokenizer, ABSAProcessor, convert_examples_to_features
from LLM import LLMEvaluater
import modelconfig
from optimization import AdamW, WarmupLinearSchedule, Warmup


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)


class DemonstrationSelector(BertPreTrainedModel):
    def __init__(self, config, tokenizer, max_seq_length, shot_num, set_num, device):
        super(DemonstrationSelector, self).__init__(config)
        config.clue_num = 0
        self.config = config
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.gru_layer = 2
        self.shot_num = shot_num
        self.set_num = set_num
        self.device = device
        self.beta = 1
        self.printN = 1

        self.bert = BertModel(config, output_attentions=False)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.decoder = torch.nn.GRU(input_size=config.hidden_size, hidden_size=config.hidden_size, num_layers=self.gru_layer, bias=True, batch_first=True)        
        self.apply(self.init_bert_weights)

    def get_example_representations(self, batch_size, examples):
        Tersorsets = convert_examples_to_features(examples, self.max_seq_length, self.tokenizer, mask_padding_with_zero=True, printN=self.printN,)
        self.printN = self.printN + 1
        dataloader = DataLoader(Tersorsets, sampler=SequentialSampler(Tersorsets), batch_size=batch_size)
        hiddens = []
        for step, batch in enumerate(dataloader):
            input_ids, input_masks = tuple(t.to(self.device) for t in batch)
            outputs = self.bert(input_ids, attention_mask=input_masks, output_all_encoded_layers=False,)
            _, pooled_output = outputs 
            hiddens.append(self.dropout(pooled_output))
        hidden_matrixs = torch.cat(hiddens, dim=0)
        torch.cuda.empty_cache()
        return hidden_matrixs

    def retrieval_demos(self, initial_index, initial_hidden, hidden_matrixs, set_num, mode="train"):
        K = set_num
        T = self.shot_num
        example_num = hidden_matrixs.size(0)
    
        query_input = initial_hidden.unsqueeze(0).unsqueeze(0).repeat(K, 1, 1)
        current_input = query_input
        hx = torch.zeros((self.gru_layer, K, self.config.hidden_size)).to(self.device)

        all_sequences = []
        all_logits = []
        for t in range(T):
            output, hx = self.decoder(current_input, hx)
            similarities = torch.matmul(input=output.squeeze(1), other=hidden_matrixs.t())
            probabilities = torch.nn.functional.softmax(similarities, dim=1)

            mask = torch.ones_like(probabilities).bool().to(self.device)
            for idxs in all_sequences:
                mask[torch.arange(K), idxs] = False
            if mode == "train":
                mask[torch.arange(K), initial_index] = False

            probabilities = probabilities.masked_fill(mask == False, 0.0)
            if mode == "train":
                alpha = 0.1
                uniform_distribution = torch.full(probabilities.shape, 1 / example_num).to(self.device)
                uniform_distribution = uniform_distribution.masked_fill(mask == False, 0.0)
                adjusted_probabilities = (1 - alpha) * probabilities + alpha * uniform_distribution
                next_indexs = torch.multinomial(adjusted_probabilities, 1).squeeze(1)
            if mode == "test":
                next_indexs = torch.argmax(probabilities, dim=1)

            all_sequences.append(next_indexs)
            all_logits.append(torch.log(probabilities[torch.arange(K), next_indexs]))
            current_input = query_input

        tuples = [tuple(row.tolist()) for row in torch.stack(all_sequences, dim=1)]
        tuples_logits = [torch.sum(row) for row in torch.stack(all_logits, dim=1)]
        return tuples, tuples_logits

    def probability_Feedback_from_LLM(self, index, examples, retrieval_results, evaluater):
        query = examples[index].text
        query_label = examples[index].label

        prompts, labels = [], []
        for demos_index in retrieval_results:
            demos = [examples[p] for p in demos_index]
            prompt = "Given a review, extract the aspect term(s) and determine their corresponding sentiment polarity. Here are some examples:\n"
            for demo in demos:
                prompt += "Review: {}\n".format(demo.text)
                prompt += "Label:{}\n".format(str(demo.label).replace("'", ""))
            prompt += "Review: {}\n".format(query)
            label = str(query_label).replace("'", "")
            prompt += "Label:{}".format(label)
            prompts.append(prompt)
            labels.append(label)

        performance_evalution = []
        with torch.no_grad():
            inputs = evaluater.tokenizer(prompts, padding="longest", return_tensors="pt")
            outputs = evaluater.model(input_ids=inputs['input_ids'].to(self.device), attention_mask=inputs['attention_mask'].to(self.device))
            logits = outputs.logits
            for i, logit in enumerate(logits):
                gold = evaluater.tokenizer(labels[i], return_tensors="pt")['input_ids'].to(self.device)
                gold_len = gold.size(1)
                probability = torch.nn.functional.softmax(logit[-gold_len:, :], dim=-1)
                log_probs = torch.gather(input=probability, dim=-1, index=gold.t())
                average_entropy = torch.mean(log_probs)
                performance_evalution.append(average_entropy)

            tensor_list = torch.stack(performance_evalution)
            mean = torch.mean(tensor_list)
            std = torch.std(tensor_list)
            standardized_list = (tensor_list - mean) / std
            return standardized_list

    def parse_list_string(self, list_string):
        trimmed = list_string.strip().strip("[]")
        items = trimmed.split('], [')
        result = []
        for item in items:
            words = item.replace("'", "").replace("\"", "").split(', ')
            if len(words) == 2:
                result.append([words[0], words[-1]])
        return result

    def getReward(self, pred, label):
        pred_list = self.parse_list_string(pred.lower())
        label_list = self.parse_list_string(label.lower())

        reward = 0
        if pred_list == label_list:
            return 5.0  # 满分        
        for lab in label_list:
            if lab in pred_list:
                reward += 2
            else:
                for out in pred_list:
                    if out[0] == lab[0]:
                        reward += 1.5
                    elif out[0] in lab[0]:
                        reward += 0.5
                    else:
                        pass
        reward -= abs(len(pred_list) - len(label_list))
        reward = 0 if reward < 0 else reward
        if (self.printN - 1) % (self.shot_num * 50 ) == 0:
            print(pred_list, label_list, reward)
        return reward

    def reward_Feedback_from_LLM(self, index, examples, retrieval_results, evaluater):
        query = examples[index].text
        query_label = examples[index].label

        prompts, labels = [], []
        for demos_index in retrieval_results:
            demos = [examples[p] for p in demos_index]
            prompt = "Given a review, extract the aspect term(s) and determine their corresponding sentiment polarity. Here are some examples:\n"
            for demo in demos:
                prompt += "Review: {}\n".format(demo.text)
                prompt += "Label:{}\n".format(str(demo.label).replace("'", ""))
            prompt += "Review: {}\n".format(query)
            prompt += "Label:"
            prompts.append(prompt)
            label = str(query_label).replace("'", "")
            labels.append(label)

        performance_evalution = []
        with torch.no_grad():
            inputs = evaluater.tokenizer(prompts, padding="longest", return_tensors="pt")
            padding_len = inputs["input_ids"].size(1)
            outputs = evaluater.model.generate(input_ids=inputs['input_ids'].to(self.device), attention_mask=inputs['attention_mask'].to(self.device), max_length=padding_len + 80, do_sample=False, num_beams=1)
            for i, output in enumerate(outputs):
                pred = evaluater.tokenizer.decode(output[padding_len:], skip_special_tokens=True)
                pred = pred.split("\n")[0]
                performance_evalution.append(self.getReward(pred=pred, label=labels[i]))

            if len(set(performance_evalution)) == 1:
                performance_evalution[0] += 1.5
            tensor_list = torch.tensor(performance_evalution).float().to(self.device)
            mean = torch.mean(tensor_list)
            std = torch.std(tensor_list)
            standardized_list = (tensor_list - mean) / std
            standardized_list = torch.where(standardized_list <= 0, 0, standardized_list)
            performance_evalution = standardized_list.tolist()
            return performance_evalution


def get_AdamW(args, named_parameters, learning_rate, steps):
    param_optimizer = [(k, v) for k, v in named_parameters if v.requires_grad == True]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=args.adam_epsilon)
    scheduler = Warmup[args.schedule](optimizer, warmup_steps=args.warmup_steps, t_total=steps)
    return optimizer, scheduler


def train(args, train_examples, model):
    evaluater = LLMEvaluater(LLM_name="LLaMA", model_path="../../pre-trained_models/meta-llama2-7b-chat")

    indexs = torch.arange(len(train_examples), dtype=torch.long)
    INDEX = TensorDataset(indexs)
    index_dataloader = DataLoader(INDEX, sampler=RandomSampler(INDEX), batch_size=args.batch_size)
    num_train_steps = int(len(index_dataloader)) * args.epochs
    logger.info("Total optimization steps = %d", num_train_steps)

    bert_optimizer, bert_scheduler = get_AdamW(args, model.bert.named_parameters(), learning_rate=1e-5, steps=args.epochs)
    decd_optimizer, decd_scheduler = get_AdamW(args, model.decoder.named_parameters(), learning_rate=4e-5, steps=num_train_steps)

    model.zero_grad()
    model.train()
    global_step = 0
    torch.autograd.set_detect_anomaly(True)
    for epoch in range(int(args.epochs)):
        example_hidden_matrixs = model.get_example_representations(args.batch_size, train_examples)
        final_loss = [] 

        for step, batch in enumerate(index_dataloader):
            losses = []
            for index in batch[0]:
                index = index.item()

                retrieval_results, results_logits = model.retrieval_demos(initial_index=index, 
                    initial_hidden=example_hidden_matrixs[index], 
                    hidden_matrixs=example_hidden_matrixs, 
                    set_num=model.set_num, 
                    mode="train",
                    )

                # REINFORCE
                performance_indicate = model.reward_Feedback_from_LLM(index, train_examples, retrieval_results, evaluater)
                for i in range(len(performance_indicate)):
                    cur_loss = - results_logits[i] * performance_indicate[i]
                    losses.append(cur_loss) 

            stacked_loss = torch.stack(losses)
            loss = torch.mean(stacked_loss, dim=0)
            loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), args.max_grad_norm)
            decd_optimizer.step()
            decd_scheduler.step()
            final_loss.append(loss)

            if global_step % args.logging_global_step == 0:
                logger.info("Epoch:{}, Global Step:{}/{}, Loss:{:.5f}".format(epoch, global_step, num_train_steps, loss.item()))
                torch.save(model.state_dict(), os.path.join(args.data_dir, 'pytorch_model.bin'))
            global_step += 1

        final_loss = torch.stack(final_loss)
        final_loss = torch.mean(final_loss, dim=0)
        final_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.bert.parameters(), args.max_grad_norm)
        bert_optimizer.step()
        bert_scheduler.step()
        model.zero_grad()

        torch.cuda.empty_cache()
    del evaluater    



def evaluate(args, train_examples, test_examples, model):
    with torch.no_grad():
        train_example_hiddens = model.get_example_representations(args.batch_size, train_examples)
        test_example_hiddens = model.get_example_representations(args.batch_size, test_examples)

        with open(os.path.join(args.data_dir, "retrieval_results.csv"), mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(["Id", "Prompt", "Label"])            
            for index in range(len(test_examples)):
                retrieval_results, _ = model.retrieval_demos(initial_index=None, 
                    initial_hidden=test_example_hiddens[index], 
                    hidden_matrixs=train_example_hiddens, 
                    set_num=1, 
                    mode="test",
                    )

                prompt = "Given a review, extract the aspect term(s) and determine their corresponding sentiment polarity. Here are some examples:\n"
                for i in retrieval_results[0]:
                    prompt += "Review: {}\n".format(train_examples[i].text)
                    prompt += "Label:{}\n".format(str(train_examples[i].label).replace("'", ""))
                prompt += "Review: {}\n".format(test_examples[index].text)
                prompt += "Label:"
                label = str(test_examples[index].label).replace("'", "")
                writer.writerow([str(index), prompt, label])


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--plm_model", default='bert_base', type=str)
    parser.add_argument("--data_dir", default=None, type=str, required=False, help="The input data dir containing json files.")
    parser.add_argument("--schedule", default="WarmupLinearSchedule", type=str)

    parser.add_argument("--max_grad_norm", default=10.0, type=float, help="Max gradient norm.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float, help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument('--logging_global_step', type=int, default=30, help="Log every X updates steps.")

    ## Other parameters
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument("--max_seq_length", default=128, type=int, help="The maximum total input sequence length after WordPiece tokenization.")
    parser.add_argument("--do_train", default=False, action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval", default=True, action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--batch_size", default=16, type=int, help="Total batch size for training and testing.")
    parser.add_argument("--learning_rate", default=3e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--epochs", default=30, type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--demons_pool_size", default=100, type=int)
    parser.add_argument("--shot_num", default=4, type=int)
    parser.add_argument("--set_num", default=15, type=int)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    args.device0 = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    Instance = ABSATokenizer(plm_name='bert_base', archive=modelconfig.MODEL_ARCHIVE_MAP[args.plm_model])
    args.tokenizer = Instance.tokenizer

    logger.info("Creating features from dataset file at %s", args.data_dir)
    processor = ABSAProcessor()
    train_examples = processor.get_train_examples(args.data_dir)
    if args.demons_pool_size != -1:
        train_examples = train_examples[:args.demons_pool_size]
    logger.info("  Num examples = %d", len(train_examples))

    selector = DemonstrationSelector.from_pretrained(modelconfig.MODEL_ARCHIVE_MAP[args.plm_model],
        tokenizer=args.tokenizer, 
        max_seq_length=args.max_seq_length,
        shot_num=args.shot_num,
        set_num=args.set_num,
        device=args.device0,)
    selector.to(args.device0)

    if args.do_train:
        train(args, train_examples, model=selector)

    if args.do_eval:
        logger.info("  load checkpoint.... %s", os.path.join(args.data_dir, 'pytorch_model.bin'))
        evaluate(args, train_examples, test_examples, model=selector)

if __name__ == "__main__":
    main()