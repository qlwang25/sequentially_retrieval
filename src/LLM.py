# coding=utf-8

import os
os.environ["CUDA_VISIBLE_DEVICES"]="1,3"
import time

import sys
sys.path.append("..")
sys.path.append("../../")
import csv

import torch
from transformer_utils.models.llama.tokenization_llama import LlamaTokenizer
from transformer_utils.models.llama.configuration_llama import LlamaConfig
from transformer_utils.utils.configuration_utils import GenerationConfig
from transformer_utils.models.llama.modeling_llama import LlamaModel, LlamaForCausalLM

from accelerate import init_empty_weights, load_checkpoint_and_dispatch, infer_auto_device_map


class LLMEvaluater(object):
    def __init__(self, LLM_name="LLaMA", file_path=None, model_path=None):
        self.device_map = {
                'model.embed_tokens': 0, 
                'model.layers.0': 0, 'model.layers.1': 0, 'model.layers.2': 0, 'model.layers.3': 0, 'model.layers.4': 0, 'model.layers.5': 0, 'model.layers.6': 0, 'model.layers.7': 0, 
                'model.layers.8': 0, 'model.layers.9': 0, 'model.layers.10': 0, 'model.layers.11': 0, 'model.layers.12': 0, 'model.layers.13': 0, 'model.layers.14': 0, 'model.layers.15': 0, 
                'model.layers.16': 0, 'model.layers.17': 0, 'model.layers.18': 0, 'model.layers.19': 0, 'model.layers.20': 0, 'model.layers.21': 0, 'model.layers.22': 0, 'model.layers.23': 0, 
                'model.layers.24': 0, 'model.layers.25': 0, 'model.layers.26': 0, 'model.layers.27': 0, 'model.layers.28': 0, 'model.layers.29': 0, 'model.layers.30': 1, 'model.layers.31': 1, 
                'model.layers.32': 1, 'model.layers.33': 1, 'model.layers.34': 1, 'model.layers.35': 1, 'model.layers.36': 1, 'model.layers.37': 1, 'model.layers.38': 1, 'model.layers.39': 1, 
                'model.norm': 1, 'lm_head': 1
                }
        self.file_path = file_path
        self.batch_size = 4

        if LLM_name == "LLaMA":
            self.model, self.tokenizer = self.load_LLaMA2(model_path)
        else:
            pass

    def load_LLaMA2(self, model_path):
        llama_tokenizer = LlamaTokenizer.from_pretrained(model_path, padding_side="left")
        llama_tokenizer.pad_token = llama_tokenizer.eos_token

        config = LlamaConfig.from_pretrained(model_path)
        config.max_length = 512
        # config.pad_token_id = config.eos_token_id

        with init_empty_weights(), torch.no_grad():
            llama_model = LlamaForCausalLM._from_config(config, torch_dtype=torch.bfloat16)
        llama_model.tie_weights()

        llama_model = load_checkpoint_and_dispatch(llama_model, model_path, device_map=self.device_map, dtype=torch.bfloat16)
        # llama_model = load_checkpoint_and_dispatch(llama_model, model_path, device_map="auto", no_split_module_classes=["LlamaDecoderLayer"], dtype=torch.bfloat16)
        llama_model = llama_model.eval()

        llama_model.generation_config = GenerationConfig.from_pretrained(pretrained_model_name=model_path, config_file_name='generation_config.json')
        llama_model.resize_token_embeddings(len(llama_tokenizer)+1)
        return llama_model, llama_tokenizer

    def inference(self, lines=[], prompt_index=1, output_index=-1):
        for i in range(len(lines)):
            if i % 20 == 0:
                print("Done {} ......".format(i))
            if i == 0 or lines[i][output_index] != "":
                pass
            else:
                prompts = [line[prompt_index] for line in lines[i:i+self.batch_size]]
                inputs = self.tokenizer(prompts, padding="longest", return_tensors="pt")
                padding_len = inputs["input_ids"].size(1)
                with torch.no_grad():
                    outputs = self.model.generate(input_ids=inputs['input_ids'].cuda(), attention_mask=inputs['attention_mask'].cuda(), max_length=padding_len + 80, do_sample=False, num_beams=1)
                    for k in range(len(prompts)):
                        predictions = self.tokenizer.decode(outputs[k, padding_len:], skip_special_tokens=True)
                        pred = predictions.split("\n")[0]
                        lines[i+k][output_index] = pred
                    self.write_csv(lines)

    def read_csv(self, current_len):
        lines = []
        with open(self.file_path, "r", encoding='utf-8') as file_obj: 
            reader_obj = csv.reader(file_obj) 
            for i, row in enumerate(reader_obj):
                if i == 0:
                    if len(row) == current_len:
                        row.insert(current_len, "meta-llama2-13b")
                    lines.append(row)
                else:
                    if len(row) == current_len:
                        row.insert(current_len, "")
                    lines.append(row)
        return lines

    def write_csv(self, lines):
        with open(self.file_path, "w", encoding='utf-8') as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            for line in lines:
                writer.writerow(line)

## TEST CODE
prompt = "Given a review, extract the aspect term(s) and determine their corresponding sentiment polarity. Here are some examples: \n"
prompt += "Review: It is always reliable , never bugged and responds well ." + "\n"
prompt += "Label:[[responds, positive]]" + "\n"
prompt += "Review: Enabling the battery timer is useless ." + "\n"
prompt += "Label:[[battery timer, negative]]" + "\n"
prompt += "Review: It rarely works and when it does it's incredibly slow ." + "\n"
prompt += "Label:[[works, negative]]" + "\n"
prompt += "Review: The machine is slow to boot up and occasionally crashes completely ." + "\n"
prompt += "Label:[[boot up, negative]]" + "\n"
prompt += "Review: Boot time is super fast , around anywhere from 35 seconds to 1 minute ." + "\n"
prompt += "Label:"

inputs = evaluater.tokenizer(prompt, return_tensors="pt")
_, txtlen = inputs['input_ids'].shape
outputs = evaluater.model.generate(input_ids=inputs['input_ids'].cuda())
predictions = evaluater.tokenizer.decode(outputs[:, txtlen:][0])
pred = predictions.split("\n")[0]
print(pred)
