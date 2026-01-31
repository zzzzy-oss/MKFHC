#!/usr/bin/env python
# coding: utf-8

# In[3]:


import os
# import pprint
import json
import copy
import time
import argparse

import numpy as np

import torch
import torch.nn as nn

from tqdm import tqdm
from functions_openllm import use_api_base, sure_infer,use_api_background,use_fusion_document,use_api_base_retrieval,use_api_base_retrievalAndfusion
from data_utils import get_em_f1

from transformers import AutoModelForCausalLM, AutoTokenizer


# parser = argparse.ArgumentParser(description='Query QA Data to GPT API.')
#     parser.add_argument('--data_name', type=str, default=None, help='Name of QA Dataset')
#     parser.add_argument('--qa_data', type=str, default=None, help='Path to QA Dataset')
#     parser.add_argument('--start', type=int, default=None, help='Start index of QA Dataset')
#     parser.add_argument('--end', type=int, default=None, help='End index of QA Dataset')
#     parser.add_argument('--lm_type', type=str, default='llama2', help='Type of LLM (llama2, gemma, mistral)')
#     parser.add_argument('--n_retrieval', type=int, default=10, help='Number of retrieval-augmented passages')
#     parser.add_argument('--infer_type', type=str, default='sure', help='Inference Method (base or sure)', choices=['base', 'sure'])
#     parser.add_argument('--output_folder', type=str, default=None, help='Path for save output files')
#     
#     args = parser.parse_args()

# In[4]:


from types import SimpleNamespace

# 模拟命令行参数
args = SimpleNamespace(
    data_name="nq",
    qa_data='../data/bm25/nq-test-bm25.json',
    start=None,
    end=None,
    lm_type='llama',
    n_retrieval=10,
    infer_type='base',
    output_folder='../data/bm25/'
)

# 现在你可以像使用 argparse 一样使用 args 对象
print(args.lm_type)
print(args.qa_data)


# In[5]:


#加载检索数据集
print("=====> Data Load...")
dataset = json.load(open(args.qa_data))
start_idx, end_idx = args.start, args.end
if start_idx is None:
    start_idx = 0
elif end_idx is None:
    end_idx = len(dataset)
else:
    if start_idx >= end_idx:
     raise ValueError
dataset = dataset[start_idx:end_idx]
print(dataset[0])
print("Number of QA Samples: {}".format(len(dataset)))


# In[6]:


#加载大模型生成文档
Ldataset = json.load(open("../data/bm25/nq_start0_endNone_base_ret10/backgroundLma70B.json"))
print("Number of backgroundQA Samples: {}".format(len(Ldataset)))


# In[7]:


#加载融合后的文档
#Fdataset = json.load(open("../data/dpr/nq_start0_endNone_base_ret10/FusionsLma70B.json"))
#print("Number of QA Samples: {}".format(len(dataset)))


# In[8]:


if args.lm_type == "gemma":
    model = AutoModelForCausalLM.from_pretrained("google/gemma-1.1-7b-it")
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-1.1-7b-it")
elif args.lm_type == "mistral":
    model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
elif args.lm_type == "llama":
    tokenizer = AutoTokenizer.from_pretrained("../pre/Llama3-70b/LLM-Research/Llama-3___3-70B-Instruct")
    model = AutoModelForCausalLM.from_pretrained("../pre/Llama3-70b/LLM-Research/Llama-3___3-70B-Instruct",device_map="auto",torch_dtype=torch.float16)
else:
    raise ValueError
#model = model.cuda()   上面已经指定了device_map="auto",torch_dtype=torch.float16，所以不需要再从内存中把模型移动到GPU上


# In[9]:


if not os.path.exists(args.output_folder):
    os.makedirs(args.output_folder)
method = f'{args.data_name}_start{start_idx}_end{end_idx}_{args.infer_type}_ret{str(args.n_retrieval)}'
method_folder = args.output_folder + '/{}'.format(method)
if not os.path.exists(method_folder):
    os.makedirs(method_folder)
method_folder


# In[10]:


print("=====> Begin Inference (type: {})".format(args.infer_type))
#让大模型生成相关背景文档
#print("=====> Begin generate")
#results = use_api_background(model, args.lm_type, tokenizer, dataset,n=1)

#让大模型生成融合后的文档
#print("=====> Begin fusion")
#results = use_fusion_document(model, args.lm_type, tokenizer, dataset,Ldataset)

#让大模型根据融合文档生成答案
#print("=====> Begin answer")
#results = use_api_base(model, args.lm_type, tokenizer, dataset,Fdataset)

#让大模型根据背景文档生成答案
results = use_api_base(model, args.lm_type, tokenizer, dataset,Ldataset)

#让大模型根据检索文档生成答案
#results = use_api_base_retrieval(model, args.lm_type, tokenizer, dataset,n_articles=10)

#让大模型根据检索文档和融合文档生成答案
#results = use_api_base_retrievalAndfusion(model, args.lm_type, tokenizer, dataset,Fdataset,n_articles=10)


#直接加载已有答案
#results = json.load(open("../data/2wiqa_start0_endNone_base_ret10/ResultLma70B.json"))
#results


# In[ ]:


print("=====> All Procedure is finished!")
with open(f'./{method_folder}/Result(background)Lma70B.json', "w", encoding='utf-8') as writer:
     writer.write(json.dumps(results, indent=4, ensure_ascii=False) + "\n")


# In[ ]:
print("=====> Results of {}".format(method))
em, f1 = get_em_f1(dataset, results)
print("EM: {} F1: {}".format(em.mean(), f1.mean()))

