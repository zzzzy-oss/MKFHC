import pprint
import json
import copy
import numpy as np
from tqdm import tqdm
import time
import openai

from datetime import timedelta, datetime

################## Basic Functions ##################

def build_prompt(input):
    '''
    llama2的promote
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    SYSTEM_PROMPT = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being factually coherent. If you don't know the answer to a question, please don't share false information."

    return f"{B_INST} {B_SYS}{SYSTEM_PROMPT}{E_SYS}{input} {E_INST} " 
    '''
    #llama3的promote
    B_text ="<|begin_of_text|>"
    S_header, E_header="<|start_header_id|>" ,"<|end_header_id|>"
    Eot = "<|eot_id|>"
    SYSTEM_PROMPT = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being factually coherent. If you don't know the answer to a question, please don't share false information."
    return f"{B_text}{S_header}system{E_header}\n{SYSTEM_PROMPT}{Eot}{S_header}user{E_header}\n{input}{Eot}{S_header}assistant{E_header}"

          
def api_query(model, model_type, tokenizer, query, return_str=False):
    if model_type == 'llama':
        if type(query) == list:
            inputs = [build_prompt(inp) for inp in query]
        else:
            inputs = [build_prompt(query)]

        input_ids = tokenizer(inputs, return_tensors="pt", add_special_tokens=True)['input_ids'].cuda()
    elif model_type == 'mistral':
        query = [{"role": "user", "content": query}]
        encodeds = tokenizer.apply_chat_template(query, return_tensors="pt")
        input_ids = encodeds.to("cuda")
    elif model_type == 'gemma':
        input_ids = tokenizer(query, return_tensors="pt", add_special_tokens=True)['input_ids'].cuda()
    else:
        raise ValueError
    
    generate_ids = model.generate(inputs=input_ids,
        return_dict_in_generate=True,
        do_sample=False,
        num_beams=1,
        max_new_tokens=512,
    )
    outputs = tokenizer.batch_decode(generate_ids.sequences[:,input_ids.shape[1]:], skip_special_tokens=True)

    for i, ans in enumerate(outputs):
        if ans.strip() == "":
            outputs[i] = 'N/A'

    if return_str and len(outputs) == 1:
        return outputs[0]
    else:
        return outputs


# Note. Without constraints on output words, LLMs are too verbose as we consider zero-shot setup. 
# Therefore, we commonly apply constraints on number of words for both baseline and ours.

def get_query_baseline(dataset, idx, n_articles=10, start_idx=0):
    data = dataset[idx]
    len_ctxs = len(data['contexts'])
    if n_articles > 0:
        text = ""
        for i in range(start_idx, start_idx+n_articles):
            idx_ctx = (i % len_ctxs)
            text += f"Passage #{i+1} Title: {data['contexts'][idx_ctx]['title']}\nPassage #{i+1} Text: {data['contexts'][idx_ctx]['text']} \n\n"
        text += f"Task description: predict the answer to the following question. Do not exceed 3 words."
        text += f"\n\nQuestion: {data['question']}."
        text += f"\n\nAnswer: "
    else:
        text = f"Task description: predict the answer to the following question. Do not exceed 3 words."
        text += f"\n\nQuestion: {data['question']}."
        text += f"\n\nAnswer: "
        
    return text

def get_query_baselineAndFusion(dataset,Fdataset, idx, n_articles=10, start_idx=0):
    data = dataset[idx]
    len_ctxs = len(data['contexts'])
    text = ""
    text += f"This is the relevant documentation:\n"
    text += f"{Fdataset[idx][0]}\n\n"
    if n_articles > 0:    
        for i in range(start_idx, start_idx+n_articles):
            idx_ctx = (i % len_ctxs)
            text += f"Passage #{i+1} Title: {data['contexts'][idx_ctx]['title']}\nPassage #{i+1} Text: {data['contexts'][idx_ctx]['text']} \n\n"
        text += f"Task description: predict the answer to the following question. Do not exceed 3 words."
        text += f"\n\nQuestion: {data['question']}."
        text += f"\n\nAnswer: "
    else:
        text = f"Task description: predict the answer to the following question. Do not exceed 3 words."
        text += f"\n\nQuestion: {data['question']}."
        text += f"\n\nAnswer: "
        
    return text

def get_answer_base(dataset,Fdataset, idx, n_articles=10, start_idx=0):
    data = dataset[idx]
    len_ctxs = len(data['contexts'])
    text=""
    #text+= f"This is the retrieval of n articles:\n"
    #for i in range(start_idx, start_idx+n_articles):
    #    idx_ctx = (i % len_ctxs)
    #    text += f"Passage #{i+1} Title: {data['contexts'][idx_ctx]['title']}\nPassage #{i+1} Text: {data['contexts'][idx_ctx]['text']} \n\n"
    text += f"This is the relevant documentation:\n"
    text += f"{Fdataset[idx][0]}\n\n"
    text += f"Task description: Please predict the answer to the following question based on the document provided above. Do not exceed 3 words."
    text += f"\n\nQuestion: {data['question']}."
    text += f"\n\nAnswer: "
    
    return text
def get_fusion_document_base(dataset,Ldataset, idx, n_articles=10, start_idx=0):
    data = dataset[idx]
    len_ctxs = len(data['contexts'])
    text = f"This is the retrieval of n articles:\n"
    for i in range(start_idx, start_idx+n_articles):
        idx_ctx = (i % len_ctxs)
        text += f"Passage #{i+1} Title: {data['contexts'][idx_ctx]['title']}\nPassage #{i+1} Text: {data['contexts'][idx_ctx]['text']} \n\n"
    text += f"This is the relevant documentation generated by the large model:\n"
    text += f"{Ldataset[idx][0]}\n\n"
    text += f"Task description: the retrieved articles are certainly authentic, but they may have lower relevance to the question. The documentation generated by the large model is certainly relevant to the issue, but it may contain false information. Now, I need you to summarize the n retrieved articles and the documentation generated by the large model, and create a new document to answer the question. When there is a conflict in content between the retrieved articles and the large model-generated documentation, you should believe the retrieved articles.\n\n"
    text += f"Only generate the document, without answering the question."
    text += f"\n\nQuestion: {data['question']}."
    text += f"\n\nThe summarized document generated: "
    
    return text
def get_background_base(dataset, idx, n=1):
    data = dataset[idx]
    text = ""
    text+= f"Generate a background document from Wikipedia to answer the given question.Only provide the background document in the required format; no need to answer the question."
    text+= f"\n\nQuestion: {data['question']}."
    text+= f"\n\nDocument: "
    
    return text


#检索内容和大模型生成内容融合
def use_fusion_document(model, model_type, tokenizer, dataset,Ldataset, n_articles=10, start_idx=0):
    res = []
    queries = []
    
    for i, example in enumerate(dataset):
        query = get_fusion_document_base(dataset,Ldataset, i, n_articles, start_idx)
        queries.append(query)
    
    for query in tqdm(queries):
        answer = api_query(model, model_type, tokenizer, query)
        res.extend([[ans] for ans in answer])

    return res

#生成背景文档
def use_api_background(model, model_type, tokenizer, dataset, n=1):
    res = []
    queries = []
    
    for i, example in enumerate(dataset):
        query = get_background_base(dataset, i)
        queries.append(query)
    
    for query in tqdm(queries):
        answer = api_query(model, model_type, tokenizer, query)
        res.extend([[ans] for ans in answer])

    return res

def use_api_base(model, model_type, tokenizer, dataset,Fdataset, n_articles=10, start_idx=0):
    res = []
    queries = []

    for i, example in enumerate(dataset):
        query = get_answer_base(dataset,Fdataset, i, n_articles, start_idx)
        queries.append(query)

    for query in tqdm(queries):
        answer = api_query(model, model_type, tokenizer, query)
        res.extend([[ans] for ans in answer])

    return res

def use_api_base_retrieval(model, model_type, tokenizer, dataset, n_articles=10, start_idx=0):
    res = []
    queries = []
    
    for i, example in enumerate(dataset):
        query = get_query_baseline(dataset, i, n_articles, start_idx)
        queries.append(query)

    for query in tqdm(queries):
        answer = api_query(model, model_type, tokenizer, query)
        res.extend([[ans] for ans in answer])

    return res

def use_api_base_retrievalAndfusion(model, model_type, tokenizer, dataset,Fdataset, n_articles=10, start_idx=0):
    res = []
    queries = []
    
    for i, example in enumerate(dataset):
        query = get_query_baselineAndFusion(dataset,Fdataset, i, n_articles, start_idx)
        queries.append(query)
    
    for query in tqdm(queries):#
        answer = api_query(model, model_type, tokenizer, query)
        res.extend([[ans] for ans in answer])

    return res