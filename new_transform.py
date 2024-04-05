import numpy as np
from typing import Optional, List
import torch.nn as nn
from transformers import AutoTokenizer, HfArgumentParser, pipeline
from tqdm import tqdm
from datasets import load_dataset
import torch
import json
from typing import Optional
from dataclasses import dataclass, field
import time
from transformers import AutoTokenizer, HfArgumentParser, pipeline, DataCollatorForSeq2Seq


import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def calculate_text_similarity(text1, text2):
    """
    计算两个文本的余弦相似度。

    参数:
    text1 (str): 第一个文本
    text2 (str): 第二个文本

    返回:
    float: 两个文本之间的余弦相似度
    """
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])

    return similarity[0][0]

#####
# This script takes a dataset as the input, where each sample is {"input": "the pormpt", "output": ["response1", "response2", "response3", ...]}
# The script will compute the reward for each input-output pair, and eventually output a new dataset, where each sample contains {"input": "the pormpt", "output": ["response1", "response2", "response3", ...], "rewards": [reward1, reward2, ...]}
# Due to memory constraint, we will set the reward of the input+output that is longer than 800 tokens as -999999, which should be discarded in later processing. It should be at most ~2% samples that are discarded.
#####

@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """
    dataset_path: Optional[str] = field(
        default="/home/xx/xw/rsf/rsf_hf_mistral_rsf_baseline/model0/data/gen_data",
        metadata={"help": "the location of the dataset name or path"},
    )
    output_dir: Optional[str] = field(
        default="",
        metadata={"help": "the location of the output file"},
    )
    my_idx: Optional[str] = field(
        default="",
        metadata={"help": "the location of the output file"},
    )
from collections import defaultdict


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

data_ret = defaultdict(list)

repeat_num_0 = 0
old_ds = load_dataset("weqweasdas/ultra_prompt_split", split="prompt" + script_args.my_idx).select(range(9999))
all_prompt = [x[0]['content'] for x in old_ds['prompt']]
print(len(all_prompt))
print(all_prompt[0])
gathered_data = []
for j in range(4):
    ds = load_dataset("json", data_files=script_args.dataset_path + str(j) + ".json", split="train",
                      field="instances")  # .select(range(500))
    for sample in ds:
        #clean_prom = sample['old_prompt'][0]['content'] #
        #clean_prom = sample['prompt'].replace("<bos><start_of_turn>user\n", "").replace("</s>\n", "").replace("<end_of_turn>\n<start_of_turn>model\n", "")
        clean_prom = sample['prompt'].replace("<|user|>\n", "").replace(
        "</s>\n", "").replace("<|assistant|>\n", "")
        
        if len(sample['responses']) < 8:
            continue
        if clean_prom in all_prompt:
            gathered_data.append({"prompt": sample['prompt'], "responses": sample['responses']})
        else:
            #all_similiairities = np.array([cosine_similarity_strings(clean_prom, z) for z in all_prompt])
            #if np.sum(all_similiairities > 0.85) > 0:
            try:
                #print(clean_prom, sample['old_prompt'][0]['content'])
                
                if calculate_text_similarity(clean_prom, sample['old_prompt'][0]['content']) > 0.85:
                    #print("yes")
                    #data_ret[sample['prompt']].append(sample['responses'][0])
                    gathered_data.append({"prompt": sample['prompt'], "responses": sample['responses']})
                    
            except:
                pass
            #print(clean_prom,  "@@@@", sample['old_prompt'][0]['content'], "\n\n\n##############")
            #time.sleep(5)




print(len(gathered_data))


#print(len(gathered_data))


output_eval_dataset = {}
output_eval_dataset['type'] = 'text_only'
output_eval_dataset['instances'] = gathered_data
with open(script_args.dataset_path + ".json", 'w', encoding='utf8') as f:
    json.dump(output_eval_dataset, f, ensure_ascii=False)

