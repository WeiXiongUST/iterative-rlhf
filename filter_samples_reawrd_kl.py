import torch.distributed as dist
from transformers import pipeline, AutoTokenizer
import time
from torch.utils.data import DataLoader

import os

from dataclasses import dataclass, field
from typing import Optional
from accelerate import Accelerator
import json
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser, pipeline

import torch.nn as nn
import torch
from typing import Optional, List
import numpy as np

tqdm.pandas()


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
    dataset_name_or_path: Optional[str] = field(
        default="/home/xiongwei/rsf/rsf_small_lr_vanilla_raft_restart/model0/data/data_with_rewards.json",
        metadata={"help": "the location of the dataset name or path"},
    )
    output_dir: Optional[str] = field(
        default="/home/xiongwei/rsf_gemma_2b_058/model1/data",
        metadata={"help": "the location of the output file"},
    )
    record_dir: Optional[str] = field(
        default="/home/xiongwei/rsf/rsf_small_lr_vanilla_raft_restart/reward_record.txt",
        metadata={"help": "the location of the output file"},
    )
    max_length: Optional[int] = field(
        default=9999999999,
        metadata={"help": "the maximum length of the prompt"},
    )
    reward_name_or_path: Optional[str] = field(
        default="weqweasdas/RM-Mistral-7B",
        metadata={"help": "the name of the gold reward model"},
    )
    input_output_delimiter: Optional[str] = field(
        default="",
        metadata={"help": "the delimiter between input and output"},
    )
    train_micro_batch_size_per_gpu: Optional[int] = field(
        default=4,
        metadata={"help": "the batch size for inference"},
    )


accelerator = Accelerator()

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
# AcceleratorState().deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = script_args.train_micro_batch_size_per_gpu


ds_dir = script_args.dataset_name_or_path
world_size = int(os.getenv("WORLD_SIZE", "1"))
#ds = load_dataset("json", data_files="/home/xiongwei/rsf_gemma_2b_058/model1/data/data_with_kls.json", split="train", field="instances")
ds = load_dataset("weqweasdas/rsf_pi0_iter1", split="train")['instances'][0]



data = []
top1_data = []
cnt = 0




C = 8
zzz = 0
# tqdm is used to show the progress bar
ratio = []
all_weights = []
with torch.no_grad():
    for sample in tqdm(ds):

        rewards = sample['rewards']
        #kls = sample['kl']

        nor_rewards = (np.array(rewards) - 2.35) / 2.8706514245147348
        exp_nor_rewards = np.exp(C * nor_rewards) #- np.array(kls))

        weights = exp_nor_rewards / np.sum(exp_nor_rewards)
        idx = np.argmax(nor_rewards)
        ratio.append(weights[idx])
        all_weights.append(np.mean(weights))
        for j in range(len(weights)):
            top1_data.append({'reward': weights[j] * 8, "prompt": sample['prompt'], 'response': sample['responses'][j]})
            cnt += 1
            
import matplotlib.pyplot as plt
plt.hist(ratio, bins=20)
plt.title("ratio 0.58 model1 gen, c = " + str(C))
plt.savefig("yy.png")
print(np.mean(ratio), np.mean(all_weights))
print(cnt)




output_eval_dataset = {}
output_eval_dataset['type'] = 'text_only'
output_eval_dataset['instances'] = top1_data
with open(script_args.output_dir + "/top1_data.json", 'w', encoding='utf8') as f:
    json.dump(output_eval_dataset, f, ensure_ascii=False)

