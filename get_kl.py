from transformers import pipeline, AutoTokenizer
from transformers import GenerationConfig, AutoModelForCausalLM
import time
from torch.utils.data import DataLoader

import os

from dataclasses import dataclass, field
from typing import Optional
from accelerate import Accelerator
import json
import torch
from datasets import load_dataset, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser, pipeline, DataCollatorForSeq2Seq

import numpy as np
import pandas as pd
tqdm.pandas()
from accelerate import Accelerator

#from trl.trainer.utils import generate

@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """
    model_name_or_path: Optional[str] = field(
        #default="/home/xiongwei/gshf_gold_test/LMFlow_RAFT_Dev/output_models/online_dpo/iter2/1e6/checkpoint-pymodel1587",
        default="/home/xiongwei/gshf_gold_test/LMFlow_RAFT_Dev/output_models/online_dpo/new_iter2/checkpoint-pymodel2120",
        #default="",
        metadata={"help": "the location of the SFT model name or path"},
    )
    ref_model: Optional[str] = field(
        #default="/home/xiongwei/zqw_data_test/iter3_online.json",
        #default="/home/xiongwei/over_opt/LMFlow_RAFT_Dev/output_models/12_30/raft/baseline/iter5/gen.json",
        #default="/home/xiongwei/rm_study/LMFlow/output_models/exp_no_sharegpt/open_llama_3b_v2_instruction_following_1epoch_on_hh_bz12/eval/my_eval.json",
        default="google/gemma-2b-it",
        metadata={"help": "the location of the output file"},
    )
    
    dataset_name_or_path: Optional[str] = field(
        default="/home/xiongwei/gshf_gold_test/LMFlow_RAFT_Dev/output_models/online_dpo/iter1",
        metadata={"help": "the location of the dataset name or path"},
    )
    output_dir: Optional[str] = field(
        default="",
        metadata={"help": "the location of the output file"},
    )
    record_dir: Optional[str] = field(
        default="",
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
accelerator2 = Accelerator()

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

base_dir = script_args.model_name_or_path.split("/checkpoint-pymodel")[0]

#ds_dir = script_args.model_name_or_path + "/eval_set/eval.json"
#ds_dir = script_args.model_name_or_path + "/ood_eval/ood.json"
model_name = script_args.model_name_or_path#"/home/xiongwei/LMFlow/DATA/sft_open_llama_3b_1epoch_plus_hh_rlhf_1epoch"
ref_name = script_args.ref_model#"/home/xiongwei/gshf_gen_data/LMFlow_RAFT_Dev/data/my_gen.json"
ds_dir = script_args.dataset_name_or_path

tokenizer = AutoTokenizer.from_pretrained(model_name)
#tokenizer.pad_token = tokenizer.eos_token
#tokenizer.pad_token_id = tokenizer.eos_token_id
#tokenizer.padding_side = "left"

generation_kwargs = {
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "temperature": 1.0,
    "max_new_tokens": 2048
}




ds = load_dataset("json", data_files=ds_dir, split="train", field="instances")



device = accelerator.device

model = AutoModelForCausalLM.from_pretrained(
    model_name,torch_dtype=torch.bfloat16,trust_remote_code=True
)
model = model.to(device)
model.gradient_checkpointing_disable()
model.config.use_cache = True


ref_model = AutoModelForCausalLM.from_pretrained(
    ref_name,torch_dtype=torch.bfloat16,trust_remote_code=True
)
ref_model = ref_model.to(device)
ref_model.gradient_checkpointing_disable()
ref_model.config.use_cache = True


local_rank = Accelerator().local_process_index


world_size = int(os.getenv("WORLD_SIZE", "1"))
data_size = len(ds['prompt'])
share = int(data_size / world_size) 
ds = ds.select(np.arange(local_rank * share, (local_rank + 1)*share))

optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)
optimizer2 = torch.optim.SGD(filter(lambda p: p.requires_grad, ref_model.parameters()), lr=0.0001)

data_collator = DataCollatorForSeq2Seq(tokenizer, max_length=9999, pad_to_multiple_of=1)

dataloader1 = DataLoader(ds, batch_size=16, shuffle=False, collate_fn=data_collator)
dataloader2 = DataLoader(ds, batch_size=16, shuffle=False, collate_fn=data_collator)


model, dl, opt = accelerator.prepare(model, dataloader1, optimizer)

ref_model, dl2, opt2 = accelerator2.prepare(ref_model, dataloader2, optimizer2)



def get_kl(ds, model, ref_model, tokenizer, device):
    """
    This function receives:
    querys = [###Human:how are you? ###Assistant:, ###Human: How are you? ] (K prompts)
    responses = [Fine, Good, ..., Great] (K responses)
    we compute the conditional KL for each sample and return [KL-1, ..., KL-K]
    """
    kl_seq = []
    prob_seq = []
    data = []
    len_count = 0
    #query = test_texts[i]
    with torch.no_grad():
        for sample in tqdm(ds):
        #for i in range(len(responses)):
            one_input = sample['prompt']
            responses = sample['responses']
            query_inputs = tokenizer(one_input, return_tensors="pt", padding=False).to(device)
            query_len = query_inputs['input_ids'].shape[1]
            tmp_kl_seq = []
            for i in range(len(responses)):
                one_response = responses[i]
    
                inputs = tokenizer(one_input + one_response, return_tensors="pt", padding=True).to(device)

                response_len = len(inputs["input_ids"][0]) - query_len
                len_count.append(len(inputs["input_ids"][0]))
                logits = model(**inputs)['logits']
                ref_logits = ref_model(**inputs)['logits']

                input_ids = inputs["input_ids"]
            
                logp = torch.nn.functional.log_softmax(logits[:, :-1, :], dim=2)
                logprobs = torch.gather(logp, 2, input_ids[:, 1:].unsqueeze(2)).squeeze(-1)

                ref_logp = torch.nn.functional.log_softmax(ref_logits[:, :-1, :], dim=2)
                ref_logprobs = torch.gather(ref_logp, 2, input_ids[:, 1:].unsqueeze(2)).squeeze(-1)

                
                # We compute the conditional KL only (p(y|x)) so we start from query_len - 1
                kl_pt = torch.sum(logprobs[:, query_len-1:query_len-1+response_len] - ref_logprobs[:, query_len-1:query_len-1+response_len], axis=1)
                #print(torch.sum(logprobs[:, query_len-1:] , axis=1))
                #prob_seq.append(torch.sum(ref_logprobs[:, query_len-1:], axis=1).item())
                kl_seq.append(kl_pt.item())
                tmp_kl_seq.append(kl_pt.item())
                #print(query_len, len(inputs["input_ids"][0]))
            assert len(tmp_kl_seq) == 8
            data.append({"prompt": sample['prompt'], "responses": sample['responses'], "rewards": sample['rewards'], "kl": tmp_kl_seq})
    
    print(np.mean(kl_seq), "kl")
    #print(prob_seq, "log_prob")
    return data, np.mean(kl_seq), np.mean(len_count)




my_data, kl_score, ave_len = get_kl(ds, model, ref_model, tokenizer, device)



import torch.distributed as dist
all_process_list =[{}] * world_size

data_to_send = {
    'data': my_data,
    'kl_scores': np.mean(kl_score),
    'len': ave_len
}
dist.all_gather_object(all_process_list, data_to_send)
gathered_data = []
gathered_scores = []
gathered_len = []

for i in range(world_size):
    gathered_scores += all_process_list[i]['kl_scores']
    gathered_data.extend(all_process_list[i]['data'])
    gathered_len.extend(all_process_list[i]['len'])
if local_rank == 0:
    output_eval_dataset = {}
    output_eval_dataset['type'] = 'text_only'
    output_eval_dataset['instances'] = gathered_data
    with open(script_args.output_dir + "/data_with_kls.json", 'w', encoding='utf8') as f:
        json.dump(output_eval_dataset, f, ensure_ascii=False)
    mean_kl = np.mean(gathered_scores)
    mean_len = np.mean(gathered_len)
    with open(script_args.record_dir, 'a') as f:
        f.write(str(mean_kl) + "\t" + str(mean_len) +  "\n")
