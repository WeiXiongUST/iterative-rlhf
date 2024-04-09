#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
"""
Supervised fine-tuning script for decoder language models.
"""
from datasets import load_dataset
import logging
import random
import sys
import numpy as np
import datasets
import torch
import transformers
from transformers import set_seed, AutoTokenizer
import torch.nn as nn
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollator,
    DataCollatorForLanguageModeling,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from transformers.modeling_utils import unwrap_model
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalPrediction
import inspect
from alignment import (
    DataArguments,
    H4ArgumentParser,
    ModelArguments,
    SFTConfig,
    apply_chat_template,
    get_checkpoint,
    get_datasets,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    get_tokenizer,
)
from peft import PeftConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union


from dataclasses import dataclass



from transformers import DataCollatorForLanguageModeling
logger = logging.getLogger(__name__)


@dataclass
class WeightedDataCollatorForLanguageModeling(DataCollatorForLanguageModeling):
    mlm: bool = False
    @staticmethod
    def tf_bernoulli(shape, probability):
        raise NotImplementedError
    def tf_mask_tokens(
        self, inputs: Any, vocab_size, mask_token_id, special_tokens_mask: Optional[Any] = None
    ) -> Tuple[Any, Any]:
        raise NotImplementedError

    def tf_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        raise NotImplementedError

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        # Handle dict or lists with proper padding and conversion to tensor.
        reward = [example.pop("reward") for example in examples]
        assert len(reward)==1 # only support bsz = 1 now
        #reward = examples.pop("reward")
        batch = super().torch_call(examples)
        batch["reward"] = torch.tensor(reward,dtype=torch.bfloat16)
        
        return batch


    def numpy_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        raise NotImplementedError

    def numpy_mask_tokens(self, inputs: Any, special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any]:
        raise NotImplementedError


class WeightedSFTTrainer(SFTTrainer):

    def __init__(
        self,
        model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        args: Optional[TrainingArguments] = None,
        data_collator: Optional[DataCollator] = None,  # type: ignore
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        peft_config: Optional["PeftConfig"] = None,
        dataset_text_field: Optional[str] = None,
        packing: Optional[bool] = False,
        formatting_func: Optional[Callable] = None,
        max_seq_length: Optional[int] = None,
        infinite: Optional[bool] = None,
        num_of_sequences: Optional[int] = 1024,
        chars_per_token: Optional[float] = 3.6,
        dataset_num_proc: Optional[int] = None,
        dataset_batch_size: int = 1000,
        neftune_noise_alpha: Optional[float] = None,
        model_init_kwargs: Optional[Dict] = None,
        dataset_kwargs: Optional[Dict] = None,
        eval_packing: Optional[bool] = None,
    ):
        assert packing==False
        super().__init__(
            model,
            args,
            data_collator,  # type: ignore
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            peft_config=peft_config,
            dataset_text_field=dataset_text_field,
            packing=packing,
            formatting_func=formatting_func,
            max_seq_length=max_seq_length,
            infinite=infinite,
            num_of_sequences=num_of_sequences,
            chars_per_token=chars_per_token,
            dataset_num_proc=dataset_num_proc,
            dataset_batch_size=dataset_batch_size,
            neftune_noise_alpha=neftune_noise_alpha,
            model_init_kwargs=model_init_kwargs,
            dataset_kwargs=dataset_kwargs,
            #eval_packing=eval_packing,
        )
    def _prepare_non_packed_dataloader(
        self,
        tokenizer,
        dataset,
        dataset_text_field,
        max_seq_length,
        formatting_func=None,
        add_special_tokens=True,
        remove_unused_columns=True,
    ):
        use_formatting_func = formatting_func is not None and dataset_text_field is None
        self._dataset_sanity_checked = False

        # Inspired from: https://huggingface.co/learn/nlp-course/chapter7/6?fw=pt
        def tokenize(element):
            outputs = tokenizer(
                element[dataset_text_field] if not use_formatting_func else formatting_func(element),
                add_special_tokens=add_special_tokens,
                truncation=True,
                padding=False,
                max_length=max_seq_length,
                return_overflowing_tokens=False,
                return_length=False,
            )

            if use_formatting_func and not self._dataset_sanity_checked:
                if not isinstance(formatting_func(element), list):
                    raise ValueError(
                        "The `formatting_func` should return a list of processed strings since it can lead to silent bugs."
                    )
                else:
                    self._dataset_sanity_checked = True

            return {"input_ids": outputs["input_ids"], "attention_mask": outputs["attention_mask"], "reward": element["reward"]}

        signature_columns = ["input_ids", "labels", "attention_mask", "reward"]

        extra_columns = list(set(dataset.column_names) - set(signature_columns))

        if not remove_unused_columns and len(extra_columns) > 0:
            warnings.warn(
                "You passed `remove_unused_columns=False` on a non-packed dataset. This might create some issues with the default collator and yield to errors. If you want to "
                f"inspect dataset other columns (in this case {extra_columns}), you can subclass `DataCollatorForLanguageModeling` in case you used the default collator and create your own data collator in order to inspect the unused dataset columns."
            )

        tokenized_dataset = dataset.map(
            tokenize,
            batched=True,
            remove_columns=dataset.column_names if remove_unused_columns else None,
            num_proc=self.dataset_num_proc,
            batch_size=self.dataset_batch_size,
        )
        return tokenized_dataset

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            # Inspect model forward signature to keep only the arguments it accepts.
            self._signature_columns = []
            # Labels may be named label or label_ids, the default data collator handles that.
            self._signature_columns += list(set(["label", "label_ids", "reward", 'attention_mask', 'input_ids'] + self.label_names))    
    
    def compute_loss(self, model, inputs, return_outputs=False):
        reward = inputs.pop('reward')
        if return_outputs:
            loss, outputs = super().compute_loss(model, inputs, return_outputs)
        else:
            loss = super().compute_loss(model, inputs, return_outputs)
        loss = loss * reward.mean()
        return (loss, outputs) if return_outputs else loss
 

def main():
    parser = H4ArgumentParser((ModelArguments, DataArguments, SFTConfig))
    model_args, data_args, training_args = parser.parse()

    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = get_checkpoint(training_args)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(
            f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    ###############
    # Load datasets
    ###############

    my_data_dir = data_args.dataset_splits
    print(data_args.dataset_splits, my_data_dir)
    assert "top1_data.json" in my_data_dir
    train_data = load_dataset(
        "json", data_files=my_data_dir, split="train", field="instances").shuffle()#.select(range(300))

    

    ################
    # Load tokenizer
    ################
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, use_fast=False)  # , trust_remote_code=True)
    print("I use the dataset", my_data_dir, " and the model ", model_args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id


    def tokenize_fn(sample):
        #sample['prompt'].replace("<s>", "") + sample['response'] + tokenizer.bos_token #replace(tokenizer.bos_token, "")
        # sample['text'] = sample['prompt'].replace(
        #    "<s>[INST]", "<|user|>\n").replace("[/INST]", "</s>\n<|assistant|>\n") + sample['response']
        #sample['text'] = sample['prompt'] + sample['response']
        #sample['reward'] = 
    
        prom = sample['prompt'].replace("<s>GPT4 Correct User: ", "").replace("<|end_of_turn|>GPT4 Correct Assistant:", "")

        resp = sample['response'].replace("GPT4 Correct Assistant:", "")
        final_resp = resp.split("GPT4 Correct User")[0]

        message = [
            {"role": "user", "content": prom},
            {"role": "assistant", "content": final_resp},
        ]
        sample['text'] = tokenizer.apply_chat_template(message, tokenize=False).replace(tokenizer.bos_token, "")# + tokenizer.eos_token
        #sample['reward'] = 1.0
        return sample
    train_dataset = train_data.map(tokenize_fn, batched=False)
    

    print(train_dataset[0])
    print(train_dataset[1])

    eval_dataset = train_dataset.select(
        range(100))  # raw_datasets["test"]

    #######################
    # Load pretrained model
    #######################
    logger.info("*** Load pretrained model ***")
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in [
            "auto", None] else getattr(torch, model_args.torch_dtype)
    )
    # quantization_config = get_quantization_config(model_args)
    quantization_config = None
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        use_flash_attention_2=model_args.use_flash_attention_2,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        # quantization_config=quantization_config,
    )
    ####
    instruction_template = "GPT4 Correct User:"
    response_template = "GPT4 Correct Assistant:"
    ####


    logger.info("*** Model loaded! ***")
    #print(model_args.model_name_or_path)
    ########################
    # Initialize the Trainer
    #########################CustomTrainer(CustomTrainer
    trainer =  SFTTrainer(
        model=model_args.model_name_or_path,
        model_init_kwargs=model_kwargs,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=training_args.max_seq_length,
        tokenizer=tokenizer,
        #data_collator=WeightedDataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        data_collator=DataCollatorForCompletionOnlyLM(response_template=response_template, instruction_template=instruction_template, tokenizer=tokenizer, mlm=False),
        #packing=True,
        # peft_config=get_peft_config(model_args),
    )
    print(trainer.train_dataset[0])


    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##########
    # Evaluate
    ##########
    training_args.do_eval = False
    # if training_args.do_eval:
    if False:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(eval_dataset)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "dataset": list(data_args.dataset_mixer.keys()),
        "dataset_tags": list(data_args.dataset_mixer.keys()),
        "tags": ["alignment-handbook"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    logger.info("*** Training complete ***")


if __name__ == "__main__":
    main()


