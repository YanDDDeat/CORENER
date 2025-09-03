import gc
import os
import sys
import time
from torch.utils.data import DataLoader
from tqdm import tqdm


def import_module():
    current_file = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file)
    utils_dir = os.path.join(current_dir, 'utils')
    sys.path.append(utils_dir)
    data_dir = os.path.join(current_dir, 'data')
    tuned = os.path.join(current_dir, 'tuned')
    sys.path.append(data_dir)
    sys.path.append(tuned)
    # code文件下
    return current_dir

import torch
import torch.nn as nn
import torch.nn.functional as F

class MLMCharaterLearning(nn.Module):
    def __init__(self, hidden_size, vocab_size, mask_token_id, mask_prob=0.15):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.mask_prob = mask_prob
        self.mask_token_id = mask_token_id  # 必须设置
        self.mlm_head = nn.Linear(hidden_size, vocab_size)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    @torch.no_grad()
    def _build_random_mask(self, shape_2d, device, rt_mask=None):
        rand_mask = torch.rand(shape_2d, device=device) < self.mask_prob
        if rt_mask is not None:
            rand_mask = rand_mask & rt_mask.bool()
        if rand_mask.sum() == 0:
            rand_mask[0, torch.randint(0, shape_2d[1], (1,), device=device)] = True
        return rand_mask  # [B, T] bool

    def forward(self, input_ids, embed_tokens, rt_mask=None, return_loss=False):
        B, T = input_ids.shape
        device = input_ids.device

        if self.training:
            random_mask = self._build_random_mask((B, T), device, rt_mask=rt_mask)
            masked_ids = input_ids.clone()
            masked_ids[random_mask] = self.mask_token_id
            mlm_labels = input_ids.clone()
            mlm_labels[~random_mask] = -100
        else:
            masked_ids = input_ids
            mlm_labels = None
        masked_embeds = embed_tokens(masked_ids)
        mlm_logits = self.mlm_head(masked_embeds)
        with torch.no_grad() if not self.training else torch.enable_grad():
            mlm_feats = F.softmax(mlm_logits, dim=-1) @ self.mlm_head.weight
        loss = None
        if self.training and return_loss and (mlm_labels is not None):
            loss = self.loss_fn(
                mlm_logits.view(-1, self.vocab_size),
                mlm_labels.view(-1)
            )

        return mlm_feats, loss


current_dir = import_module()
import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoTokenizer, TrainingArguments, Trainer, pipeline,AutoModelForCausalLM
from transformers import LlamaForCausalLM
from modelling_llama import LlamaForCausalLM
import logging

"""
    加载model和dataset
"""

def prepare_base_model_dataset(args, train=False, tokenizer=None):
    if not tokenizer:
        _, tokenizer = load_tokenizer_model_on_lora(args=args, only_tokenizer=True, train=train)
    dataset = load_sft_dataset(args, tokenizer, train=train)
    x = dataset[0]
    logging.info("========================load llm to gpu!===============================")
    base_model = LlamaForCausalLM.from_pretrained(pretrained_model_name_or_path=args.base_model_dir,
                                                  low_cpu_mem_usage=True, torch_dtype=torch.bfloat16,
                                                  device_map="auto", load_in_8bit=True, trust_remote_code=True)


    base_model.new_layer = MLMCharaterLearning(base_model.config.hidden_size, base_model.config.vocab_size, rank=8)
    base_model.new_layer = base_model.new_layer.to(torch.bfloat16)
    base_model.new_layer.to(next(base_model.parameters()).device)
    temp_model = LlamaForCausalLM.from_pretrained(pretrained_model_name_or_path=args.base_model_dir,
                                                  low_cpu_mem_usage=True, torch_dtype=torch.bfloat16,
                                                  device_map="auto", load_in_8bit=True, trust_remote_code=True)
    pretrained_weights = temp_model.state_dict()
    missing_keys, unexpected_keys = base_model.load_state_dict(pretrained_weights, strict=False)
    del temp_model
    gc.collect()
    torch.cuda.empty_cache()
    logging.info("========================load llm to gpu success!===========================")
    base_model.config.use_cache = False
    base_model.config.pretraining_tp = 1
    return base_model, tokenizer, dataset

def prepare_base_model_dataset_origin(args, train=False, tokenizer=None):
    if not tokenizer:
        _, tokenizer = load_tokenizer_model_on_lora(args=args, only_tokenizer=True, train=train)
    dataset = load_sft_dataset(args, tokenizer, train=train)
    x = dataset[0]
    logging.info("========================load llm to gpu!===============================")
    base_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=args.base_model_dir,
                                                  low_cpu_mem_usage=True, torch_dtype=torch.bfloat16,
                                                 load_in_8bit=True,trust_remote_code=True)
    return base_model, tokenizer, dataset

def load_sft_dataset(args, tokenizer=None, train=True, valid=False, ans=True):
    from utils.data_process import SFTDataProcess
    if train:
        dataset = SFTDataProcess(args.train_data_path, tokenizer, args.max_len, args.auto_adapt,rag=args.rag)
    elif valid is False:
        dataset = SFTDataProcess(args.test_data_path, tokenizer, args.max_len, args.auto_adapt, train=False, ans=ans,rag=args.rag)
    else:
        dataset = SFTDataProcess(args.valid_data_path, tokenizer, args.max_len, args.auto_adapt, train=False,
                                 valid=True, ans=ans,rag=args.rag)
    return dataset


"""
    加载tokenizer/model(lora)
"""

#

import numpy as np

def load_tokenizer_model_on_lora(args, train=False, only_tokenizer=False, only_model=False, lora_path=None,
                                 lora_path_simple=None, can_unmerge=False, base_model=None):
    merge_model = None
    tokenizer = None
    if not only_tokenizer:
        if not base_model:
            base_model = LlamaForCausalLM.from_pretrained(args.base_model_dir, low_cpu_mem_usage=True,
                                                          torch_dtype=torch.bfloat16, device_map="auto",
                                                          load_in_8bit=True)
        if lora_path:
            merge_model = PeftModel.from_pretrained(base_model, lora_path)
            # 合并原模型和lora微调参数
            if can_unmerge:
                merge_model.merge_adapter()
            else:
                merge_model.merge_and_unload()
        else:
            merge_model = base_model
    if not only_model:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model_dir, trust_remote_code=True)
        # QWenTokenizer比较特殊，pad_token_id、bos_token_id、eos_token_id均为None。eod_id对应的token为<|endoftext|>
        # if tokenizer.__class__.__name__ == 'QWenTokenizer':
        #     tokenizer.pad_token_id = tokenizer.eod_id
        #     tokenizer.bos_token_id = tokenizer.eod_id
        #     tokenizer.eos_token_id = tokenizer.eod_id
        # if tokenizer.bos_token is None:  # qwen没有bos_token，要设置一下，不然dpo train时会报错。
        #     tokenizer.add_special_tokens({"bos_token": tokenizer.eos_token})
        #     tokenizer.bos_token_id = tokenizer.eos_token_id
        # assert tokenizer.pad_token_id is not None, "pad_token_id should not be None"
        # assert tokenizer.eos_token_id is not None, "eos_token_id should not be None"
        # llama2
        # chat_template = "{% for message in messages %}{{bos_token + message['role'] + '\n ' + message['content'] + ' ' +eos_token + '\n'}}{% endfor %}{% if add_generation_prompt %}{{bos_token + 'assistant\n' }}{% endif %}"

        chat_template = "{% for message in messages %}{{bos_token + message['role'] + '\n' + message['content'] + eos_token + '\n'}}{% endfor %}{% if add_generation_prompt %}{{bos_token + 'assistant\n' }}{% endif %}"
        tokenizer.chat_template = chat_template
        tokenizer.pad_token_id = tokenizer.eos_token_id  # 设置为 eos_token_id
        tokenizer.padding_side = "right" if train else "left"
        special_tokens = ["<|reserved_special_token_0|>"]
        tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
        tokenizer.add_generation_prompt = not train
    return merge_model, tokenizer



class MLMLossCallback(TrainerCallback):
    def __init__(self, mlm_optimizer: torch.optim.Optimizer, grad_accum_steps: int = 1):
        super().__init__()
        self.mlm_optimizer = mlm_optimizer
        self.grad_accum_steps = max(1, grad_accum_steps)
        self._did_mlm_backward = False

    def on_after_backward(self, args, state, control, **kwargs):
        outputs = kwargs.get("outputs", None)
        if outputs is None:
            return
        mlm_loss = getattr(outputs, "mlm_loss", None)
        if mlm_loss is None and isinstance(outputs, dict):
            mlm_loss = outputs.get("mlm_loss", None)
        if mlm_loss is None or (not torch.is_tensor(mlm_loss)):
            return
        if self._did_mlm_backward:
            return

        self.mlm_optimizer.zero_grad()
        mlm_loss = mlm_loss.float()
        mlm_loss.backward()
        self._did_mlm_backward = True

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.grad_accum_steps != 0:
            return
        if not self._did_mlm_backward:
            return
        # balaalabalabalablabl

class TrainerExcludeMLM(Trainer):
    def __init__(self, *args, mlm_optimizer=None, mlm_scheduler=None,**kwargs):
        super().__init__(*args, **kwargs)
        self.mlm_scheduler=mlm_scheduler
        self.mlm_optimizer = mlm_optimizer
    def create_optimizer(self):
        if getattr(self, "optimizer", None) is None:
            # collect param ids to exclude: any parameter whose name contains 'new_layer'
            mlm_param_ids = {id(p) for n, p in self.model.named_parameters() if "new_layer" in n}
            params = [p for n, p in self.model.named_parameters() if p.requires_grad and id(p) not in mlm_param_ids]
            self.optimizer = torch.optim.AdamW(params, lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        return self.optimizer

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if (self.label_smoother is not None or self.compute_loss_func is not None) and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        if self.model_accepts_loss_kwargs:
            loss_kwargs = {}
            if num_items_in_batch is not None:
                loss_kwargs["num_items_in_batch"] = num_items_in_batch
            inputs = {**inputs, **loss_kwargs}
        outputs = model(**inputs)
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            unwrapped_model = self.accelerator.unwrap_model(model)
            if _is_peft_model(unwrapped_model):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            # User-defined compute_loss function
            if self.compute_loss_func is not None:
                loss = self.compute_loss_func(outputs, labels, num_items_in_batch=num_items_in_batch)
            elif model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "error"
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        if self.args.average_tokens_across_devices and self.model_accepts_loss_kwargs:
            loss *= self.accelerator.num_processes
        #  单独 backward
        mlm_loss = outputs["mlm_loss"]
        if mlm_loss is not None and self.mlm_optimizer is not None:
            self.mlm_optimizer.zero_grad()
            mlm_loss.backward()
            self.mlm_optimizer.step()
            self.mlm_scheduler.step()
        return (loss, outputs) if return_outputs else loss

def train(args, tokenizer=None):
    from utils.data_collator import SFTDataCollator
    if args.rag:
        base_model, train_tokenizer, dataset = prepare_base_model_dataset(args=args, train=True,
                                                                          tokenizer=tokenizer)
    else:
        base_model, train_tokenizer, dataset = prepare_base_model_dataset_origin(args=args, train=True,
                                                                          tokenizer=tokenizer)
    _, test_tokenizer = load_tokenizer_model_on_lora(args=args, train=False, only_tokenizer=True)
    peft_param = LoraConfig(
        lora_alpha=16,  # Lora超参数，用于缩放低秩适应的权重
        lora_dropout=0.1,  # Lora层的丢弃率
        r=64,  # Lora中的秩
        bias="none",
        task_type="CAUSAL_LM",  # Llama属于因果语言模型,
        modules_to_save = ["new_layer"],
        # target_modules = ["q_proj", "v_proj"],
        # target_modules=["query_key_value"],
    )

    for name, param in base_model.new_layer.named_parameters():
        param.requires_grad = True

    model = get_peft_model(base_model, peft_param)
    model.enable_input_require_grads()  # 开启梯度检查点时，要执行该方法
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=f"./last_dance/{args.F}/{args.data_name}/{args.train_name}/model/",
        per_device_train_batch_size=args.batch_size,
        logging_steps=10,
        num_train_epochs=args.epoch,
        save_steps=args.validate_iter,
        # save_strategy="epoch",
        gradient_checkpointing=True,
        learning_rate=2e-4,
        weight_decay=0.001,
        group_by_length=True,
        lr_scheduler_type="constant",
        max_grad_norm=0.3,
        warmup_ratio=0.03,
    )

    mlm_optimizer = torch.optim.AdamW(base_model.new_layer.parameters(), lr=5e-5)
    # callback 废弃
    callback = MLMLossCallback(mlm_optimizer, grad_accum_steps=training_args.gradient_accumulation_steps)

    trainer = TrainerExcludeMLM(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=train_tokenizer,
        data_collator=SFTDataCollator(tokenizer=train_tokenizer, max_length=args.max_len,rag=args.rag)
    )

    if args.validate == 1:
        from evaluate import EvaluateCallable
        trainer.add_callback(
            EvaluateCallable(trainer=trainer, test_tokenizer=test_tokenizer, validate_iter=args.validate_iter,
                             use_rag=(args.use_rag == 1)))
    logging.info("========================train start!!===========================")
    start_time = time.time()
    trainer.train()
    end_time = time.time()
    elapsed_time = end_time - start_time
    formatted_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
    logging.info(f"========================train end Training time: {formatted_time}!!===========================")
    logging.info("========================test start!!===========================")
    # infer_result_list = do_evaluation(args=args, model=model, tokenizer=test_tokenizer,
    #                                   write_path=f"result/{args.train_name}/infer_test_result/lora_infer.txt",
    #                                   valid=False)
    # precision, recall, f1 = metric_by_result(infer_result_list, test_tokenizer, valid=False, rag=args.rag)
    do(args,model,test_tokenizer,formatted_time)
    # trainer.save_model()
    # merged_model = model.merge_and_unload()
    model.save_pretrained(f"./last_dance/{args.F}/{args.data_name}/{args.train_name}/final_model")
    logging.info(f"========================test Finished!!=======================")

def do(args,model,tokenizer,formatted_time):
    batch_size = 1
    model.eval()
    from utils.test_proccess import SFTDataProcess
    from utils.test_collator import SFTDataCollator
    entity_qa_list = SFTDataProcess(args.test_data_path, tokenizer, args.max_len, args.auto_adapt, train=False,
                                    ans=False,rag=args.rag)
    dataset = Dataset.from_dict({"data": entity_qa_list})
    dataset = dataset.map(lambda example: example['data'])
    data_collator = SFTDataCollator(tokenizer, max_length=args.max_len,rag=args.rag)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=data_collator)

    # 存储所有结果
    answer_result_word_list = []
    output_ids = None
    # 按批次处理数据
    # 迭代 DataLoader
    for batch in tqdm(dataloader):
        # 将数据移动到模型所在设备
        batch = {k: v.to(model.device) for k, v in batch.items()}
        batch.pop("labels", None)
        # 调用 generate 方法进行推理
        if args.rag:
            output_ids = model.generate(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                rt_ids=batch['rt_ids'],  # 额外参数，模型需要处理
                rt_mask=batch['rt_mask'],  # 额外参数，模型需要处理
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=512,
                do_sample=False,  # 关闭随机采样，保证稳定输出
                temperature=0,  # 设为 0，确保选择概率最高的 token
                top_k=1,  # 只选最可能的 token
                top_p=1.0  # 确保不进行 nucleus sampling
            )
        else:
            output_ids = model.generate(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                # rt_ids=batch['rt_ids'],  # 额外参数，模型需要处理
                # rt_mask=batch['rt_mask'],  # 额外参数，模型需要处理
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=512,
                do_sample=False,  # 关闭随机采样，保证稳定输出
                temperature=0,  # 设为 0，确保选择概率最高的 token
                top_k=1,  # 只选最可能的 token
                top_p=1.0  # 确保不进行 nucleus sampling
            )

        # 对每个生成结果进行解码
        for sequence in output_ids:
            generated_text = tokenizer.decode(sequence, skip_special_tokens=False)
            # print(generated_text)
            answer_result_word_list.append(generated_text)
        del output_ids  # 删除生成的张量
        torch.cuda.empty_cache()
    write_path = f"./last_dance/{args.F}/{args.data_name}/{args.train_name}/result.txt"

    if write_path:
        # 确保目录存在，如果不存在则创建
        os.makedirs(os.path.dirname(write_path), exist_ok=True)
        with open(write_path, "w") as file:
            for answer in answer_result_word_list:
                file.write(answer)
                file.write("\n")
    logging.info(f"-----------inference finished!------------")

    from metric import metric_by_result,metric_by_file
    metric_by_file(args,write_path, rag=args.rag,formatted_time=formatted_time)

