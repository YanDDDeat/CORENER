import argparse
import os
import logging
from datetime import datetime
import torch
from datasets import Dataset
from peft import PeftModel
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import pipeline
from utils.test_collator import SFTDataCollator
from utils.test_proccess import SFTDataProcess


def do_evaluation(args, model, tokenizer, write_path=None, use_rag=False,valid=True):
    from tune_model import load_sft_dataset
    return infer_llama_lora_by_prompt(
        args,
        load_sft_dataset(args, tokenizer, train=False,valid=valid,ans=False),
        write_path=write_path, model=model,
        tokenizer=tokenizer,
        lora_path=f"result/{args.train_name}/model/ie-ft")



def infer_llama_lora_by_prompt(args, entity_qa_list, write_path=None, url=None, model=None, tokenizer=None,lora_path=None,rag=False):
    model.eval()
    from tune_model import load_tokenizer_model_on_lora, current_dir
    if not model or not tokenizer:
        model, tokenizer = load_tokenizer_model_on_lora(args=args, train=False, lora_path=lora_path)
    dataset = Dataset.from_dict({"data": entity_qa_list})
    dataset = dataset.map(
        lambda x: {'prompt_text': tokenizer.apply_chat_template(x["data"], tokenize=False,
                                                                add_generation_prompt=True)})
    try:
        logging.info(dataset[0])
    except Exception as e:
        pass
    # 准备批量处理的数据
    prompt_texts = [dataset[i]["prompt_text"] for i in range(len(dataset))]

    # 存储所有结果
    answer_result_word_list = []
    # llama_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer,do_sample=True)
    llama_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        do_sample=False,  # 关闭随机采样，保证稳定输出
        temperature=0,  # 设为 0，确保选择概率最高的 token
        top_k=1,  # 只选最可能的 token
        top_p=1.0  # 确保不进行 nucleus sampling
    )
    batch_size=32
    # 按批次处理数据
    for start_idx in tqdm(range(0, len(prompt_texts), batch_size)):
        end_idx = min(start_idx + batch_size, len(prompt_texts))
        batch = prompt_texts[start_idx:end_idx]
        inference_batch_results = llama_pipeline(batch, eos_token_id=tokenizer.eos_token_id, max_new_tokens=512)
        # 提取生成的文本
        answer_result_word_list.extend(
            [result["generated_text"] for inference_results in inference_batch_results for result in inference_results])
    if write_path:
        # 确保目录存在，如果不存在则创建
        os.makedirs(os.path.dirname(write_path), exist_ok=True)
        with open(write_path, "w") as file:
            for answer in answer_result_word_list:
                file.write(answer)
                file.write("\n")
    model.train()
    logging.info(f"-----------inference finished!------------")
    return answer_result_word_list

if __name__=="__main__":
    current_time = datetime.now()
    dt = current_time.strftime("%Y-%m-%d %H:%M:%S")

    from utils.log_utils import set_logger

    parser = argparse.ArgumentParser()
    group = parser.add_argument_group("train_args")
    group.add_argument("--auto_adapt", type=bool, default=True)
    group.add_argument("--batch_size", type=int, default=6)
    group.add_argument("--local_rank", type=int, default=0)
    group.add_argument("--max_steps", type=int, default=20000)
    group.add_argument("--epoch", type=int, default=20)
    group.add_argument("--use_rag", type=int, default=0)
    group.add_argument("--validate", type=int, default=0)
    group.add_argument("--max_len", type=int, default=2048)
    group.add_argument("--max_seq_length", type=int, default=2048)
    group.add_argument("--base_model_dir", type=str, default="../models/Meta-Llama-3-8B-Instruct")
    group.add_argument("--train_data_path", type=str, default="./data/BC5CDR-chem/rt_exp8/test.json")
    group.add_argument("--valid_data_path", type=str, default="./data/BC5CDR-chem/valid.json")
    group.add_argument("--test_data_path", type=str, default= "./data/BC5CDR-chem/rt_exp8/test.json")
    group.add_argument("--validate_iter", type=int, default=40000)
    group.add_argument("--write_path", type=str, default="./infer_data")
    group.add_argument("--dataset_text_field", type=str, default="formatted_chat")
    group.add_argument("--lora_output", type=str, default="ie-ft")
    group.add_argument("--train_name", type=str, default=f"{dt}")
    group.add_argument("--deepspeed", type=str, default=None)
    group.add_argument("--rag", type=str, default=False)
    args = parser.parse_args()
    # from modelling_llama import LlamaForCausalLM
    from safetensors.torch import load_file
    from transformers import AutoTokenizer,AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained("/data/yanfujian/models/Meta-Llama-3-8B-Instruct", low_cpu_mem_usage=True,
                                                          torch_dtype=torch.bfloat16, device_map="cuda:0",
                                                          load_in_8bit=True)
    from tune_model import load_tokenizer_model_on_lora
    from tune_model import load_tokenizer_model_on_lora
    merge_model = PeftModel.from_pretrained(model, "/data/yanfujian/lp/result/测试1-llama3/model/ie-ft/checkpoint-5110")
    _, tokenizer = load_tokenizer_model_on_lora(args=args, train=False, only_tokenizer=True)
    write_path = "./result/BC5CDR-chem-EXPER1--lora-2025/infer_result/lora_infer.txt"

    batch_size = 1
    model.eval()
    set_seed(42)
    from tune_model import load_sft_dataset
    entity_qa_list = SFTDataProcess(args.test_data_path, tokenizer, args.max_len, args.auto_adapt, train=False, ans=False)
    dataset = Dataset.from_dict({"data": entity_qa_list})
    dataset = dataset.map(lambda example: example['data'])
    data_collator = SFTDataCollator(tokenizer, max_length=args.max_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=data_collator)

    # 存储所有结果
    answer_result_word_list = []
    output_ids=None
    # 按批次处理数据
    # 迭代 DataLoader
    for batch in tqdm(dataloader):
        # 将数据移动到模型所在设备
        batch = {k: v.to(model.device) for k, v in batch.items()}
        batch.pop("labels", None)
        # 调用 generate 方法进行推理
        output_ids = merge_model.generate(
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

    if write_path:
        # 确保目录存在，如果不存在则创建
        os.makedirs(os.path.dirname(write_path), exist_ok=True)
        with open(write_path, "w") as file:
            for answer in answer_result_word_list:
                file.write(answer)
                file.write("\n")
    logging.info(f"-----------inference finished!------------")

    from metric import metric_by_result
    from tune_model import load_tokenizer_model_on_lora, load_sft_dataset

    precision, recall, f1 = metric_by_result(answer_result_word_list, tokenizer.eos_token, valid=False, rag=args.rag)
    logging.info(
        f"========================precision: {str(precision)}, recall: {str(recall)}, f1: {str(f1)} =======================")

