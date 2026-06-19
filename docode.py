import os
command = "pip install -r req.txt -i https://pypi.tuna.tsinghua.edu.cn/simple"
result = os.system(command)
result = os.system("pip install -U bitsandbytes -i https://pypi.tuna.tsinghua.edu.cn/simple")
import os

os.environ['HTTP_PROXY'] = 'http://192.168.31.158:7890'
os.environ['HTTPS_PROXY'] = 'http://192.168.31.158:7890'
import argparse
from datetime import datetime
current_time = datetime.now()
dt = current_time.strftime("%Y-%m-%d %H:%M:%S")
parser = argparse.ArgumentParser()
group = parser.add_argument_group("train_args")
group.add_argument("--auto_adapt", type=bool, default=True)
group.add_argument("--batch_size", type=int, default=None)  # 默认值设为None，后面会根据数据集设置
group.add_argument("--local_rank", type=int, default=0)
group.add_argument("--epoch", type=int, default=20)
group.add_argument("--use_rag", type=int, default=0)
group.add_argument("--validate", type=int, default=0)
group.add_argument("--max_len", type=int, default=None)  # 默认值设为None，后面会根据数据集设置
group.add_argument("--max_seq_length", type=int, default=None)  # 默认值设为None，后面会根据数据集设置
# group.add_argument("--base_model_dir", type=str, default="../models/Qwen/Qwen3-8B")
# group.add_argument("--base_model_dir", type=str, default="../models/Meta-Llama-3-8B-Instruct")
group.add_argument("--base_model_dir", type=str, default="../models/Meta-Llama-3-8B-Instruct")
group.add_argument("--data_name", type=str, default="BC5CDR-chem")
group.add_argument("--train_data_path", type=str, default="./data/BC5CDR-chem/train.json")
group.add_argument("--valid_data_path", type=str, default="./data/BC5CDR-chem/valid.json")
group.add_argument("--test_data_path", type=str, default="./data/BC5CDR-chem/test.json")
group.add_argument("--validate_iter", type=int, default=40000)
group.add_argument("--write_path", type=str, default="./infer_data")
group.add_argument("--dataset_text_field", type=str, default="formatted_chat")
group.add_argument("--lora_output", type=str, default="ie-ft")
group.add_argument("--train_name", type=str, default=f"rt_exp9")
group.add_argument("--deepspeed", type=str, default=None)
group.add_argument("--rag", type=bool, default=True)
group.add_argument("--model_name", type=str, default="Llama3")
group.add_argument("--lora_path", type=str, default="/data/yanfujian/lp/F1/ChemRxnExtractor/rt_exp8/model/checkpoint-7710")
group.add_argument("--eval", type=bool, default=True)
group.add_argument("--F", type=str, default="MLM")

args = parser.parse_args()

# 根据数据集名称自动设置参数
if args.data_name in DATASET_CONFIGS:
    config = DATASET_CONFIGS[args.data_name]
    if args.batch_size is None:
        args.batch_size = config["batch_size"]
    if args.max_len is None:
        args.max_len = config["max_len"]
    if args.max_seq_length is None:
        args.max_seq_length = config["max_len"]
args.max_seq_length = args.max_len
args.train_data_path=f"./data/{args.data_name}/train.json"
args.valid_data_path=f"./data/{args.data_name}/valid.json"
args.test_data_path=f"./data/{args.data_name}/test.json"
if args.rag:
    args.train_data_path = f"./data/{args.data_name}/{args.train_name}/train.json"
    args.valid_data_path = f"./data/{args.data_name}/{args.train_name}/valid.json"
    args.test_data_path =  f"./data/{args.data_name}/{args.train_name}/test.json"
#  数据集 、 测评（rag）、保存名称

# set_logger(f"./result/{args.train_name}/train_logs/train-{dt}.log")
if __name__ == '__main__':

    from tune_model import train
    train(args)


