import os



def do_statistic(predict_list, label_list, half_precision=False):
    true_positive = 0
    false_positive = 0
    false_negative = 0
    for predict in predict_list:
        if predict in label_list:
            true_positive = true_positive + 1
        else:
            false_positive = false_positive + 1
    for label in label_list:
        if label not in predict_list:
            false_negative = false_negative + 1
    if half_precision:
        for predict in predict_list:
            if predict in label_list:
                continue
            for label in label_list:
                if predict in label:
                    true_positive = true_positive + 0.5
        for label in label_list:
            if label in predict_list:
                continue
            for predict in predict_list:
                if label in predict:
                    false_negative = false_negative - 0.5
    return true_positive, false_positive, false_negative


def do_metric(predict_list, label_list, half_precision=False, bos_token=None, metric_by_file=False, rag=False):
    tp, fp, fn, error_json, total_json = 0, 0, 0, 0, len(predict_list)
    i = 0
    # if rag:
    #     label_list = [e[3]["content"] for e in label_list]
    # else:
    #     label_list = [e[2]["content"] for e in label_list]
    # 一个数据有几个角色
    if len(label_list[0]) == 4:
        label_list = [e[3]["content"] for e in label_list]
    else:
        label_list = [e[2]["content"] for e in label_list]
    for predict_json, label_json in zip(predict_list, label_list):
        i += 1
        entity_tp, entity_fp, entity_fn = 0, 0, 0
        try:
            if metric_by_file is False:
                pred_assis = extract_assistant_content(predict_json, bos_token.bos_token)
                label_json = [content.strip() for content in label_json.split("|")]
                entity_tp, entity_fp, entity_fn = do_statistic(pred_assis,
                                                               label_json, half_precision)
            else:
                label_json = [content.strip() for content in label_json.split("|")]
                entity_tp, entity_fp, entity_fn = do_statistic(predict_json,
                                                               label_json, half_precision)
        except Exception as e:
            logging.info(f"========================error metric No.{i}======================")
        tp = tp + entity_tp
        fp = fp + entity_fp
        fn = fn + entity_fn
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


import re


def extract_assistant_content(text, bos_token):
    # 正则表达式提取<assistant>标签之间的内容
    # assistant_texts = re.findall(r'<s>assistant(.*?)$', text, re.DOTALL)
    # assistant_texts = re.findall(r'{}assistant(.*?){}'.format(bos_token, bos_token), text, re.DOTALL)
    assistant_texts = re.findall(rf'{re.escape(bos_token)}assistant(.*?)$', text, re.DOTALL)
    # 使用 strip() 去除换行符和空白字符
    data = [content.strip() for content in assistant_texts][0].split("|")
    return [item.strip() for item in data]


def metric_by_result(result_list, test_tokenizer, valid=True, rag=False):
    from tune_model import current_dir, load_sft_dataset
    from main import args
    data = load_sft_dataset(args, test_tokenizer, train=False, valid=valid)
    precision, recall, f1 = do_metric(result_list, data, bos_token=test_tokenizer, rag=rag)
    return precision, recall, f1

def extract_assistant_contents(text):
    blocks = text.split('<s>')
    assistant_responses = []

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        role_part, _, content_part = block.partition('\n')
        if role_part.strip().lower() == 'assistant':
            # 处理内容，去除 </s>，并拆分成单独的条目
            content = content_part.split('</s>', 1)[0].strip()
            if content:
                responses = [resp.strip() for resp in content.split('\n') if resp.strip()]
                assistant_responses.append(responses)

    return assistant_responses

def metric_by_file(args, file_path, rag=False,formatted_time=None):
    # < | begin_of_text | > system
    # Extract
    # with |.Example output: 'compoundA | compoundB'. < | eot_id | >
    # < | begin_of_text | > user
    # Torsade< | eot_id | >
    # < | begin_of_text | > assistant
    # dobutamine
    import re
    from tune_model import load_sft_dataset, load_tokenizer_model_on_lora
    def extract_all_text(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        with open(file_path, 'r', encoding='utf-8') as f:
            last_line = None
            for line in f:
                last_line = line.strip()  # 逐行读取并更新最后一行
        if False:
            extracted = extract_assistant_contents(content)
            extracted_text = [text[0].split("|") for text in extracted]
        else:
            extracted = re.findall(r'<\|begin_of_text\|>assistant(.*?)(?=<\|begin_of_text\||<\|eot_id\|>)', content,
                               re.DOTALL)
            extracted.append(last_line)
            extracted_text = [text.split("|") for text in extracted]
        cleaned_list = [[item.strip() for item in sublist] for sublist in extracted_text]
        _, test_tokenizer = load_tokenizer_model_on_lora(args=args, train=False, only_tokenizer=True)
        index = 2
        if args.rag:
            index=3
        x= [i[index]["content"] for i in load_sft_dataset(args, test_tokenizer, train=False, valid=False)]
        cleaned_list1 = [i.split("|") for i in x]
        return do_metric(cleaned_list,
                         load_sft_dataset(args, test_tokenizer, train=False, valid=False),
                         metric_by_file=True, rag=rag)
    precision, recall, f1 = extract_all_text(file_path)
    print(
        f"========================precision: {str(precision)}, recall: {str(recall)}, f1: {str(f1)} time:{str(formatted_time)}=======================")
    write_path = f"./last_dance/{args.F}/{args.data_name}/{args.train_name}/indicate.txt"
    if write_path:
        # 确保目录存在，如果不存在则创建
        os.makedirs(os.path.dirname(write_path), exist_ok=True)
        with open(write_path, "w") as file:
            file.write(str(args))
            file.write("\n")
            file.write(
                f"========================precision: {str(precision)}, recall: {str(recall)}, f1: {str(f1)} time:{str(formatted_time)}=======================")


if __name__ == '__main__':
    # pre = ["<s>assistant silica gel | 1-(1-adamantyl)-3-phenyl-propane-1,3-dione</s>"]
    import argparse
    import logging
    from datetime import datetime
    from tune_model import load_tokenizer_model_on_lora

    current_time = datetime.now()
    dt = current_time.strftime("%Y-%m-%d %H:%M:%S")
    from utils.log_utils import set_logger

    parser = argparse.ArgumentParser()
    group = parser.add_argument_group("train_args")
    group.add_argument("--auto_adapt", type=bool, default=True)
    group.add_argument("--batch_size", type=int, default=32)
    group.add_argument("--file", type=str, default="/data/yanfujian/lp/last_dance/MLM/BC5CDR-chem/rt_exp9/result.txt")
    group.add_argument("--rag", type=bool, default=False)
    group.add_argument("--local_rank", type=int, default=0)
    group.add_argument("--max_steps", type=int, default=20000)
    group.add_argument("--epoch", type=int, default=20)
    group.add_argument("--use_rag", type=int, default=0)
    group.add_argument("--validate", type=int, default=0)
    group.add_argument("--llama2", type=bool, default=False)
    group.add_argument("--max_len", type=int, default=2048)
    group.add_argument("--max_seq_length", type=int, default=2048)
    group.add_argument("--base_model_dir", type=str, default="../models/Meta-Llama-3-8B-Instruct")
    group.add_argument("--train_data_path", type=str, default="./data/BC5CDR-chem/train.json")
    group.add_argument("--valid_data_path", type=str, default="./data/BC5CDR-chem/valid.json")
    group.add_argument("--test_data_path", type=str, default= "./data/BC5CDR-chem/test.json")
    group.add_argument("--validate_iter", type=int, default=5000)
    group.add_argument("--write_path", type=str, default="./infer_data")
    group.add_argument("--dataset_text_field", type=str, default="formatted_chat")
    group.add_argument("--lora_output", type=str, default="ie-ft")
    group.add_argument("--train_name", type=str, default=f"A6000_base")
    group.add_argument("--deepspeed", type=str, default=None)
    args = parser.parse_args()
    metric_by_file(args, args.file, rag=args.rag)
