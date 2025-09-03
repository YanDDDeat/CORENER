# import json
# import logging
# from torch.utils.data import Dataset
# import json
# import logging
#
# from torch.utils.data import Dataset
#
# class SFTDataProcess(Dataset):
#     def __init__(self, file, tokenizer, max_length, auto_adapt=True,train=True,valid=False,ans=True):
#         self.tokenizer = tokenizer
#         self.max_length = max_length
#         self.ans=ans
#         self.valid=valid
#         # logger.info(f'Loading data: {file}')
#         # {
#         #     "text": "1-hydroxy-triazole (0.1 mol) was dissolved in 10 mL of NMP, and a mixture containing 50% wt sodium hydride and solid wax was added in portions, and react for 0.5 hour under stiffing. At the same time, dissolve 3-chloro-bromopropane (0.015 mol) in 5 ml of NMP, and add the solution in the mixture solution above, and allow to react under room temperature for 12 hours under stirring. The reaction solution is then poured into 50 ml of water and extracted with ethyl acetate (3×50 mL); then the organic phases are combined and washed with 30 mL of water; Anhydrous magnesium sulfate was then used to dry the organic phase, which was then filtered and evaporated to dryness; the oily substance obtained was then purified by chromatography using neutral Al2O3 or separated and purified by preparative HPLC to obtain 1-(3-chloropropoxy)benzotriazole with a yield of 75.0-85.0%.",
#         #     "chem_compounds": ["1-hydroxy-triazole", "NMP", "sodium hydride", "3-chloro-bromopropane", "NMP", "water",
#         #                        "ethyl acetate", "water", "1-(3-chloropropoxy)benzotriazole"]}
#         with open(file, 'r', encoding='utf8') as f:
#             data_list = f.readlines()
#         logging.info(data_list[0])
#         # logger.info(f"There are {len(data_list)} data in dataset")
#         self.data_list = data_list
#         self.auto_adapt = auto_adapt
#         self.train=train
#     def __len__(self):
#         return len(self.data_list)
#
#     def __getitem__(self, item):
#         # 开始自动判断并适配chat template
#         data = self.data_list[item]
#         data = json.loads(data)
#         rt_mask,input_ids, target_mask = [],[], []
#         message = data['messages']
#         rt_list = [m for m in message if m["role"] == "retrieve"][0]['content']
#         message = [m for m in message if m["role"] != "retrieve"]
#         rt_ids = self.tokenizer.encode(rt_list, add_special_tokens=False)
#         message.pop()
#         text = self.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
#         input_ids += self.tokenizer.encode(text, add_special_tokens=False)
#         target_mask += [0] * len(input_ids)
#         rt_mask += [0] * len(rt_ids)
#         assert len(input_ids) == len(target_mask)
#         # 对长度进行截断
#         rt_ids = rt_ids[:self.max_length]
#         input_ids = input_ids[:self.max_length]
#         rt_mask = [1] * len(rt_ids)
#         attention_mask = [1] * len(input_ids)
#         assert len(input_ids)  == len(attention_mask)
#         inputs = {
#             "rt_mask": rt_mask,
#             "rt_ids":rt_ids,
#             "input_ids": input_ids,
#             'labels':input_ids,
#             "attention_mask": attention_mask
#         }
#
#         return inputs
#
#
# def find_sublist_start(main_list, sub_list):
#     """
#     find_sublist_start
#
#     Args:
#     main_list (list)
#     sub_list (list)
#     """
#     sub_len = len(sub_list)
#     main_len = len(main_list)
#
#     for i in range(main_len - sub_len + 1):
#         if main_list[i:i + sub_len] == sub_list:
#             return i
#         # 因为会有开头decode不一样的情况出现
#         elif main_list[i + 1:i + sub_len] == sub_list[1:]:
#             return i
#     return -1

import json
import logging
import re

from torch.utils.data import Dataset
import json
import logging

from torch.utils.data import Dataset

class SFTDataProcess(Dataset):
    def __init__(self, file, tokenizer, max_length, auto_adapt=True,train=True,valid=False,ans=True,rag=False):
        self.tokenizer = tokenizer
        self.rag=rag
        self.max_length = max_length
        self.ans=ans
        self.valid=valid
        with open(file, 'r', encoding='utf8') as f:
            data_list = f.readlines()
        logging.info(data_list[0])
        # logger.info(f"There are {len(data_list)} data in dataset")
        self.data_list = data_list
        self.auto_adapt = auto_adapt
        self.train=train
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        # 开始自动判断并适配chat template
        data = self.data_list[item]
        data = json.loads(data)
        rt_mask,input_ids, target_mask = [],[], []
        message = data['messages']
        if self.rag:
            rt_list = [m for m in message if m["role"] == "retrieve"][0]['content']
            rt_ids = self.tokenizer.encode(rt_list, add_special_tokens=False)
            chars = list(rt_list)  # 拆字符
            char_ids = []
            for c in chars:
                char_ids.extend(self.tokenizer.encode(c, add_special_tokens=False))  # flatten
            rt_ids = char_ids  # 直接用字符级id
            message[0]['content'] = "You are a chemist and you extract compound entities from the following document and partition them with |, if any document has compound entities output: 'compoundA | compoundB. If none output instance: 'N/A'"

        message = [m for m in message if m["role"] != "retrieve"]
        message.pop()
        text = self.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        input_ids += self.tokenizer.encode(text, add_special_tokens=False)
        target_mask += [0] * len(input_ids)

        assert len(input_ids) == len(target_mask)
        # 对长度进行截断
        if self.rag:
            rt_mask += [0] * len(rt_ids)
            rt_ids = rt_ids[:self.max_length]
            input_ids = input_ids[:self.max_length]
            rt_mask = [1] * len(rt_ids)
        attention_mask = [1] * len(input_ids)
        assert len(input_ids)  == len(attention_mask)
        if self.rag:
            inputs = {
                "rt_mask": rt_mask,
                "rt_ids":rt_ids,
                "input_ids": input_ids,
                'labels': input_ids,
                "attention_mask": attention_mask
            }
            return inputs

        inputs = {
            # "rt_mask": rt_mask,
            # "rt_ids":rt_ids,
            "input_ids": input_ids,
            'labels':input_ids,
            "attention_mask": attention_mask
        }

        return inputs


def find_sublist_start(main_list, sub_list):
    """
    find_sublist_start

    Args:
    main_list (list)
    sub_list (list)
    """
    sub_len = len(sub_list)
    main_len = len(main_list)

    for i in range(main_len - sub_len + 1):
        if main_list[i:i + sub_len] == sub_list:
            return i
        # 因为会有开头decode不一样的情况出现
        elif main_list[i + 1:i + sub_len] == sub_list[1:]:
            return i
    return -1
