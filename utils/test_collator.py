# from typing import Any, Dict, List
# import torch
# import logging
# class SFTDataCollator:
#     def __init__(self, tokenizer, max_length):
#         self.tokenizer = tokenizer
#         self.max_length = max_length
#         self.pad_token_id = tokenizer.pad_token_id
#
#     def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
#         # 找出最大长度
#         length = [len(x['input_ids']) for x in batch if x['input_ids'] is not None]
#         # 每个batch中的最大长度
#         max_batch_length = min(self.max_length, max(length))
#         input_ids_batch, attention_mask_batch, target_mask_batch, rt_ids_batch = [], [], [], []
#         for x in batch:
#             input_ids = x['input_ids']
#             attention_mask = x['attention_mask']
#             target_mask = x['labels']
#             # =====================================================================
#             # 计算 rt_ids 的最大长度
#             rt_ids = x['rt_ids']
#             padding_rt_len = max_batch_length - len(rt_ids)
#             rt_ids = [self.pad_token_id] * padding_rt_len + rt_ids  # **左填充**
#             rt_ids = rt_ids[-self.max_length:]  # **截断**
#             rt_ids_batch.append(rt_ids)
#             # =====================================================================
#             if input_ids is None:
#                 logging.info('some input_ids is None, and now continue')
#                 continue
#             padding_len = max_batch_length - len(input_ids)
#             # **左填充**
#             input_ids = [self.pad_token_id] * padding_len + input_ids
#             attention_mask = [0] * padding_len + attention_mask
#             target_mask = [0] * padding_len + target_mask
#
#             # **截断**
#             input_ids = input_ids[-self.max_length:]
#             attention_mask = attention_mask[-self.max_length:]
#             target_mask = target_mask[-self.max_length:]
#
#             # 加入 batch
#             input_ids_batch.append(input_ids)
#             attention_mask_batch.append(attention_mask)
#             target_mask_batch.append(target_mask)
#
#         # **转换为 PyTorch Tensor**
#         input_ids_batch = torch.tensor(input_ids_batch, dtype=torch.long)
#         attention_mask_batch = torch.tensor(attention_mask_batch, dtype=torch.long)
#         target_mask_batch = torch.tensor(target_mask_batch, dtype=torch.long)
#         rt_ids_batch = torch.tensor(rt_ids_batch, dtype=torch.long)
#
#         # **计算 rt_mask**
#         rt_mask = (rt_ids_batch != self.pad_token_id).to(torch.long)
#
#         # **计算 labels**
#         labels = torch.where(target_mask_batch == 1, input_ids_batch, -100)
#
#         inputs = {
#             "rt_mask": rt_mask,
#             "rt_ids": rt_ids_batch,
#             "input_ids": input_ids_batch,
#             "attention_mask": attention_mask_batch,
#             "labels": labels
#         }
#         return inputs

from typing import Any, Dict, List
import torch
import logging
class SFTDataCollator:
    def __init__(self, tokenizer, max_length,rag=False):
        self.tokenizer = tokenizer
        self.rag=rag
        self.max_length = max_length
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        # 找出最大长度
        length = [len(x['input_ids']) for x in batch if x['input_ids'] is not None]
        # 每个batch中的最大长度
        max_batch_length = min(self.max_length, max(length))
        input_ids_batch, attention_mask_batch, target_mask_batch, rt_ids_batch = [], [], [], []
        for x in batch:
            input_ids = x['input_ids']
            attention_mask = x['attention_mask']
            target_mask = x['labels']
            # =====================================================================
            # 计算 rt_ids 的最大长度
            if self.rag:
                rt_ids = x['rt_ids']
                padding_rt_len = max_batch_length - len(rt_ids)
                rt_ids = [self.pad_token_id] * padding_rt_len + rt_ids  # **左填充**
                rt_ids = rt_ids[self.max_length:]  # **截断**
                rt_ids_batch.append(rt_ids)
            # =====================================================================
            if input_ids is None:
                logging.info('some input_ids is None, and now continue')
                continue
            padding_len = max_batch_length - len(input_ids)
            # **左填充**
            input_ids = [self.pad_token_id] * padding_len + input_ids
            attention_mask = [0] * padding_len + attention_mask
            target_mask = [0] * padding_len + target_mask

            # **截断**
            input_ids = input_ids[-self.max_length:]
            attention_mask = attention_mask[-self.max_length:]
            target_mask = target_mask[-self.max_length:]

            # 加入 batch
            input_ids_batch.append(input_ids)
            attention_mask_batch.append(attention_mask)
            target_mask_batch.append(target_mask)

        # **转换为 PyTorch Tensor**
        input_ids_batch = torch.tensor(input_ids_batch, dtype=torch.long)
        attention_mask_batch = torch.tensor(attention_mask_batch, dtype=torch.long)
        target_mask_batch = torch.tensor(target_mask_batch, dtype=torch.long)


        # **计算 labels**
        labels = torch.where(target_mask_batch == 1, input_ids_batch, -100)
        if self.rag:
            rt_ids_batch = torch.tensor(rt_ids_batch, dtype=torch.long)
            rt_mask = (rt_ids_batch != self.pad_token_id).to(torch.long)
            inputs = {
                "rt_mask": rt_mask,
                "rt_ids": rt_ids_batch,
                "input_ids": input_ids_batch,
                "attention_mask": attention_mask_batch,
                "labels": labels
            }
            return inputs


        inputs = {
            # "rt_mask": rt_mask,
            # "rt_ids": rt_ids_batch,
            "input_ids": input_ids_batch,
            "attention_mask": attention_mask_batch,
            "labels": labels
        }
        return inputs
