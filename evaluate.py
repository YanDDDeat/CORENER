import os
import sys
def import_module():
    current_file = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file)
    # code文件下
    return current_dir

import logging
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl

class EvaluateCallable(TrainerCallback):
    def __init__(self, trainer, validate_iter=0, test_tokenizer=None, use_rag=False):
        super().__init__()
        self.trainer = trainer
        self.validate_iter = validate_iter
        self.test_tokenizer = test_tokenizer
        self.use_rag=use_rag

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
          if self.validate_iter != 0 and state.global_step != 1 and (state.global_step % self.validate_iter == 0):
            from main import args as source_args
            logging.info(f"========================Evaluate curr_step:{state.global_step} =======================-")
            source_args.curr_step=state.global_step
            current_dir=import_module()
            write_path = os.path.join(current_dir, f"result/{source_args.train_name}/infer_valid_result/lora_infer_{state.global_step}.txt")
            from evaluate_utils import do_evaluation
            from metric import metric_by_result
            infer_result_list = do_evaluation(source_args,self.trainer.model, self.test_tokenizer,write_path=write_path, use_rag=self.use_rag,valid=True)
            precision, recall, f1= metric_by_result(infer_result_list,self.test_tokenizer,valid=True,rag= source_args.rag)
            logging.info(f"========================precision: {str(precision)}, recall: {str(recall)}, f1: {str(f1)} =======================")
