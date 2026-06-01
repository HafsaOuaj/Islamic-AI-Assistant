import pandas as pd
import os
import json
from datasets import Dataset

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)

#from trl.trainer.sft_trainer import SFTTrainer
import torch

from peft import LoraConfig, get_peft_model, TaskType


class FineTuning:
    def __init__(self, data_path,
                 model_name="Qwen/Qwen2.5-3B-Instruct"):

        print("\n[INIT] Starting FineTuning pipeline...")

        self.data_path = data_path
        self.model_name = model_name

        print(f"[DATA] Loading dataset from: {data_path}")

        with open(data_path, "r") as f:
            ft_data = json.load(f)

        print(f"[DATA] Raw samples loaded: {len(ft_data)}")

        self.dataset = Dataset.from_list(ft_data)

        print("[DATA] Splitting train/test...")
        self.dataset = self.dataset.train_test_split(test_size=0.1)

        print(f"[DATA] Train size: {len(self.dataset['train'])}")
        print(f"[DATA] Test size: {len(self.dataset['test'])}")

        print("[DATA] Formatting examples...")
        self.dataset = self.dataset.map(self.format_example)

        print("[MODEL] Loading model...")

        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=self.bnb_config,
            device_map="auto"
        )

        print("[MODEL] Model loaded successfully")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        print("[TOKENIZER] Ready (pad_token = eos_token)")

        self.trainer = None

    def fine_tuning(self, output_dir, lora_rank=16, lora_alpha=32, save_pretrained=True):

        print("\n[TRAIN] Starting fine-tuning process...")

        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            learning_rate=1e-4,
            num_train_epochs=3,
            fp16=False,
            bf16=False,
            logging_steps=10,
            save_steps=100,
            save_total_limit=2,
            report_to="none"
        )

        print("[TRAIN] Training arguments ready")

        print("[LoRA] Configuring adapters...")

        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=0.05,
            bias='none',
            task_type=TaskType.CAUSAL_LM
        )

        self.model = get_peft_model(self.model, lora_config)

        print("[LoRA] Activated successfully")
        print("[LoRA] Trainable parameters:")
        self.model.print_trainable_parameters()

        print("[TRAIN] Initializing SFTTrainer...")

        """self.trainer = SFTTrainer(
            model=self.model,
            train_dataset=self.dataset['train'],
            eval_dataset=self.dataset['test'],
            args=training_args,
            processing_class=self.tokenizer,
        )

        print("[TRAIN] Starting training... 🚀")

        self.trainer.train()"""

        print("[TRAIN] Training completed ✅")

        if save_pretrained:
            print(f"[SAVE] Saving model to: {output_dir}")
            self.trainer.save_model(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            print("[SAVE] Done ✅")

    @staticmethod
    def format_example(example):

        instruction = example["instruction"]
        input_text = example["input"]
        output = example["output"]

        prompt = f"""### Instruction:
                {instruction}

                ### Input:
                {input_text}

                ### Response:
                {output}
                """

        return {"text": prompt}
    
