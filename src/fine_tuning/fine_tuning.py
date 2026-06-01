import pandas as pd
import os
import json
import torch

from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline
)

from peft import LoraConfig, get_peft_model, TaskType


class FineTuning:
    def __init__(self, data_path,
                 model_name="Qwen/Qwen2.5-3B-Instruct"):

        print("\n[INIT] Starting FineTuning pipeline...")

        self.data_path = data_path
        self.model_name = model_name

        # ---------------------------
        # LOAD DATA
        # ---------------------------
        print(f"[DATA] Loading dataset from: {data_path}")

        with open(data_path, "r", encoding="utf-8") as f:
            ft_data = json.load(f)

        print(f"[DATA] Samples loaded: {len(ft_data)}")

        self.dataset = Dataset.from_list(ft_data)

        print("[DATA] Train/test split...")
        self.dataset = self.dataset.train_test_split(test_size=0.1)

        print(f"[DATA] Train: {len(self.dataset['train'])}")
        print(f"[DATA] Test: {len(self.dataset['test'])}")

        self.dataset = self.dataset.map(self.format_example)

        # ---------------------------
        # MODEL
        # ---------------------------
        print("[MODEL] Loading base model...")

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

        print("[MODEL] Loaded successfully")

        # ---------------------------
        # TOKENIZER
        # ---------------------------
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        print("[TOKENIZER] Ready")

        # ---------------------------
        # PIPELINE (FOR INFERENCE)
        # ---------------------------
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            do_sample=False,
            return_full_text=False
        )

        self.trainer = None

    # =========================================================
    # TRAINING
    # =========================================================
    def fine_tuning(self, output_dir,
                    lora_rank=16,
                    lora_alpha=32,
                    save_pretrained=True):

        print("\n[TRAIN] Starting fine-tuning...")

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

        print("[LoRA] Configuring adapters...")

        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )

        self.model = get_peft_model(self.model, lora_config)

        print("[LoRA] Trainable parameters:")
        self.model.print_trainable_parameters()

        # ---------------------------
        # TRAINER (FIXED IMPORT NEEDED)
        # ---------------------------
        from trl import SFTTrainer

        self.trainer = SFTTrainer(
            model=self.model,
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset["test"],
            args=training_args,
            processing_class=self.tokenizer,
        )

        print("[TRAIN] Training started 🚀")
        self.trainer.train()
        print("[TRAIN] Training finished ✅")

        if save_pretrained:
            print(f"[SAVE] Saving to {output_dir}")
            self.trainer.save_model(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            print("[SAVE] Done ✅")

    # =========================================================
    # PROMPT
    # =========================================================
    def generate_prompt(self, ayah):
        return f"""
You are a strict JSON generator.

Return ONLY one valid JSON object.

RULES:
- No explanations
- No extra text
- No multiple outputs

TASK:
Extract Tajweed rules in Warsh recitation.

ALLOWED:
Ikhfa, Idgham, Iqlab, Izhar, Madd, Naql, Ghunnah

FORMAT:
{{
  "rules": [
    "rule: span"
  ]
}}

If no rule exists → return empty list.

AYAH:
{ayah}

OUTPUT:
"""

    # =========================================================
    # SAFE JSON PARSER
    # =========================================================
    def _safe_parse(self, text):
        try:
            start = text.find("{")
            end = text.rfind("}") + 1

            if start == -1 or end == -1:
                return {"rules": []}

            return json.loads(text[start:end])

        except Exception:
            return {"rules": []}

    # =========================================================
    # SINGLE PREDICTION
    # =========================================================
    def predict(self, ayah):

        output = self.pipe(
            self.generate_prompt(ayah),
            max_new_tokens=512,
            do_sample=False,
            return_full_text=False
        )[0]["generated_text"]

        return self._safe_parse(output)

    # =========================================================
    # EVALUATION
    # =========================================================
    def evaluate(self, ft_data):

        eval_data = []

        for line in ft_data:

            ayah = line["input"]["ayah"]

            raw = self.pipe(
                self.generate_prompt(ayah),
                max_new_tokens=512,
                do_sample=False,
                return_full_text=False
            )[0]["generated_text"]

            predicted = self._safe_parse(raw)

            eval_data.append({
                "ayah": ayah,
                "output": line["output"],
                "predicted": predicted
            })

        return eval_data

    # =========================================================
    # FORMAT DATASET
    # =========================================================
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