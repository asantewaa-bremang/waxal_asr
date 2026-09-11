"""
Whisper LoRA fine-tuning script for African multilingual ASR
(Shona / Lingala / Luganda), converted from whisper.ipynb.
"""

import os
import torch
import dataclasses
import datasets
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import numpy as np

from datasets import load_dataset, interleave_datasets, Audio, Dataset
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from audiomentations import Compose, AddBackgroundNoise, TimeStretch, PitchShift, Gain

# ==========================================
# 1. SETUP & PROCESSOR
# ==========================================
model_id = "openai/whisper-medium"
processor = WhisperProcessor.from_pretrained(model_id)

## function for data preprocessing 
augment_pipeline = Compose([
    PitchShift(min_semitones=-2, max_semitones=2, p=0.3),
    Gain(min_gain_db=-6, max_gain_db=6, p=0.4),
])

def prepare_dataset_with_aug(batch):

    audio_data = batch["audio"]["array"]
    sample_rate = batch["audio"]["sampling_rate"]
    
    # Apply raw audio augmentation on the fly
    augmented_audio = augment_pipeline(samples=audio_data.astype(np.float32), sample_rate=sample_rate)
    
    # Process into 80-channel log-Mel Spectrogram frames
    batch["input_features"] = processor.feature_extractor(
        augmented_audio, 
        sampling_rate=sample_rate
    ).input_features[0]
    
    # Tokenize transcripts to extract labels
    batch["labels"] = processor.tokenizer(batch["transcription"]).input_ids
    return batch

def is_longer_than_2s(example):
    audio = example["audio"]
    duration = len(audio["array"]) / audio["sampling_rate"]
    return duration >= 2.0



# ==========================================
# 2. LOAD DATASET & TEMPERATURE RESAMPLE
# ==========================================
SAVED_DATASET_PATH = "waxal_train_val"
dataset_dict = datasets.load_from_disk(SAVED_DATASET_PATH)

train_dataset = dataset_dict["train"]
val_dataset = dataset_dict["validation"]

train_dataset = train_dataset.cast_column("audio", Audio(sampling_rate=16000))
val_dataset = val_dataset.cast_column("audio", Audio(sampling_rate=16000))
train_dataset = train_dataset.filter(is_longer_than_2s)


lin_data, lug_data, sna_data = [], [], []

for i in train_dataset:
    if i["language"] == "lin":
        lin_data.append(i)
    elif i["language"] == "lug":
        lug_data.append(i)
    elif i["language"] == "sna":
        sna_data.append(i)

# Convert the lists back into Hugging Face Dataset objects
lin_dataset = Dataset.from_list(lin_data)
lug_dataset = Dataset.from_list(lug_data)
sna_dataset = Dataset.from_list(sna_data)

# Get sizes for temperature-scaled sampling
sizes = [len(lin_dataset), len(lug_dataset), len(sna_dataset)]
total_size = sum(sizes)

p = [size / total_size for size in sizes]
temperature = 1.3
upsampled_p = [prob ** (1 / temperature) for prob in p]
sum_upsampled_p = sum(upsampled_p)
probabilities = [u / sum_upsampled_p for u in upsampled_p]

# Interleave the individual language datasets using the computed probabilities
train_dataset = interleave_datasets(
    [lin_dataset, lug_dataset, sna_dataset],
    probabilities=probabilities,
    seed=42,
    stopping_strategy="all_exhausted",
)

# Crucial next step: Always remember to shuffle right after!
train_dataset = train_dataset.shuffle(seed=42)
train_dataset = train_dataset.cast_column("audio", Audio(sampling_rate=16000))
val_dataset = val_dataset.cast_column("audio", Audio(sampling_rate=16000))



# Execute mapping across the CPU cores
train_dataset = train_dataset.map(
    prepare_dataset_with_aug,

)


# ==========================================
# 3. DYNAMIC MULTILINGUAL PREPROCESSING
# ==========================================
def preprocess_function(batch):
    # Process audio

    audio = batch["audio"]
    batch["input_features"] = processor.feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features[0]

    # Dynamic Language Handling via Prompt Forcing
    lang = batch["language"].lower()  # Ensure your dataset has a "language" key

    if "shona" in lang or lang == "sna":
        processor.tokenizer.set_prefix_tokens(language="shona", task="transcribe")
    elif "lingala" in lang or lang == "lin":
        processor.tokenizer.set_prefix_tokens(language="lingala", task="transcribe")
    elif "luganda" in lang or lang == "lug":
        # Luganda proxy mapping to Swahili token
        processor.tokenizer.set_prefix_tokens(language="swahili", task="transcribe")

    # Encode target text tokens
    text = batch["transcription"].strip()
    batch["labels"] = processor.tokenizer(text).input_ids
    return batch




# Apply preprocessing (adjust num_proc based on your CPU cores)
train_dataset = train_dataset.map(
    preprocess_function)#, remove_columns=train_dataset.column_names )  # , num_proc=4)
val_dataset = val_dataset.map(
    preprocess_function, remove_columns=val_dataset.column_names
)  
# ==========================================
# 4. MULTILINGUAL DATA COLLATOR
# ==========================================
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # Split input features and labels
        input_features = [
            {"input_features": feature["input_features"]} for feature in features
        ]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(
            label_features, return_tensors="pt"
        )

        # Replace padding token id with -100 to correctly ignore loss calculation
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch["attention_mask"].ne(1), -100
        )

        # If the first token is a bos token, remove it as it gets appended automatically
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

# Cell 7: Model Optimization Configurations
from transformers import WhisperConfig, WhisperForConditionalGeneration
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# 1. Instantiate Core Config and Enable SpecAugment (Micro-Acoustic Protection)
config = WhisperConfig.from_pretrained(model_id)
config.apply_spec_augment = True         
config.freq_mask_param = 15 # Frequency masking width              
config.time_mask_param = 40 # Time masking width

# ==========================================
# 5. LOAD QUANTIZED MODEL & APPLY LORA
# ==========================================
# Load model in 8-bit to save massive VRAM footprints on Whisper-Large
quantization_config = BitsAndBytesConfig(load_in_8bit=True)

model = WhisperForConditionalGeneration.from_pretrained(
    model_id,
    config=config,
    quantization_config=quantization_config,
    device_map="auto",
)

model.gradient_checkpointing_enable()
# Prepare model for low precision k-bit training
model = prepare_model_for_kbit_training(model)

# Define LoRA Configuration targeting attention projection matrices
peft_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["q_proj",  "v_proj","fc1"],
    lora_dropout=0.05,
    bias="none",
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()



# evaluation

import evaluate
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer

# ==========================================
# 5.5 DEFINE EVALUATION METRIC (WER)
# ==========================================
# Load the Word Error Rate (WER) metric
metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")
def compute_metrics(pred):
    """Decodes token IDs into strings and calculates the Word Error Rate."""
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # Replace padding tokens (-100) with the tokenizer's pad token id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    # Calculate WER (Word Error Rate)
    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    cer = 100 * cer_metric.compute(predictions=pred_str, references=label_str)
    metric - 100 -((cer+wer)/2)
    return {"output": metric}


# ==========================================
# 6. TRAINING CONFIGURATION
# ==========================================
training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-medium2",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,  # Effective batch size = 32
    learning_rate=1e-4,
    num_train_epochs=5,
    warmup_steps=500,
    gradient_checkpointing=True,
    fp16=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    # Validation settings
    eval_strategy="steps",  # Run validation every X steps
    eval_steps=1000,  # Matches save_steps
    per_device_eval_batch_size=4,
    save_steps=1000,
    logging_steps=500,

    # VITAL FOR GENERATIVE METRICS (WER)
    predict_with_generate=True,        # Tells the trainer to actually generate text tokens
    generation_max_length=225,     
    
    report_to=["tensorboard"],
    remove_unused_columns=False,  # Vital for PEFT datasets
    label_names=["labels"],  # Vital for PEFT datasets
)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
     compute_metrics=compute_metrics
)


def main():
    # Clear cache and run training
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    trainer.train()

    # Save only the lightweight LoRA adapters
    model.save_pretrained("./whisper-medium_adapter2")


if __name__ == "__main__":
    main()