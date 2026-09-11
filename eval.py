"""
Phase 2 inference script.

Usage:
    python whisper_evaluate.py \
        --audio_dir /path/to/phase2_audio \
        --model_dir ./whisper-medium_merged \
        --output_csv submission.csv

    # If you only saved the LoRA adapter (not merged):
    python whisper_evaluate.py \
        --audio_dir /path/to/phase2_audio \
        --model_dir ./whisper-medium_adapter2 \
        --base_model openai/whisper-medium \
        --output_csv submission.csv
"""

import os
import argparse
import csv
import glob
import pandas as pd
import numpy as np
import torch
import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration

try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

SAMPLE_RATE = 16000
CHUNK_SECONDS = 30          # Whisper's native context window
CHUNK_OVERLAP_SECONDS = 1   # small overlap to avoid cutting words at boundaries
AUDIO_EXTS = (".wav", ".mp3", ".flac", ".ogg", ".m4a")


def load_model(model_dir: str, base_model: str, device: str):
    """Loads either a merged full model, or a base model + LoRA adapter,
    depending on what's present in model_dir."""
    processor = WhisperProcessor.from_pretrained(model_dir)

    adapter_config_path = os.path.join(model_dir, "adapter_config.json")
    if os.path.exists(adapter_config_path):
        if not PEFT_AVAILABLE:
            raise RuntimeError(
                "model_dir looks like a LoRA adapter but `peft` is not installed."
            )
        print(f"Loading base model '{base_model}' + LoRA adapter from '{model_dir}'")
        base = WhisperForConditionalGeneration.from_pretrained(base_model)
        model = PeftModel.from_pretrained(base, model_dir)
        model = model.merge_and_unload()  # merge for faster, simpler inference
    else:
        print(f"Loading merged model from '{model_dir}'")
        model = WhisperForConditionalGeneration.from_pretrained(model_dir)
 
    model.generation_config.forced_decoder_ids = None
    model.config.forced_decoder_ids = None
    model.generation_config.language = None
    model.generation_config.task = "transcribe"

    model.to(device)
    model.eval()
    return processor, model


def load_audio(path: str) -> np.ndarray:
    audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    return audio.astype(np.float32)


def chunk_audio(audio: np.ndarray) -> list:
    """Splits audio into <=30s chunks with slight overlap. Returns a list
    of numpy arrays. If audio is already <=30s, returns it as a single
    chunk (no wasted work)."""
    chunk_len = CHUNK_SECONDS * SAMPLE_RATE
    overlap = CHUNK_OVERLAP_SECONDS * SAMPLE_RATE

    if len(audio) <= chunk_len:
        return [audio]

    chunks = []
    start = 0
    while start < len(audio):
        end = min(start + chunk_len, len(audio))
        chunks.append(audio[start:end])
        if end == len(audio):
            break
        start = end - overlap
    return chunks


@torch.no_grad()
def transcribe_file(path: str, processor, model, device: str, fp16: bool) -> str:
    audio = load_audio(path)
    chunks = chunk_audio(audio)

    texts = []
    for chunk in chunks:
        inputs = processor.feature_extractor(
            chunk, sampling_rate=SAMPLE_RATE, return_tensors="pt"
        )
        input_features = inputs.input_features.to(device)
        if fp16 and device == "cuda":
            input_features = input_features.half()

        generated_ids = model.generate(
            input_features,
            max_new_tokens=225,
            num_beams=5,
        )
        text = processor.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0].strip()
        texts.append(text)

    return " ".join(texts).strip()


def find_audio_files(audio_dir: str) -> list:
    files = []
    for ext in AUDIO_EXTS:
        files.extend(glob.glob(os.path.join(audio_dir, f"**/*{ext}"), recursive=True))
    return sorted(files)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_dir", required=True, help="Folder of Phase 2 audio files")
    parser.add_argument("--model_dir", required=True, help="Merged model dir, or LoRA adapter dir")
    parser.add_argument("--base_model", default="openai/whisper-medium",
                         help="Only needed if --model_dir is a LoRA adapter")
    parser.add_argument("--output_csv", default="submission.csv")
    parser.add_argument("--fp16", action="store_true", default=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor, model = load_model(args.model_dir, args.base_model, device)
    if args.fp16 and device == "cuda":
        model = model.half()

    audio_files = find_audio_files(args.audio_dir)
    print(f"Found {len(audio_files)} audio files in {args.audio_dir}")

    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "Target"])  # rename columns to match the exact submission spec
        csv_phase2 = pd.read_csv('Test_phase2.csv')
        print(csv_phase2.head())
        for i, ID in enumerate(csv_phase2['ID']):
            file_id = ID 
            path = f"{args.audio_dir}/{ID}.wav"

      
            try:
                text = transcribe_file(path, processor, model, device, args.fp16)
            except Exception as e:
                print(f"WARNING: failed on {path}: {e}")
                text = ""
            writer.writerow([file_id, text])

            if (i + 1) % 50 == 0:
                print(f"Transcribed {i + 1}/{len(audio_files)} files")

    print(f"Done. Predictions written to {args.output_csv}")


if __name__ == "__main__":
    main()