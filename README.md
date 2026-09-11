waxal_asr

waxal_asr is a collection of python scripts  used to train an evaluate the waxal dataset for the waxal hackathon on zindi.

### Step 1: Run train.py 

This step installs dependencies, configures Fairseq, and prepares the environment.

`python train.py`

### Step 2: Run the evaluation dataset 

```
python eval.py \
        --audio_dir /path/to/phase2_audio \
        --model_dir ./whisper-medium_adapter2 \
        --base_model openai/whisper-medium \
        --output_csv submission.csv
```

