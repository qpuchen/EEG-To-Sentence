# EEG-To-Sentence
In brain-to-text applications, a pressing need has arisen to develop effective models that can accurately capture the intricate details of EEG signals, such as global and local contextual information and long-term dependencies. 

### 1. Download training data
Download from https://osf.io/q3zws/files/ and https://osf.io/2urht/files/

### 2. Train and edit hyperparameters
```
python main.py --model_name Decoder_GRU_masked_residual_attention --pretrained BART --eeg_type FFD --task_name task1_task2_taskNRv2 -l 8 --num_epoch 60 -lr 0.0000005 -b 32 -s ./checkpoints -cuda cuda:2
```

### 3. Evaluate Model
```
python eval_decoding.py --checkpoint_path ./checkpoints/8/best/task1_task2_taskNRv2_finetune_Decoder_GRU_masked_residual_attention_skipstep1_b32_20_30_5e-05_5e-07_unique_sent.pt --config_path config/8/task1_task2_taskNRv2_finetune_Decoder_GRU_masked_residual_attention_skipstep1_b32_20_30_5e-05_5e-07_unique_sent.json -cuda cuda:0
```
