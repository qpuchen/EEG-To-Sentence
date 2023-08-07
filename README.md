
## 1. Download training data
Download from https://osf.io/q3zws/files/ and https://osf.io/2urht/files/


## 2. Preprocess datasets

Data preprocessing according to this [link](https://github.com/MikeWangWZHL/EEG-To-Text/tree/main#preprocess-datasets)


## 3. Train and edit hyperparameters
```
python main.py --model_name Decoder_GRU_masked_residual_attention --pretrained BART --eeg_type FFD --task_name task1_task2_taskNRv2 -l 8 --num_epoch 20 -bn 5 -lr 0.00005 -b 16 -s ./checkpoints -cuda cuda:0
```


## 4. Evaluate Model
```
python eval.py --model_name Decoder_GRU_masked_residual_attention --pretrained BART --eeg_type FFD --task_name task1_task2_taskNRv2 -l 8 --num_epoch 20 -bn 5 -lr 0.00005 -b 16 -s ./checkpoints -cuda cuda:0
```
