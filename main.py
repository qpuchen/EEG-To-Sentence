import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import pickle
import json
import time
import copy
from tqdm import tqdm
from transformers import BartTokenizer, BartForConditionalGeneration, AutoModelForSeq2SeqLM, AutoTokenizer

from data import ZuCo_dataset
from model import Decoder_Naive, Decoder_Transformer, Decoder_LSTM_masked_residual_attention, \
    Decoder_GRU_masked_residual_attention
from config import get_config


def train_model(dataloaders, device, model, criterion, optimizer, scheduler, num_epochs=25, fold_index=0,
                checkpoint_path_best='./checkpoints/best/temp_decoding.pt',
                checkpoint_path_last='./checkpoints/last/temp_decoding.pt'):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 100000000000

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'dev']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0

            # Iterate over data.
            for input_embeddings, seq_len, input_masks, input_mask_invert, target_ids, target_mask, sentiment_labels, sent_level_EEG in tqdm(
                    dataloaders[phase]):

                # load in batch
                input_embeddings_batch = input_embeddings.to(device).float()
                input_masks_batch = input_masks.to(device)
                input_mask_invert_batch = input_mask_invert.to(device)
                target_ids_batch = target_ids.to(device)
                """replace padding ids in target_ids with -100"""
                target_ids_batch[target_ids_batch == tokenizer.pad_token_id] = -100

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                seq2seqLMoutput = model(input_embeddings_batch, input_masks_batch, input_mask_invert_batch,
                                        target_ids_batch)

                """calculate loss"""
                # logits = seq2seqLMoutput.logits # 8*48*50265
                # logits = logits.permute(0,2,1) # 8*50265*48

                # loss = criterion(logits, target_ids_batch_label) # calculate cross entropy loss only on encoded target parts
                # NOTE: my criterion not used
                loss = seq2seqLMoutput.loss  # use the BART language modeling loss

                # backward + optimize only if in training phase
                if phase == 'train':
                    # with torch.autograd.detect_anomaly():
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.item() * input_embeddings_batch.size()[0]  # batch loss

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]

            print('{} Loss: {:.4f}'.format(phase, epoch_loss))

            # deep copy the model
            if phase == 'dev' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                '''save checkpoint'''
                torch.save(model.state_dict(), checkpoint_path_best)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))
    torch.save(model.state_dict(), checkpoint_path_last)
    print(f'update last checkpoint: {checkpoint_path_last}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def show_require_grad_layers(model):
    print()
    print(' require_grad layers:')
    # sanity check
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(' ', name)


if __name__ == '__main__':
    args = get_config('train_decoding')

    ''' config param'''
    dataset_setting = 'unique_sent'

    num_epochs = args['num_epoch']

    bn = args['block_number']
    lr = args['learning_rate']
    dr = args['dropout']

    batch_size = args['batch_size']

    model_name = args['model_name']
    pretrained_name = args['pretrained']

    # task_name = 'task1'
    # task_name = 'task1_task2'
    # task_name = 'task1_task2_task3'
    # task_name = 'task1_task2_taskNRv2'
    task_name = args['task_name']
    save_path = args['save_path']
    num_layers = args['num_layers']

    print(f'[INFO]using model: {model_name}, pretrained model: {pretrained_name}')

    # subject_choice = 'ALL
    subject_choice = args['subjects']
    print(f'![Debug]using {subject_choice}')
    # eeg_type_choice = 'GD
    eeg_type_choice = args['eeg_type']
    print(f'[INFO]eeg type {eeg_type_choice}')
    # bands_choice = ['_t1']
    # bands_choice = ['_t1','_t2','_a1','_a2','_b1','_b2','_g1','_g2']
    bands_choice = args['eeg_bands']
    print(f'[INFO]using bands {bands_choice}')

    ''' set random seeds '''
    seed_val = 312
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    ''' set up device '''
    # use cuda
    if torch.cuda.is_available():
        # dev = "cuda:3"
        dev = args['cuda']
    else:
        dev = "cpu"
    device = torch.device(dev)
    print(f'[INFO]using device {dev}')
    print()

    ''' set up dataloader '''
    whole_dataset_dicts = []
    if 'task1' in task_name:
        dataset_path_task1 = r'./dataset/ZuCo/task1-SR/pickle/task1-SR-dataset.pickle'
        with open(dataset_path_task1, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))
    if 'task2' in task_name:
        dataset_path_task2 = r'./dataset/ZuCo/task2-NR/pickle/task2-NR-dataset.pickle'
        with open(dataset_path_task2, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))
    if 'task3' in task_name:
        dataset_path_task3 = r'./dataset/ZuCo/task3-TSR/pickle/task3-TSR-dataset.pickle'
        with open(dataset_path_task3, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))
    if 'taskNRv2' in task_name:
        dataset_path_taskNRv2 = r'./dataset/ZuCo/task2-NR-2.0/pickle/task2-NR-2.0-dataset.pickle'
        with open(dataset_path_taskNRv2, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))

    use_model = ''
    ''' set up model '''
    decoder_embedding_size = 768
    if pretrained_name == 'BART':
        decoder_embedding_size = 1024
        tokenizer = BartTokenizer.from_pretrained('./pretrain_model/bart-large')
        pretrained = BartForConditionalGeneration.from_pretrained('./pretrain_model/bart-large')
    elif pretrained_name == 'T5':
        decoder_embedding_size = 1024
        tokenizer = AutoTokenizer.from_pretrained('./pretrain_model/t5-large')
        pretrained = AutoModelForSeq2SeqLM.from_pretrained('./pretrain_model/t5-large')

    # 5-fold cross-validation
    for i in range(5):
        save_name = f'{task_name}_finetune_{model_name}_{pretrained_name}_{eeg_type_choice}_b_{batch_size}_epoch_{num_epochs}_bn_{bn}_lr_{lr}_dr_{dr}_dataset_setting_{dataset_setting}_fold_index_{i}'

        output_checkpoint_name_best = save_path + f'/{num_layers}/best/{save_name}.pt'
        output_checkpoint_name_last = save_path + f'/{num_layers}/last/{save_name}.pt'

        # train dataset
        train_set = ZuCo_dataset(whole_dataset_dicts, 'train', tokenizer, subject=subject_choice,
                                 eeg_type=eeg_type_choice, bands=bands_choice, setting=dataset_setting, fold_index=i)
        # dev dataset
        dev_set = ZuCo_dataset(whole_dataset_dicts, 'dev', tokenizer, subject=subject_choice, eeg_type=eeg_type_choice,
                               bands=bands_choice, setting=dataset_setting, fold_index=i)

        dataset_sizes = {'train': len(train_set), 'dev': len(dev_set)}
        print('[INFO]train_set size: ', len(train_set))
        print('[INFO]dev_set size: ', len(dev_set))

        # train dataloader
        train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
        # dev dataloader
        val_dataloader = DataLoader(dev_set, batch_size=batch_size, shuffle=False, num_workers=4)
        # dataloaders
        dataloaders = {'train': train_dataloader, 'dev': val_dataloader}

        if model_name == 'Decoder_Transformer':
            model = Decoder_Transformer(pretrained, in_feature=105 * len(bands_choice),
                                        decoder_embedding_size=decoder_embedding_size, num_layers=num_layers,
                                        block_number=bn, dropout=dr)
        elif model_name == 'Decoder_LSTM_masked_residual_attention':
            model = Decoder_LSTM_masked_residual_attention(pretrained, in_feature=105 * len(bands_choice),
                                                           decoder_embedding_size=decoder_embedding_size,
                                                           num_layers=num_layers, block_number=bn, dropout=dr)
        elif model_name == 'Decoder_GRU_masked_residual_attention':
            model = Decoder_GRU_masked_residual_attention(pretrained, in_feature=105 * len(bands_choice),
                                                          decoder_embedding_size=decoder_embedding_size,
                                                          num_layers=num_layers, block_number=bn, dropout=dr)
        elif model_name == 'Decoder_Naive':
            model = Decoder_Naive(pretrained, in_feature=105 * len(bands_choice),
                                  decoder_embedding_size=decoder_embedding_size, num_layers=num_layers, block_number=bn,
                                  dropout=dr)

        model.to(device)

        ''' training loop '''
        for name, param in model.named_parameters():
            param.requires_grad = True

        ''' set up optimizer and scheduler'''
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

        ''' set up loss function '''
        criterion = nn.CrossEntropyLoss()

        print()
        print('=== start training ... ===')
        # print training layers
        show_require_grad_layers(model)

        '''main loop'''
        trained_model = train_model(dataloaders, device, model, criterion, optimizer, exp_lr_scheduler,
                                    num_epochs=num_epochs, fold_index=i,
                                    checkpoint_path_best=output_checkpoint_name_best,
                                    checkpoint_path_last=output_checkpoint_name_last)
