import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pickle
import json
from transformers import BertLMHeadModel, BartTokenizer, BartForConditionalGeneration, BertTokenizer, BertConfig
from data import ZuCo_dataset
from model import Decoder_Naive, Decoder_GRU_masked_residual_attention
from nltk.translate.bleu_score import corpus_bleu
from rouge import Rouge
from config import get_config


def eval_model(dataloaders, device, tokenizer, model, output_all_results_path='./results/temp.txt',
               output_total_results_path='./results/temp.txt'):
    model.eval()  # Set model to evaluate mode
    running_loss = 0.0

    # Iterate over data.
    sample_count = 0

    target_tokens_list = []
    target_string_list = []
    pred_tokens_list = []
    pred_string_list = []
    with open(output_all_results_path, 'w') as f:
        for input_embeddings, seq_len, input_masks, input_mask_invert, target_ids, target_mask, sentiment_labels, sent_level_EEG in \
                dataloaders['test']:
            # load in batch
            input_embeddings_batch = input_embeddings.to(device).float()
            input_masks_batch = input_masks.to(device)
            target_ids_batch = target_ids.to(device)
            input_mask_invert_batch = input_mask_invert.to(device)

            target_tokens = tokenizer.convert_ids_to_tokens(target_ids_batch[0].tolist(), skip_special_tokens=True)
            target_string = tokenizer.decode(target_ids_batch[0], skip_special_tokens=True)
            f.write(f'target string: {target_string}\n')

            # add to list for later calculate bleu metric
            target_tokens_list.append([target_tokens])
            target_string_list.append(target_string)

            """replace padding ids in target_ids with -100"""
            target_ids_batch[target_ids_batch == tokenizer.pad_token_id] = -100

            # forward
            seq2seqLMoutput = model(input_embeddings_batch, input_masks_batch, input_mask_invert_batch,
                                    target_ids_batch)

            """calculate loss"""
            loss = seq2seqLMoutput.loss  # use the BART language modeling loss

            # get predicted tokens
            logits = seq2seqLMoutput.logits
            probs = logits[0].softmax(dim=1)
            values, predictions = probs.topk(1)
            predictions = torch.squeeze(predictions)
            predicted_string = tokenizer.decode(predictions).split('</s></s>')[0].replace('<s>', '')
            # print('predicted string:',predicted_string)
            f.write(f'predicted string: {predicted_string}\n')
            f.write(f'################################################\n\n\n')

            # convert to int list
            predictions = predictions.tolist()
            truncated_prediction = []
            for t in predictions:
                if t != tokenizer.eos_token_id:
                    truncated_prediction.append(t)
                else:
                    break
            pred_tokens = tokenizer.convert_ids_to_tokens(truncated_prediction, skip_special_tokens=True)
            # print('predicted tokens:',pred_tokens)
            pred_tokens_list.append(pred_tokens)
            pred_string_list.append(predicted_string)
            # print('################################################')
            # print()

            sample_count += 1
            # statistics
            running_loss += loss.item() * input_embeddings_batch.size()[0]  # batch loss
            # print('[DEBUG]loss:',loss.item())
            # print('#################################')

    epoch_loss = running_loss / dataset_sizes['test_set']
    print('test loss: {:4f}'.format(epoch_loss))

    """ calculate corpus bleu score """
    weights_list = [(1.0,), (0.5, 0.5), (1. / 3., 1. / 3., 1. / 3.), (0.25, 0.25, 0.25, 0.25)]
    for weight in weights_list:
        # print('weight:',weight)
        corpus_bleu_score = corpus_bleu(target_tokens_list, pred_tokens_list, weights=weight)
        print(f'corpus BLEU-{len(list(weight))} score:', corpus_bleu_score)

    print()
    """ calculate rouge score """
    rouge = Rouge()
    rouge_scores = rouge.get_scores(pred_string_list, target_string_list, avg=True)
    print(rouge_scores)


if __name__ == '__main__':
    ''' get args'''
    args = get_config('eval_decoding')
    checkpoint_path = args['checkpoint_path']

    ''' load training config'''
    training_config = json.load(open(args['config_path']))

    batch_size = 1
    subject_choice = training_config['subjects']
    print(f'[INFO]subjects: {subject_choice}')
    eeg_type_choice = training_config['eeg_type']
    print(f'[INFO]eeg type: {eeg_type_choice}')
    bands_choice = training_config['eeg_bands']
    print(f'[INFO]using bands: {bands_choice}')

    dataset_setting = 'unique_sent'
    task_name = training_config['task_name']
    num_layers = training_config['num_layers']
    model_name = training_config['model_name']
    pretrained_name = training_config['pretrained']

    output_all_results_path = f'./eval_results/{num_layers}/{task_name}-{model_name}-{pretrained_name}-{eeg_type_choice}-all_decoding_results.txt'
    output_total_results_path = f'./eval_results/{num_layers}/{task_name}-{model_name}-{pretrained_name}-{eeg_type_choice}-total_decoding_results.txt'

    ''' set random seeds '''
    seed_val = 312
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    ''' set up device '''
    # use cuda
    if torch.cuda.is_available():
        dev = args['cuda']
    else:
        dev = "cpu"
    # CUDA_VISIBLE_DEVICES=0,1,2,3  
    device = torch.device(dev)
    print(f'[INFO]using device {dev}')

    ''' set up dataloader '''
    whole_dataset_dicts = []
    if 'task1' in task_name:
        dataset_path_task1 = './dataset/ZuCo/task1-SR/pickle/task1-SR-dataset.pickle'
        with open(dataset_path_task1, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))
    if 'task2' in task_name:
        dataset_path_task2 = './dataset/ZuCo/task2-NR/pickle/task2-NR-dataset.pickle'
        with open(dataset_path_task2, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))
    if 'task3' in task_name:
        dataset_path_task3 = './dataset/ZuCo/task3-TSR/pickle/task3-TSR-dataset.pickle'
        with open(dataset_path_task3, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))
    if 'taskNRv2' in task_name:
        dataset_path_taskNRv2 = './dataset/ZuCo/task2-NR-2.0/pickle/task2-NR-2.0-dataset.pickle'
        with open(dataset_path_taskNRv2, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))
    print()

    use_model = ''
    ''' set up model '''
    decoder_embedding_size = 768
    if pretrained_name == 'BART':
        decoder_embedding_size = 1024
        tokenizer = BartTokenizer.from_pretrained('./pretrain_model/bart-large')
        pretrained = BartForConditionalGeneration.from_pretrained('./pretrain_model/bart-large')
    elif pretrained_name == 'BERT':
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        config = BertConfig.from_pretrained("bert-base-cased")
        config.is_decoder = True
        pretrained = BertLMHeadModel.from_pretrained('bert-base-cased', config=config)

    # test dataset
    test_set = ZuCo_dataset(whole_dataset_dicts, 'test', tokenizer, subject=subject_choice, eeg_type=eeg_type_choice,
                            bands=bands_choice, setting=dataset_setting)

    dataset_sizes = {"test_set": len(test_set)}
    print('[INFO]test_set size: ', len(test_set))

    # dataloaders
    test_dataloader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=4)

    dataloaders = {'test': test_dataloader}

    if model_name == 'Decoder_GRU_masked_residual_attention':
        model = Decoder_GRU_masked_residual_attention(pretrained, in_feature=105 * len(bands_choice),
                                                      decoder_embedding_size=decoder_embedding_size,
                                                      num_layers=num_layers)
    elif model_name == 'Decoder_Naive':
        model = Decoder_Naive(pretrained, in_feature=105 * len(bands_choice),
                              decoder_embedding_size=decoder_embedding_size, num_layers=num_layers)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)

    criterion = nn.CrossEntropyLoss()

    ''' eval '''
    eval_model(dataloaders, device, tokenizer, model, output_all_results_path=output_all_results_path,
               output_total_results_path=output_total_results_path)
