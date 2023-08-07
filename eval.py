import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import pickle
import json

from tqdm import tqdm
from transformers import BartTokenizer, BartForConditionalGeneration, AutoModelForSeq2SeqLM, AutoTokenizer
from data import ZuCo_dataset
from model import Decoder_Naive, Decoder_Transformer, Decoder_LSTM_masked_residual_attention, \
    Decoder_GRU_masked_residual_attention
from config import get_config
import nltk
from nltk.translate.bleu_score import corpus_bleu
from nltk.tokenize import word_tokenize
from rouge import Rouge
from torchmetrics.text import TranslationEditRate
import evaluate
import math
from datasets import DownloadConfig


# NIST is an improvement to BLEU, introducing the concept of the amount of information for each n-gram.
def calculate_nist(reference, translation):
    nist_score = nltk.translate.nist_score.sentence_nist([reference], translation)
    return nist_score


def tokenize_sentence(sentence):
    return word_tokenize(sentence)


# METEOR is an improvement on BLEU, which considers the alignment relationship
# between the generated text and the reference text, and uses WordNet to calculate
# specific sequence matching, synonyms, roots and affixes, and the matching relationship between paraphrases.
def calculate_meteor(reference, translation):
    reference_tokens = tokenize_sentence(reference)
    translation_tokens = tokenize_sentence(translation)
    meteor_score_val = nltk.translate.meteor_score.meteor_score([reference_tokens], translation_tokens)
    return meteor_score_val


def calculate_ter(reference, translation):
    metric = TranslationEditRate()
    ter_score = metric([reference], translation)
    return ter_score

def calculate_bleurt(reference, translation, bleurt_model):
    results = bleurt_model.compute(predictions=[translation], references=[reference])
    return results['scores'][0]


def get_results(target_tokens_list, pred_tokens_list, pred_string_list, target_string_list, test_epoch,
                output_total_results_path, bleurt_model):
    """ calculate corpus bleu score """
    weights_list = [(1.0,), (0.5, 0.5), (1. / 3., 1. / 3., 1. / 3.), (0.25, 0.25, 0.25, 0.25)]

    for weight in weights_list:
        # print('weight:',weight)
        corpus_bleu_score = corpus_bleu(target_tokens_list, pred_tokens_list, weights=weight)
        with open(output_total_results_path, 'a+') as f:
            f.write(f'corpus BLEU-{len(list(weight))} score: {corpus_bleu_score}\n')

    """ calculate rouge score """
    rouge = Rouge()
    try:
        rouge_scores = rouge.get_scores(pred_string_list, target_string_list, avg=True)
    except:
        rouge_scores = {"rouge-1": {"r": 0.0, "p": 0.0, "f": 0.0},
                        "rouge-2": {"r": 0.0, "p": 0.0, "f": 0.0},
                        "rouge-l": {"r": 0.0, "p": 0.0, "f": 0.0}}

    with open(output_total_results_path, 'a+') as f:
        f.write('\n')
        f.write(json.dumps(rouge_scores, ensure_ascii=True))
        f.write('\n')

    nist_scores = 0.0
    meteor_scores = 0.0
    ter_scores = 0.0
    bleurt_scores = 0.0

    for i in range(len(target_string_list)):
        # print(target_string_list[i])

        if pred_string_list[i] == "" or pred_string_list[i] == None:
            continue

        """ calculate nist score """
        nist_score = calculate_nist(target_string_list[i], pred_string_list[i])
        nist_scores += nist_score

        """ calculate meteor score """
        meteor_score = calculate_meteor(target_string_list[i], pred_string_list[i])
        meteor_scores += meteor_score

        """ calculate ter score """
        ter_score = calculate_ter(target_string_list[i], pred_string_list[i])
        ter_scores += ter_score

        """ calculate bleurt score """
        bleurt_score = calculate_bleurt(target_string_list[i], pred_string_list[i], bleurt_model)
        bleurt_scores += bleurt_score

    nist_score = nist_scores / len(target_string_list)
    meteor_score = meteor_scores / len(target_string_list)
    ter_score = ter_scores / len(target_string_list)
    bleurt_score = bleurt_scores / len(target_string_list)

    with open(output_total_results_path, 'a+') as f:
        f.write('\n')

        bleu_score = corpus_bleu(target_tokens_list, pred_tokens_list, weights=(1.0,))
        f.write(f'bleu score: {bleu_score}\n')

        rouge_score = rouge_scores['rouge-l']['f']
        f.write(f'rouge score: {rouge_score}\n')
        f.write(f'nist score: {nist_score}\n')
        f.write(f'meteor_score: {meteor_score}\n')
        f.write(f'ter score: {ter_score}\n')
        f.write(f'bleurt score: {bleurt_score}\n')

    return bleu_score, rouge_score, nist_score, meteor_score, ter_score, bleurt_score


def get_mean_variance(results_list):
    # Calculate the mean
    mean = sum(results_list) / len(results_list)

    # Calculate the standard deviation
    variance = sum((result - mean) ** 2 for result in results_list) / len(results_list)
    std = math.sqrt(variance)
    return mean, std


def eval_model(dataloaders, device, model, num_layers=8, fold_index=0, test_epoch=19, bleurt_model=None):
    output_total_results_path = f'./eval_results/{num_layers}/{task_name}-{model_name}-{pretrained_name}-{eeg_type_choice}_bn_{bn}_lr_{lr}_dr_{dr}-total_decoding_results_fold_index_{fold_index}.txt'

    model.eval()  # Set model to evaluate mode

    # Iterate over data.
    target_tokens_list = []
    target_string_list = []
    pred_tokens_list = []
    pred_string_list = []

    # Iterate over data.
    for input_embeddings, seq_len, input_masks, input_mask_invert, target_ids, target_mask, sentiment_labels, sent_level_EEG in tqdm(
            dataloaders['test']):

        # load in batch
        input_embeddings_batch = input_embeddings.to(device).float()
        input_masks_batch = input_masks.to(device)
        input_mask_invert_batch = input_mask_invert.to(device)
        target_ids_batch = target_ids.to(device)

        target_tokens = tokenizer.convert_ids_to_tokens(target_ids_batch[0].tolist(),
                                                        skip_special_tokens=True)
        target_string = tokenizer.decode(target_ids_batch[0], skip_special_tokens=True)
        target_tokens_list.append([target_tokens])
        target_string_list.append(target_string)

        # forward
        seq2seqLMoutput = model(input_embeddings_batch, input_masks_batch, input_mask_invert_batch,
                                target_ids_batch)

        logits = seq2seqLMoutput.logits  # 8*48*50265
        # logits = logits.permute(0,2,1)
        # print('permuted logits size:', logits.size())
        probs = logits[0].softmax(dim=1)
        # print('probs size:', probs.size())
        values, predictions = probs.topk(1)
        # print('predictions before squeeze:',predictions.size())
        predictions = torch.squeeze(predictions)
        predicted_string = tokenizer.decode(predictions).split('</s></s>')[0].replace('<s>', '')
        # print('predicted string:',predicted_string)

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

    return get_results(target_tokens_list, pred_tokens_list, pred_string_list, target_string_list, test_epoch,
                       output_total_results_path, bleurt_model)


def show_require_grad_layers(model):
    print()
    print(' require_grad layers:')
    # sanity check
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(' ', name)


if __name__ == '__main__':
    args = get_config('eval_decoding')

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

    bleu_score_list_loss, rouge_score_list_loss, nist_score_list_loss, meteor_score_list_loss, ter_score_list_loss, bleurt_score_list_loss = [], [], [], [], [], []

    bleurt_model = evaluate.load("bleurt", "BLEURT-20", download_config=DownloadConfig(use_etag=False))

    # 5-fold cross-validation
    for i in range(5):
        save_name = f'{task_name}_finetune_{model_name}_{pretrained_name}_{eeg_type_choice}_b_{batch_size}_epoch_{num_epochs}_bn_{bn}_lr_{lr}_dr_{dr}_dataset_setting_{dataset_setting}_fold_index_{i}'
        checkpoint_path_best_loss = save_path + f'/{num_layers}/best/{save_name}.pt'

        # test dataset
        test_set = ZuCo_dataset(whole_dataset_dicts, 'dev', tokenizer, subject=subject_choice, eeg_type=eeg_type_choice,
                                bands=bands_choice, setting=dataset_setting, fold_index=i)

        dataset_sizes = {'test': len(test_set)}
        print('[INFO] testset size: ', len(test_set))

        # test dataloader
        test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)
        # dataloaders
        dataloaders = {'test': test_dataloader}

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

        model.load_state_dict(torch.load(checkpoint_path_best_loss, map_location=device))
        model.to(device)

        ''' training loop '''
        for name, param in model.named_parameters():
            param.requires_grad = True

        print()
        print('=== start training ... ===')
        # print training layers
        show_require_grad_layers(model)

        '''main loop'''
        bleu_score_loss, rouge_score_loss, nist_score_loss, meteor_score_loss, ter_score_loss, bleurt_score_loss = eval_model(
            dataloaders, device, model, num_layers=num_layers, fold_index=i, test_epoch=test_epoch,
            bleurt_model=bleurt_model)
        bleu_score_list_loss.append(bleu_score_loss)
        rouge_score_list_loss.append(rouge_score_loss)
        nist_score_list_loss.append(nist_score_loss)
        meteor_score_list_loss.append(meteor_score_loss)
        ter_score_list_loss.append(ter_score_loss)
        bleurt_score_list_loss.append(bleurt_score_loss)

    # Calculate mean and variance
    mean_bleu_score_loss, std_bleu_score_loss = get_mean_variance(bleu_score_list_loss)
    mean_rouge_score_loss, std_rouge_score_loss = get_mean_variance(rouge_score_list_loss)
    mean_nist_score_loss, std_nist_score_loss = get_mean_variance(nist_score_list_loss)
    mean_meteor_score_loss, std_meteor_score_loss = get_mean_variance(meteor_score_list_loss)
    mean_ter_score_loss, std_ter_score_loss = get_mean_variance(ter_score_list_loss)
    mean_bleurt_score_loss, std_bleurt_score_loss = get_mean_variance(bleurt_score_list_loss)

    output_men_results_path = f'./eval_results/{num_layers}/{task_name}-{model_name}-{pretrained_name}-{eeg_type_choice}_bn_{bn}_lr_{lr}_dr_{dr}-total_decoding_results_mean_std.txt'
    with open(output_men_results_path, 'a+') as f:
        f.write(f'mean_bleu_score: {mean_bleu_score_loss}, std_bleu_score: {std_bleu_score_loss}\n')
        f.write(f'mean_rouge_score: {mean_rouge_score_loss}, std_rouge_score: {std_rouge_score_loss}\n')
        f.write(f'mean_nist_score: {mean_nist_score_loss}, std_nist_score: {std_nist_score_loss}\n')
        f.write(f'mean_meteor_score: {mean_meteor_score_loss}, std_meteor_score: {std_meteor_score_loss}\n')
        f.write(f'mean_ter_score: {mean_ter_score_loss}, std_ter_score: {std_ter_score_loss}\n')
        f.write(f'mean_bleurt_score: {mean_bleurt_score_loss}, std_bleurt_score: {std_bleurt_score_loss}\n')

    print("******************* finished *******************")
