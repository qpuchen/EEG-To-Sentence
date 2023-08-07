import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_config(case):
    if case == 'train_decoding':
        # args config for training EEG-To-Text decoder
        parser = argparse.ArgumentParser(description='Specify config args for training EEG-To-Text decoder')

        parser.add_argument('-m', '--model_name',
                            help='choose from {Decoder_Transformer, Decoder_Naive, Decoder_LSTM,Decoder_GRU}',
                            default="Decoder_", required=True)
        parser.add_argument('-t', '--task_name',
                            help='choose from {task1,task1_task2, task1_task2_task3,task1_task2_taskNRv2}',
                            default="task1", required=True)
        parser.add_argument('-pre', '--pretrained', help='choose from {BART, mT5}', default="BART", required=True)
        parser.add_argument('-epoch', '--num_epoch', type=int, help='num_epoch', default=30, required=True)
        parser.add_argument('-bn', '--block_number', type=int, help='block number', default=4, required=False)
        parser.add_argument('-lr', '--learning_rate', type=float, help='learning_rate', default=0.0000005,
                            required=True)
        parser.add_argument('-dr', '--dropout', type=float, help='dropout', default=0.1,
                            required=False)
        parser.add_argument('-b', '--batch_size', type=int, help='batch_size', default=32, required=True)
        parser.add_argument('-l', '--num_layers', type=int, help='num_layers', default=8, required=True)
        parser.add_argument('-s', '--save_path', help='checkpoint save path', default='./checkpoints/decoding',
                            required=True)
        parser.add_argument('-subj', '--subjects', help='use all subjects or specify a particular one', default='ALL',
                            required=False)
        parser.add_argument('-eeg', '--eeg_type', help='choose from {GD, FFD, TRT}', default='GD', required=False)
        parser.add_argument('-band', '--eeg_bands', nargs='+', help='specify freqency bands',
                            default=['_t1', '_t2', '_a1', '_a2', '_b1', '_b2', '_g1', '_g2'], required=False)
        parser.add_argument('-cuda', '--cuda', help='specify cuda device name, e.g. cuda:0, cuda:1, etc',
                            default='cuda:0')

        args = vars(parser.parse_args())

    elif case == 'eval_decoding':
        # args config for evaluating EEG-To-Text decoder
        parser = argparse.ArgumentParser(description='Specify config args for evaluate EEG-To-Text decoder')
        parser.add_argument('-m', '--model_name',
                            help='choose from {Decoder_Transformer, Decoder_Naive, Decoder_LSTM,Decoder_GRU}',
                            default="Decoder_", required=True)
        parser.add_argument('-t', '--task_name',
                            help='choose from {task1,task1_task2, task1_task2_task3,task1_task2_taskNRv2}',
                            default="task1", required=True)
        parser.add_argument('-pre', '--pretrained', help='choose from {BART, mT5}', default="BART", required=True)
        parser.add_argument('-epoch', '--num_epoch', type=int, help='num_epoch', default=30, required=True)
        parser.add_argument('-bn', '--block_number', type=int, help='block number', default=4, required=False)
        parser.add_argument('-lr', '--learning_rate', type=float, help='learning_rate', default=0.0000005,
                            required=True)
        parser.add_argument('-dr', '--dropout', type=float, help='dropout', default=0.1,
                            required=False)
        parser.add_argument('-b', '--batch_size', type=int, help='batch_size', default=32, required=True)
        parser.add_argument('-l', '--num_layers', type=int, help='num_layers', default=8, required=True)
        parser.add_argument('-s', '--save_path', help='checkpoint save path', default='./checkpoints/decoding',
                            required=True)
        parser.add_argument('-subj', '--subjects', help='use all subjects or specify a particular one', default='ALL',
                            required=False)
        parser.add_argument('-eeg', '--eeg_type', help='choose from {GD, FFD, TRT}', default='GD', required=False)
        parser.add_argument('-band', '--eeg_bands', nargs='+', help='specify freqency bands',
                            default=['_t1', '_t2', '_a1', '_a2', '_b1', '_b2', '_g1', '_g2'], required=False)
        parser.add_argument('-cuda', '--cuda', help='specify cuda device name, e.g. cuda:0, cuda:1, etc',
                            default='cuda:0')
        args = vars(parser.parse_args())

    return args
