import argparse


def get_argparse():
    parser = argparse.ArgumentParser()
    # paths
    parser.add_argument("--device", type=str, default="2", help="which device to use, if -1 use cpu, else use gpu")
    parser.add_argument("--data_dir", type=str, default='./data/processed/data_all', help="this is dataset path")
    parser.add_argument("--checkpoint_path", type=str, default='./save_model', help="path to save the model")
    parser.add_argument("--bert_path", type=str, default="./bert/albert_chinese_large",
                        help="path that stores bert_base_chinese model")

    parser.add_argument("--max_seq_length", type=int, default=128, help='the length of sequence')
    parser.add_argument("--do_train", type=bool, default=True, help="For distant debugging.")
    parser.add_argument("--train_batch_size", type=int, default=16, help="For distant debugging.")
    parser.add_argument("--predict_batch_size", type=int, default=16, help="For distant debugging.")
    parser.add_argument("--eval_batch_size", type=int, default=16, help="For distant debugging.")

    parser.add_argument("--epochs", type=int, default=10, help="For distant debugging.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="For distant debugging.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help='the initial learning rate for Adam')
    parser.add_argument("--crf_learning_rate", default=5e-5, type=float, help='the initial learning rate for '
                                                                              'crf and linear layer')

    parser.add_argument("--use_lstm", default=True, type=bool, help="whether to use lstm")
    parser.add_argument("--lstm_hidden_dim", default=384, type=int, help='the hidden size of lSTM')
    parser.add_argument("--lstm_rnn_layers", default=1, type=int, help="layers of rnn")
    parser.add_argument("--lstm_dropout", default=0.2, type=float, help="probability of dropout")
    parser.add_argument("--lstm_bidirectional", default=True, type=bool, help="whether to use bidirectional")

    parser.add_argument("--warmup_proportion", default=0.05, type=float, help='the initial learning rate for Adam')
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help='Epsilon for Adam optimizer')
    parser.add_argument("--weight_decay", default=0.01, type=float, help='Weight decay if we apply some')

    return parser
