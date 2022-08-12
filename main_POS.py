import argparse
import time
from collections import defaultdict

import numpy as np

import torch
from transformers import BertConfig, BertTokenizerFast, TrainingArguments, Trainer, AutoTokenizer, AutoModel

from utils import seed_everything, empty_cuda_cache, compute_metrics
from modeling import JointBERT, JointBERT_POS
from data_loader import LoadDataset
from data_tokenizer import TokenizeDataset, TokenizeDataset_POS


def main(args):
    # Parse Argument
    TASK = str(args.task)
    EPOCH = int(args.epoch)
    LR = float(args.lr)
    BATCH_SIZE = int(args.batch)
    SEED = int(args.seed)
    print(f'============================================================')
    print(f"{time.strftime('%c', time.localtime(time.time()))}")
    print(f'TASK: {TASK}')
    print(f'EPOCH: {EPOCH}')
    print(f'LR: {LR}')
    print(f'BATCH_SIZE: {BATCH_SIZE}')
    print(f'SEED: {SEED}\n')


    # Set Random Seed
    seed_everything(SEED)
    

    # Load Dataset
    seq_train = LoadDataset.load_dataset(f'./data/{TASK}/train/seq.in')
    seq_dev = LoadDataset.load_dataset(f'./data/{TASK}/dev/seq.in')
    seq_test = LoadDataset.load_dataset(f'./data/{TASK}/test/seq.in')

    intent_train = LoadDataset.load_dataset(f'./data/{TASK}/train/label')
    intent_dev = LoadDataset.load_dataset(f'./data/{TASK}/dev/label')
    intent_test = LoadDataset.load_dataset(f'./data/{TASK}/test/label')
    intent_labels = LoadDataset.load_dataset(f'./data/{TASK}/intent_label_vocab')

    slot_train = LoadDataset.load_dataset(f'./data/{TASK}/train/seq.out', slot = True)
    slot_dev = LoadDataset.load_dataset(f'./data/{TASK}/dev/seq.out', slot = True)
    slot_test = LoadDataset.load_dataset(f'./data/{TASK}/test/seq.out', slot = True)
    slot_labels = LoadDataset.load_dataset(f'./data/{TASK}/slot_label_vocab')


    # Label Indexing
    intent_word2idx = defaultdict(int, {k: v for v, k in enumerate(intent_labels)})
    intent_idx2word = {v: k for v, k in enumerate(intent_labels)}

    slot_word2idx = defaultdict(int, {k: v for v, k in enumerate(slot_labels)})
    slot_idx2word = {v: k for v, k in enumerate(slot_labels)}


    # Load Tokenizer & Model
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    pos_tokenizer = AutoTokenizer.from_pretrained("TweebankNLP/bertweet-tb2_ewt-pos-tagging")

    model_config = BertConfig.from_pretrained("bert-base-uncased", num_labels = len(intent_idx2word), problem_type = "single_label_classification", id2label = intent_idx2word, label2id = intent_word2idx)

    pos_model = AutoModel.from_pretrained("TweebankNLP/bertweet-tb2_ewt-pos-tagging")
    pos_model.eval()
    model = JointBERT_POS.from_pretrained("bert-base-uncased", config = model_config, intent_labels = intent_labels, slot_labels = slot_labels, pos_model = pos_model)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device);


    # Tokenize Datasets
    train_dataset = TokenizeDataset_POS(seq_train, intent_train, slot_train, intent_word2idx, slot_word2idx, tokenizer, pos_tokenizer)
    dev_dataset = TokenizeDataset_POS(seq_dev, intent_dev, slot_dev, intent_word2idx, slot_word2idx, tokenizer, pos_tokenizer)
    test_dataset = TokenizeDataset_POS(seq_test, intent_test, slot_test, intent_word2idx, slot_word2idx, tokenizer, pos_tokenizer)


    # Set Training Arguments and Train
    arguments = TrainingArguments(
        output_dir='checkpoints',
        do_train=True,
        do_eval=True,

        num_train_epochs=EPOCH,
        learning_rate = LR,

        save_strategy="epoch",
        save_total_limit=3,
        evaluation_strategy="epoch",
        load_best_model_at_end=True,
        
        report_to = 'none',

        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=64,
        gradient_accumulation_steps=1,
        dataloader_num_workers=0,
        fp16=True,

    )

    trainer = Trainer(
        model,
        arguments,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset
    )

    empty_cuda_cache()
    trainer.train()
    model.save_pretrained(f"checkpoints/{TASK}_ep{EPOCH}")

    # last_model = JointBERT.from_pretrained("./checkpoints/checkpoint-1050", config = model_config, intent_labels = intent_labels, slot_labels = slot_labels)


    # Get Intent, Slot Labels
    intent_label_ids = []
    slot_label_ids = []

    with open(f'./data/{TASK}/test/label', 'r', encoding='utf-8') as intent_f, \
        open(f'./data/{TASK}/test/seq.out', 'r', encoding='utf-8') as slot_f:
        for line in intent_f:
            line = line.strip()
            intent_label_ids.append(line)
        intent_label_ids = np.array(intent_label_ids)
        for line in slot_f:
            line = line.strip().split()
            slot_label_ids.append(line)


    # Predict
    def predict(model, seqs):
        model.to('cpu')
        pred_intent_ids = []
        pred_slot_ids = []

        for i in range(len(seqs)):
            input_seq = tokenizer(seqs[i], padding='max_length', max_length=50, truncation=True, return_tensors='pt')
            pos_input_seq = pos_tokenizer(seqs[i], padding='max_length', max_length=50, truncation=True, return_tensors='pt')
            
            model.eval()
            with torch.no_grad():
                _, (intent_logits, slot_logits) = model(input_ids = input_seq['input_ids'],
                                                        attention_mask = input_seq['attention_mask'],
                                                        token_type_ids = input_seq['token_type_ids'],

                                                        pos_input_ids = pos_input_seq['input_ids'],
                                                        pos_attention_mask = pos_input_seq['attention_mask'],
                                                        pos_token_type_ids = pos_input_seq['token_type_ids'],
                                                        )

            # Intent
            pred_intent_ids.append(intent_idx2word[intent_logits[0].argmax().item()])

            # Slot
            slot_logits_size = slot_logits[0].shape[0]
            slot_logits_mask = np.array(test_dataset[i]['slot_label_ids'][:slot_logits_size]) != -100
            slot_logits_clean = slot_logits[0][slot_logits_mask]
            pred_slot_ids.append([slot_idx2word[i.item()] for i in slot_logits_clean.argmax(dim=1)])

        return np.array(pred_intent_ids), pred_slot_ids

    pred_intent_ids, pred_slot_ids = predict(model, seq_test)

    print(f"\n{time.strftime('%c', time.localtime(time.time()))}")
    res = compute_metrics(pred_intent_ids, intent_label_ids, pred_slot_ids, slot_label_ids)
    for k, v in res.items():
        print(f'{k:<20}: {v}')
    print(f'============================================================\n\n\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--task', default='snips')
    parser.add_argument('--epoch', default=30)
    parser.add_argument('--lr', default=5e-5)
    parser.add_argument('--batch', default=128)
    parser.add_argument('--seed', default=1234)

    args = parser.parse_args()
    main(args)