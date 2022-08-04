from collections import defaultdict

import numpy as np

import gc

import torch
from transformers import BertConfig, BertTokenizerFast, TrainingArguments, Trainer



from utils import seed_everything, empty_cuda_cache, compute_metrics
from modeling import JointBERT
from data_loader import LoadDataset
from data_tokenizer import TokenizeDataset


TASK = 'atis'

seed_everything(1234)
    
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

intent_word2idx = defaultdict(int, {k: v for v, k in enumerate(intent_labels)})
intent_idx2word = {v: k for v, k in enumerate(intent_labels)}

slot_word2idx = defaultdict(int, {k: v for v, k in enumerate(slot_labels)})
slot_idx2word = {v: k for v, k in enumerate(slot_labels)}

tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

model_config = BertConfig.from_pretrained("bert-base-uncased", num_labels = len(intent_idx2word), problem_type = "single_label_classification", id2label = intent_idx2word, label2id = intent_word2idx)
# model_config.classifier_dropout

model = JointBERT.from_pretrained("bert-base-uncased", config = model_config, intent_labels = intent_labels, slot_labels = slot_labels)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device);


train_dataset = TokenizeDataset(seq_train, intent_train, slot_train, intent_word2idx, slot_word2idx, tokenizer)
dev_dataset = TokenizeDataset(seq_dev, intent_dev, slot_dev, intent_word2idx, slot_word2idx, tokenizer)
test_dataset = TokenizeDataset(seq_test, intent_test, slot_test, intent_word2idx, slot_word2idx, tokenizer)

arguments = TrainingArguments(
    output_dir='checkpoints',
    do_train=True,
    do_eval=True,

    num_train_epochs=30,
    learning_rate = 5e-5,

    save_strategy="epoch",
    save_total_limit=2,
    evaluation_strategy="epoch",
    load_best_model_at_end=True,
    
    report_to = 'none',

    per_device_train_batch_size=128,
    per_device_eval_batch_size=32,
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
model.save_pretrained(f"checkpoints/first_checkpoint")

intent_label_ids = []
slot_label_ids = []

with open(f'./data/{TASK}/test/label', 'r', encoding='utf-8') as intent_f, \
    open(f'./data/{TASK}/test/seq.out', 'r', encoding='utf-8') as slot_f:
    for line in intent_f:
        line = line.strip()
        intent_label_ids.append(line)
    for line in slot_f:
        line = line.strip().split()
        slot_label_ids.append(line)

intent_label_ids = np.array(intent_label_ids)
slot_label_ids = np.array(slot_label_ids)

def predict(model, seqs):
    model.to('cpu')
    pred_intent_ids = []
    pred_slot_ids = []

    for i in range(len(seqs)):
        input_seq = tokenizer(seq_test[i], return_tensors='pt')
        
        model.eval()
        with torch.no_grad():
            _, (intent_logits, slot_logits) = model(**input_seq)

        # Intent
        pred_intent_ids.append(intent_idx2word[intent_logits[0].argmax().item()])

        # Slot
        slot_logits_size = slot_logits[0].shape[0]
        slot_logits_mask = np.array(test_dataset[i]['slot_label_ids'][:slot_logits_size]) != -100
        slot_logits_clean = slot_logits[0][slot_logits_mask]
        pred_slot_ids.append([slot_idx2word[i.item()] for i in slot_logits_clean.argmax(dim=1)])

    return np.array(pred_intent_ids), np.array(pred_slot_ids)

pred_intent_ids, pred_slot_ids = predict(model, seq_test)

res = compute_metrics(pred_intent_ids, intent_label_ids, pred_slot_ids, slot_label_ids)


