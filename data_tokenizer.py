class TokenizeDataset:
    def __init__(self, seqs, intent_labels, slot_labels, intent_word2idx, slot_word2idx, tokenizer):
        self.seqs = seqs
        self.intent_labels = intent_labels
        self.slot_labels = slot_labels
        
        self.intent_word2idx = intent_word2idx
        self.slot_word2idx = slot_word2idx
        
        self.tokenizer = tokenizer
        
    def align_label(self, seq, intent_label, slot_label):
        tokens = self.tokenizer(seq, padding='max_length', max_length=50, truncation=True)
        
        slot_label_ids = [-100]
        for word_idx, word in enumerate(seq.split()):
            slot_label_ids += [self.slot_word2idx[slot_label[word_idx]]] + [-100]*(len(self.tokenizer.tokenize(word))-1)    # [slot label id] + [subword tails padding]
        if len(slot_label_ids) >= 50:
            slot_label_ids = slot_label_ids[:49] + [-100]
        else:
            slot_label_ids += [-100]*(50-len(slot_label_ids))
        
        tokens['intent_label_ids'] = [self.intent_word2idx[intent_label]]
        tokens['slot_label_ids'] = slot_label_ids
        
        return tokens

    def __getitem__(self, index):
        bert_input = self.align_label(self.seqs[index], self.intent_labels[index], self.slot_labels[index])
        return bert_input
    
    def __len__(self):
        return len(self.seqs)


class TokenizeDataset_POS:
    def __init__(self, seqs, intent_labels, slot_labels, intent_word2idx, slot_word2idx, tokenizer, pos_tokenizer):
        self.seqs = seqs
        self.intent_labels = intent_labels
        self.slot_labels = slot_labels
        
        self.intent_word2idx = intent_word2idx
        self.slot_word2idx = slot_word2idx
        
        self.tokenizer = tokenizer
        self.pos_tokenizer = pos_tokenizer
    
    def align_label(self, seq, intent_label, slot_label):
        tokens = self.tokenizer(seq, padding='max_length', max_length=50, truncation=True)
        pos_tokens = self.pos_tokenizer(seq, padding='max_length', max_length=50, truncation=True)
        
        slot_label_ids = [-100]
        for word_idx, word in enumerate(seq.split()):
            slot_label_ids += [self.slot_word2idx[slot_label[word_idx]]] + [-100]*(len(self.tokenizer.tokenize(word))-1)    # [slot label id] + [subword tails padding]
        if len(slot_label_ids) >= 50:
            slot_label_ids = slot_label_ids[:49] + [-100]
        else:
            slot_label_ids += [-100]*(50-len(slot_label_ids))
        
        
        tokens['intent_label_ids'] = [self.intent_word2idx[intent_label]]
        tokens['slot_label_ids'] = slot_label_ids

        tokens['pos_input_ids'] = pos_tokens['input_ids']
        tokens['pos_attention_mask'] = pos_tokens['attention_mask']
        tokens['pos_token_type_ids'] = pos_tokens['token_type_ids']
        
        return tokens

    def __getitem__(self, index):
        bert_input = self.align_label(self.seqs[index], self.intent_labels[index], self.slot_labels[index])
        return bert_input
    
    def __len__(self):
        return len(self.seqs)