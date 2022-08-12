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
        token_idxs = tokens.word_ids()
        
        pre_word_idx = None
        slot_label_ids = []
        for word_idx in token_idxs:
            if word_idx != pre_word_idx:
                try:
                    slot_label_ids.append(self.slot_word2idx[slot_label[word_idx]])
                except:
                    slot_label_ids.append(-100)

            elif word_idx == pre_word_idx or word_idx is None:
                slot_label_ids.append(-100)

            pre_word_idx = word_idx
        
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
        token_idxs = tokens.word_ids()
        
        pos_tokens = self.pos_tokenizer(seq, padding='max_length', max_length=50, truncation=True)
        
        pre_word_idx = None
        slot_label_ids = []
        for word_idx in token_idxs:
            if word_idx != pre_word_idx:
                try:
                    slot_label_ids.append(self.slot_word2idx[slot_label[word_idx]])
                except:
                    slot_label_ids.append(-100)

            elif word_idx == pre_word_idx or word_idx is None:
                slot_label_ids.append(-100)

            pre_word_idx = word_idx
        
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