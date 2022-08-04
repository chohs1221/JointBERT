import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel

class IntentClassifier(nn.Module):
    def __init__(self, hidden_size, num_intent_labels, classifier_dropout):
        super(IntentClassifier, self).__init__()
        self.dropout = nn.Dropout(classifier_dropout)
        self.linear = nn.Linear(hidden_size, num_intent_labels)

    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)


class SlotClassifier(nn.Module):
    def __init__(self, hidden_size, num_slot_labels, classifier_dropout):
        super(SlotClassifier, self).__init__()
        self.dropout = nn.Dropout(classifier_dropout)
        self.linear = nn.Linear(hidden_size, num_slot_labels)

    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)
    
    
class JointBERT(BertPreTrainedModel):
    def __init__(self, config, intent_labels, slot_labels):
        super().__init__(config)
        self.num_intent_labels = len(intent_labels)
        self.num_slot_labels = len(slot_labels)
        self.config = config

        self.bert = BertModel(config)

        classifier_dropout = config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        
        self.intent_classifier = IntentClassifier(config.hidden_size, self.num_intent_labels, classifier_dropout)
        self.slot_classifier = SlotClassifier(config.hidden_size, self.num_slot_labels, classifier_dropout)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids = None,
        attention_mask = None,
        token_type_ids = None,
        position_ids = None,
        head_mask = None,
        inputs_embeds = None,
        intent_label_ids = None,
        slot_label_ids = None,
        output_attentions = None,
        output_hidden_states = None,
        # return_dict = None
        ):
        # return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            # return_dict=return_dict,
        )   # sequence_output, pooled_output, (hidden_states), (attentions)

        sequence_output = outputs[0]
        pooled_output = outputs[1]  # [CLS]

        intent_logits = self.intent_classifier(pooled_output)
        slot_logits = self.slot_classifier(sequence_output)

        total_loss = 0
        # 1. Intent Softmax
        if intent_label_ids is not None:
            if self.num_intent_labels == 1:
                intent_loss_fct = nn.MSELoss()
                intent_loss = intent_loss_fct(intent_logits.squeeze(), intent_label_ids.squeeze())
            else:
                intent_loss_fct = nn.CrossEntropyLoss()
                intent_loss = intent_loss_fct(intent_logits.view(-1, self.num_intent_labels), intent_label_ids.view(-1))
            total_loss += intent_loss

        # 2. Slot Softmax
        if slot_label_ids is not None:
            loss_fct = nn.CrossEntropyLoss()
            slot_loss = loss_fct(slot_logits.view(-1, self.num_slot_labels), slot_label_ids.view(-1))
            total_loss += slot_loss

        outputs = ((intent_logits, slot_logits),) + outputs[2:]  # add hidden states and attention if they are here

        outputs = (total_loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions) # Logits is a tuple of intent and slot logits