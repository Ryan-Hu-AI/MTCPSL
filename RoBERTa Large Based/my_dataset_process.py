from torch.utils.data import Dataset
import torch
import json
from transformers import AutoTokenizer

class News_Propa_Reason_Score_Inputs_PropaFakeDataset(Dataset):
    def __init__(self, jsonl_path, model_name, max_sequence_length):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.data = []

        for line in open(jsonl_path, 'r', encoding='utf-8'):
            inst = json.loads(line)

            # Tokenize and encode fake news
            original_inputs = self.tokenizer(inst['original_txt'],  max_length=max_sequence_length, padding="max_length", truncation=True)
            # Tokenize and encode propaganda news
            prop_inputs = self.tokenizer(inst['prop_txt'],  max_length=max_sequence_length, padding="max_length", truncation=True)
            # Tokenize and encode propaganda news
            reason_inputs = self.tokenizer(inst['reason'], max_length=max_sequence_length, padding="max_length", truncation=True)
            prop_label = self.class_encode(inst['prop_label'])
            self.data.append({
                'input_ids_original': original_inputs['input_ids'],
                'attention_mask_original': original_inputs['attention_mask'],
                'input_ids_prop': prop_inputs['input_ids'],
                'attention_mask_prop': prop_inputs['attention_mask'],
                'input_ids_reason': reason_inputs['input_ids'],
                'attention_mask_reason': prop_inputs['attention_mask'],  #In paper 4.Method, Aligh Reason with Propaganda(P Mask R)
                'news_label': inst['news_label'],  # Label for  news
                'prop_label': prop_label,   # Label for propaganda news
                'prop_score': float(inst['prop_score'])  # Propaganda score as a float
            })
    def class_encode(self, label):
        # Convert numeric label to number encoded format
        if label == "Loaded_Language":
            class_label = 0
        elif label == "Exaggeration":
            class_label = 1
        elif label == "Appeal_to_Fear":
            class_label = 2
        return class_label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        return (data['input_ids_original'], data['attention_mask_original'],
                data['input_ids_prop'], data['attention_mask_prop'],
                data['input_ids_reason'], data['attention_mask_reason'],
                data['news_label'], data['prop_label'], data['prop_score'])

    def collate_fn(self, batch):
        input_ids_news = torch.cuda.LongTensor([inst[0] for inst in batch])
        attention_masks_news = torch.cuda.LongTensor([inst[1] for inst in batch])
        input_ids_prop = torch.cuda.LongTensor([inst[2] for inst in batch])
        attention_masks_prop = torch.cuda.LongTensor([inst[3] for inst in batch])
        input_ids_reason = torch.cuda.LongTensor([inst[4] for inst in batch])
        attention_masks_reason = torch.cuda.LongTensor([inst[5] for inst in batch])
        news_labels = torch.cuda.FloatTensor([inst[6] for inst in batch])
        prop_labels = torch.cuda.LongTensor([inst[7] for inst in batch])
        prop_scores = torch.cuda.FloatTensor([inst[8] for inst in batch])

        return input_ids_news, attention_masks_news, input_ids_prop, attention_masks_prop, input_ids_reason, attention_masks_reason, news_labels, prop_labels, prop_scores


