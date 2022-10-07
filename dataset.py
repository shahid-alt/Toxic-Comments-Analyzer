from utils import args
import torch

class CustomDataset:
    def __init__(self,data,tokenizer):
        self.data = data
        self.text = self.data['text'].values.tolist()
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.text)
    
    def __getitem__(self,idx):
        text = self.text[idx]
        encoded_text = self.tokenizer.encode_plus(
            text,
            None,
            max_length = args().max_len,
            truncation=True,
            padding = 'max_length',
            add_special_tokens=True, 
        )
    
        input_ids = encoded_text['input_ids']
        attention_mask = encoded_text['attention_mask']
        
        return {
            'ids': torch.tensor(input_ids,dtype=torch.long),
            'mask': torch.tensor(attention_mask,dtype=torch.long)
        }
    