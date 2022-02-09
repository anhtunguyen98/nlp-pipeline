import torch
import pandas as pd


class NewDataset(torch.utils.data.Dataset):
    def __init__(self,max_length, tokenizer, file_path,label_set):
        label_set = {x:i for i,x in enumerate(label_set)}
        self.data = pd.read_csv(file_path)
        self.tokenizer = tokenizer
        texts = self.data['sentence'].tolist()
        labels = self.data['label'].tolist()
        self.labels = [label_set[w] for w in labels]
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor([self.labels[idx]])
        return item

    def __len__(self):
        return len(self.data)