import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoTokenizer


def read_data():
    df = pd.read_csv("train.csv", encoding="latin-1")
    df2 = pd.read_csv("test.csv", encoding="latin-1")
    text = df['text']
    text2 = df2['text']
    label = df2['sentiment']
    sentiment = df['sentiment']
    t1 = []
    l1 = []
    t2 = []
    l2 = []
    for i in range(len(text2)):
        if label[i] == 'positive':
            l2.append(1)
            t2.append(text2[i])
        elif label[i] == 'negative':
            l2.append(0)
            t2.append(text2[i])
    for i in range(len(text)):
        if sentiment[i] == 'positive':
            l1.append(1)
            t1.append(text[i])
        elif sentiment[i] == 'negative':
            l1.append(0)
            t1.append(text[i])
    return t1, l1, t2, l2


class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        return {'input_ids': encoding['input_ids'][0], 'attention_mask': encoding['attention_mask'][0], 'labels': label}


# define the training data
def CREATE_DS():
    train_texts, train_labels, eval_texts, eval_labels = read_data()
    # define the evaluation data

    # initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    # create the training and evaluation datasets
    train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer)
    eval_dataset = TextClassificationDataset(eval_texts, eval_labels, tokenizer)
    return train_dataset, eval_dataset
