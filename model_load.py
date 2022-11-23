import torch
import torch.nn as nn
from pytorch_pretrained import BertModel, BertTokenizer
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import models.bert as model_bert
import transformers
from transformers import BertTokenizerFast


# def create_data_loader(df, tokenizer, max_len, batch_size):
  

pred_data = pd.read_csv("./predict_data/comments.csv")
pred_data = pred_data.dropna()

class Model(nn.Module):
  def __init__(self):
    super(Model, self).__init__()
    self.bert = BertModel.from_pretrained('./bert_pretrain')
    for param in self.bert.parameters():
      param.requires_grad = True
    self.fc = nn.Linear(768, 2)
    self.num_epochs = 3
    self.batch_size = 128
    self.pad_size = 32
    self.learning_rate = 5e-5
    self.tokenizer = BertTokenizer.from_pretrained('./bert_pretrain')

  def forward(self, x):
      context = x[0]  # 输入的句子
      mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
      _, pooled = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
      # _, pooled = self.bert(input_ids, attention_mask=mask, output_all_encoded_layers=False)
      out = self.fc(pooled)
      return self.out(out)
    
bertmodel = Model()
bertmodel = torch.load('model.pth')
bertmodel.eval()

# for param in model.parameters():
#   print(param)

# one single sentence test
tokenizer = BertTokenizerFast.from_pretrained('./bert_pretrain')
sample_txt = "你是个傻逼吗？"
# tokens = tokenizer.tokenize(sample_txt)

# token_ids = tokenizer.convert_tokens_to_ids(tokens)
# print(f' Sentence: {sample_txt}')
# print(f'   Tokens: {tokens}')
# print(f'Token IDs: {token_ids}')

encoding = tokenizer.encode_plus(
  sample_txt,
  max_length = 32,
  add_special_tokens = True,
  return_token_type_ids = False,
  pad_to_max_length = True,
  return_attention_mask = True,
  return_tensors = 'pt',
)

encoding.keys()
# dict_keys(['input_ids', 'attention_mask'])
# print(encoding['input_ids'][0])
# print(tokenizer.convert_ids_to_tokens(encoding['input_ids'][0]))

# token_lens = []
# for txt in pred_data.text:
#   tokens = tokenizer.encode(txt, max_length=512)
#   token_lens.append(len(tokens))

# sns.distplot(token_lens)
# plt.xlim([0, 256]);
# plt.xlabel('Token count');

class Collect_Dataset(Dataset):
  def __init__(self, texts, tokenizer, max_len=32):
    self.texts = texts
    # self.targets = targets
    self.tokenizer = tokenizer
    self.max_len = max_len
  def __len__(self):
    return len(self.texts)
  def __getitem__(self, item):
    text = str(self.texts[item])
    # target = self.targets[item]
    encoding = self.tokenizer.encode_plus(
      text,
      add_special_tokens=True,
      max_length=self.max_len,
      return_token_type_ids=False,
      pad_to_max_length=True,
      return_attention_mask=True,
      return_tensors='pt',
    )
    return {
      'text': text,
      'input_ids': encoding['input_ids'].flatten(),
      'attention_mask': encoding['attention_mask'].flatten()
      # 'targets': torch.tensor(target, dtype=torch.long)
    }
    
def data_loader(df, tokenizer, max_len, batch_size):
  dataset = Collect_Dataset(
    texts=df.to_numpy(),
    tokenizer=tokenizer,
    max_len=max_len
  )
  
  return DataLoader(
    dataset,
    batch_size=batch_size
  )

BATCH_SIZE = 128

pred_data_loader = data_loader(pred_data, tokenizer, 32, BATCH_SIZE)

# data = next(iter(pred_data_loader))
# print(data['input_ids'].shape)
# print(data['attention_mask'].shape)

# max_seq_len = 32
# tokenize and encode sequences in the predict set
# tokens_pred = tokenizer.batch_encode_plus(
#   data.tolist(),
#   max_length = max_seq_len,
#   padding='max_length',
#   truncation=True,
#   return_token_type_ids=False
# )
# print(tokens_pred)

# convert integer sequences to tensors
# pred_seq = torch.tensor(tokens_pred['input_ids'])
# pred_mask = torch.tensor(tokens_pred['attention_mask'])

def get_predictions(model, data_loader):
  model = model.eval()
  texts = []
  predictions = []
  prediction_probs = []
  real_values = []
  with torch.no_grad():
    for d in data_loader:
      texts = d["text"]
      input_ids = d["input_ids"].to("cpu")
      attention_mask = d["attention_mask"].to("cpu")
      # targets = d["targets"].to("cpu")
      outputs = model(
        d
        # attention_mask=attention_mask
      )
      _, preds = torch.max(outputs, dim=1)
      texts.extend(texts)
      predictions.extend(preds)
      prediction_probs.extend(outputs)
      # real_values.extend(targets)
  predictions = torch.stack(predictions).cpu()
  prediction_probs = torch.stack(prediction_probs).cpu()
  real_values = torch.stack(real_values).cpu()
  return texts, predictions, prediction_probs, real_values

y_review_texts, y_pred, y_pred_probs, y_test = get_predictions(
  bertmodel,
  pred_data_loader
)

print(y_pred)

# print(classification_report(y_test, y_pred, target_names=class_names))

