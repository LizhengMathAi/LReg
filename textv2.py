from typing import Tuple, List, Union, Optional, Callable
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchtext.datasets import AG_NEWS
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data.dataset import random_split
from torchtext.data.functional import to_map_style_dataset
import time

from utilsv2 import LagrangeEmbedding


# Time efficiency test of LagrangeEmbedding-based network: 
# GPU: RTX 2080Ti
# Setting BATCH_SIZE = 1, the the LagrangeEmbedding initialization initialization takes 1.89 seconds,
# and each epoch training takes 74.99 seconds.


# Time efficiency test of the original network (https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html): 
# GPU: RTX 2080Ti
# Setting BATCH_SIZE = 1, each epoch training takes 91.24 seconds.


# ---------------------------------- data pipeline ----------------------------------  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = get_tokenizer("basic_english")


def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)


train_iter = AG_NEWS(split="train")
vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])
text_pipeline = lambda x: vocab(tokenizer(x))
label_pipeline = lambda x: int(x) - 1
num_class = len(set([label for (label, text) in train_iter]))
VOCAB_SIZE = len(vocab)

# Count the frequency of every word.
COUNT = torch.zeros(VOCAB_SIZE, num_class, dtype=torch.int64)
for (label, text) in train_iter:
    vocab_indices = vocab(tokenizer(text))
    label_index = int(label) - 1
    COUNT[vocab_indices, label_index] += 1
COUNT = COUNT.to(device)
MASK = (torch.sum((COUNT != 0).to(torch.int64), dim=1) == 1).to(torch.int64)


def collate_batch(batch):
    text_list, offsets, label_list = [], [0], []
    for _label, _text in batch:
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        text_list.append(processed_text)
        offsets.append(processed_text.size(0))
        label_list.append(label_pipeline(_label))
    text_list = torch.cat(text_list)
    offsets = torch.tensor(offsets).cumsum(dim=0)
    label_list = torch.tensor(label_list, dtype=torch.int64)
    return text_list.to(device), offsets.to(device), label_list.to(device)


# ---------------------------------- Model ----------------------------------  
class TextClassificationModel(nn.Module):
    def pre_proc(self, raw_data, **kwargs):
        count = COUNT if COUNT.device == raw_data.device else COUNT.to(raw_data.device)

        data = count[raw_data, :] / (torch.relu(torch.sum(count[raw_data, :], dim=1, keepdim=True) - 1) + 1)
        return data
    
    def post_proc(self, x, *args):
        offsets, = args
        cums = torch.cumsum(x, dim=0)  # [N, dof] -> [N, dof]
        x = torch.cat([cums[offsets[1]-1][None, :], cums[offsets[2:]-1] - cums[offsets[1:-1]-1], ], dim=0)  # [N, dof], [B+1, ] -> [B, dof]
        x = x / (torch.relu(offsets[1:] - offsets[:-1] - 1) + 1)[:, None]  # [B, dof] -> [B, dof]
        return x

    def __init__(self, data_distribution: Tuple[torch.Tensor, torch.Tensor], dof: int, n_cls: int): 
        super(TextClassificationModel, self).__init__()

        self.backbone = LagrangeEmbedding(data_distribution, dof, pre_proc=self.pre_proc)

        self.fc = nn.Linear(dof, n_cls, bias=False)

    def forward(self, text, offsets):
        x = self.pre_proc(text)  # [N, ] -> [N, ?]
        x = self.backbone(x)  # [N, ?] -> [N, dof]
        x = self.post_proc(x, offsets)  # [N, dof] -> [B, dof]
        return self.fc(x)


# ---------------------------------- Training ----------------------------------  
def train(dataloader):
    model.train()
    total_acc, total_count = 0, 0
    log_interval = 2500
    start_time = time.time()

    for idx, (text, offsets, label) in enumerate(dataloader):
        optimizer.zero_grad()
        predicted_label = model(text, offsets)
        loss = criterion(predicted_label, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        total_acc += (predicted_label.argmax(1) == label).sum().item()
        total_count += label.size(0)
        if idx % log_interval == 0 and idx > 0:
            elapsed = time.time() - start_time
            print(
                "| epoch {:3d} | {:5d}/{:5d} batches "
                "| accuracy {:8.3f}".format(epoch, idx, len(dataloader), total_acc / total_count)
            )
            total_acc, total_count = 0, 0
            start_time = time.time()


def evaluate(dataloader):
    model.eval()
    total_acc, total_count = 0, 0

    with torch.no_grad():
        for idx, (text, offsets, label) in enumerate(dataloader):
            predicted_label = model(text, offsets)
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
    return total_acc / total_count


# Hyperparameters
EPOCHS = 5  # epoch
LR = 5  # learning rate
BATCH_SIZE = 6  # batch size for training

train_iter, test_iter = AG_NEWS()
train_dataset = to_map_style_dataset(train_iter)
test_dataset = to_map_style_dataset(test_iter)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)

dof = 64
model = TextClassificationModel([torch.arange(VOCAB_SIZE, device=device), torch.sum(MASK[:, None] * COUNT, dim=1)], dof, n_cls=num_class).to(device)
# Count the number of trainable parameters
num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of Trainable Parameters: {num_trainable_params}")

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 2.0, gamma=0.1)

for epoch in range(1, EPOCHS + 1):
    epoch_start_time = time.time()
    train(train_dataloader)
    accu_val = evaluate(test_dataloader)
    scheduler.step()
    print("-" * 58)
    print(
        "| end of epoch {:3d} | time: {:5.2f}s | "
        "test accuracy {:8.3f} ".format(
            epoch, time.time() - epoch_start_time, accu_val
        )
    )
    print("-" * 58)

ag_news_label = {1: "World", 2: "Sports", 3: "Business", 4: "Sci/Tec"}

def predict(text, text_pipeline):
    with torch.no_grad():
        text = torch.tensor(text_pipeline(text))
        output = model(text, torch.tensor([0, text.shape[0]]))
        return output.argmax(1).item() + 1


ex_text_str = "MEMPHIS, Tenn. – Four days ago, Jon Rahm was \
    enduring the season’s worst weather conditions on Sunday at The \
    Open on his way to a closing 75 at Royal Portrush, which \
    considering the wind and the rain was a respectable showing. \
    Thursday’s first round at the WGC-FedEx St. Jude Invitational \
    was another story. With temperatures in the mid-80s and hardly any \
    wind, the Spaniard was 13 strokes better in a flawless round. \
    Thanks to his best putting performance on the PGA Tour, Rahm \
    finished with an 8-under 62 for a three-stroke lead, which \
    was even more impressive considering he’d never played the \
    front nine at TPC Southwind."

model = model.to("cpu")

print("This is a %s news" % ag_news_label[predict(ex_text_str, text_pipeline)])
