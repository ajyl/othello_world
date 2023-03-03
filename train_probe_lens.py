import os

# set up logging
import logging

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

# make deterministic
from mingpt.utils import set_seed

set_seed(42)

import time
import multiprocessing
import pickle
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
import argparse
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from data import get_othello
from data.othello import permit, start_hands, OthelloBoardState
from mingpt.dataset import CharDataset
from mingpt.model import GPT, GPTConfig, GPTforProbing
from mingpt.probe_trainer import Trainer, TrainerConfig
from mingpt.probe_model import (
    BatteryProbeClassification,
    BatteryProbeClassificationTwoLayer,
)

parser = argparse.ArgumentParser(description="Train classification network")

parser.add_argument("--epo", default=16, type=int)
parser.add_argument("--mid_dim", default=128, type=int)
parser.add_argument("--random", dest="random", action="store_true")
parser.add_argument("--championship", dest="championship", action="store_true")
parser.add_argument("--exp", default="state", type=str)

args, _ = parser.parse_known_args()

folder_name = f"battery_othello/{args.exp}"

if args.random:
    folder_name = folder_name + "_random"
if args.championship:
    folder_name = folder_name + "_championship"

print(f"Running experiment for {folder_name}")
#othello = get_othello(data_root="data/othello_championship")
othello = get_othello(ood_num=-1)

train_dataset = CharDataset(othello)
if torch.cuda.is_available():
    device = torch.cuda.current_device()
    model = model.to(device)


PROP_BATCH_SIZE=8192
loader = DataLoader(
    train_dataset, shuffle=False, pin_memory=True, batch_size=PROP_BATCH_SIZE
)

def get_property(x):
    """
    Parallelize generating states.
    """
    tbf = [train_dataset.itos[_] for _ in x.tolist()]
    a = OthelloBoardState()
    properties = a.get_gt(tbf, "get_" + args.exp)  # [block_size, ]
    return properties
 

act_container = []
num_proc = multiprocessing.cpu_count()
p = multiprocessing.Pool(num_proc)

for idx, (x, y, z) in tqdm(enumerate(loader), total=len(loader)):
    output_filepath = os.path.join(
        "/scratch/mihalcea_root/mihalcea98/ajyl",
        f"probe_data/properties_{idx}.pkl"
    )
    if os.path.isfile(output_filepath):
        continue

    property_container = []
    for can in tqdm(p.imap(get_property, x), total=PROP_BATCH_SIZE):
        property_container.extend(can)

    with open(output_filepath, "wb") as file_p:
        pickle.dump(property_container, file_p)

age_container = []
for x, y, z in tqdm(loader, total=len(loader)):
    tbf = [train_dataset.itos[_] for _ in x.tolist()[0]]
    valid_until = tbf.index(-100) if -100 in tbf else 999
    a = OthelloBoardState()
    ages = a.get_gt(tbf[:valid_until], "get_age")  # [block_size, ]
    age_container.extend(ages)

if args.exp == "state":
    probe_class = 3

probe = BatteryProbeClassification(device, probe_class=probe_class, num_task=64)


class ProbingDataset(Dataset):
    def __init__(self, act, y, age):
        assert len(act) == len(y)
        assert len(act) == len(age)
        print(f"{len(act)} pairs loaded...")
        self.act = act
        self.y = y
        self.age = age
        print(
            np.sum(np.array(y) == 0), np.sum(np.array(y) == 1), np.sum(np.array(y) == 2)
        )

        long_age = []
        for a in age:
            long_age.extend(a)
        long_age = np.array(long_age)
        counts = [np.count_nonzero(long_age == i) for i in range(60)]
        del long_age
        print(counts)

    def __len__(
        self,
    ):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            self.act[idx],
            torch.tensor(self.y[idx]).to(torch.long),
            torch.tensor(self.age[idx]).to(torch.long),
        )


probing_dataset = ProbingDataset(act_container, property_container, age_container)
train_size = int(0.8 * len(probing_dataset))
test_size = len(probing_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(
    probing_dataset, [train_size, test_size]
)
sampler = None
train_loader = DataLoader(
    train_dataset,
    shuffle=False,
    sampler=sampler,
    pin_memory=True,
    batch_size=128,
    num_workers=1,
)
test_loader = DataLoader(
    test_dataset, shuffle=True, pin_memory=True, batch_size=128, num_workers=1
)

max_epochs = args.epo
t_start = time.strftime("_%Y%m%d_%H%M%S")
tconf = TrainerConfig(
    max_epochs=max_epochs,
    batch_size=1024,
    learning_rate=1e-3,
    betas=(0.9, 0.999),
    lr_decay=True,
    warmup_tokens=len(train_dataset) * 5,
    final_tokens=len(train_dataset) * max_epochs,
    num_workers=4,
    weight_decay=0.0,
    ckpt_path=os.path.join("./ckpts/", folder_name, f"layer{args.layer}"),
)
trainer = Trainer(probe, train_dataset, test_dataset, tconf)
trainer.train(prt=True)
trainer.save_traces()
trainer.save_checkpoint()
