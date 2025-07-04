import datetime
import random
import torch
import numpy as np
from model import *
from parameters import *
from data import *

def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))

def seed_everything(
    seed_value: int
) -> None:
    random.seed(seed_value) 
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
    if torch.backends.cudnn.is_available:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

import os
import sys
import yaml
from torch.utils.data import DataLoader
from pprint import pprint
import torch
import torch.nn as nn
import numpy as np
import time

print(f"GPU: {torch.cuda.get_device_name(0)}")
_, total = torch.cuda.mem_get_info(device=0)
print(f"GPU memory: {total / 1024**3:.2f}GB")

with open("config.yaml", "r") as file_obj:
    config = yaml.safe_load(file_obj)
print()
pprint(config)
if config["data_path"] is None:
    config["data_path"] = os.environ["TMPDIR"]
    print("data_path:", config["data_path"])
print()

seed_everything(config["seed"])

all_inputs, all_outputs = [], []
for x in ["/kaggle/input/open-wfi-1/openfwi_float16_1", "/kaggle/input/open-wfi-2/openfwi_float16_2"]:
    all_inputs1, all_outputs1 = get_train_files(x)
    all_inputs.extend(all_inputs1)
    all_outputs.extend(all_outputs1)
print("Total number of input/output files:", len(all_inputs))

valid_inputs = [all_inputs[i] for i in range(0, len(all_inputs), config["valid_frac"])]
train_inputs = [f for f in all_inputs if not f in valid_inputs]
if config["train_frac"] > 1:
    train_inputs = [train_inputs[i] for i in range(0, len(train_inputs), config["train_frac"])]

print("Number of train files:", len(train_inputs))
print("Number of valid files:", len(valid_inputs))
print()

train_outputs = inputs_files_to_output_files(train_inputs)
valid_outputs = inputs_files_to_output_files(valid_inputs)

dstrain = SeismicDataset(train_inputs, train_outputs)
dltrain = DataLoader(
    dstrain,
    batch_size=config["batch_size"],
    shuffle=True,
    pin_memory=False,
    drop_last=True,
    num_workers=4,
    persistent_workers=True,
)

dsvalid = SeismicDataset(valid_inputs, valid_outputs)
dlvalid = DataLoader(
    dsvalid,
    batch_size=4*config["batch_size"],
    shuffle=False,
    pin_memory=False,
    drop_last=False,
    num_workers=4,
    persistent_workers=True,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNet(**config["model"]["unet_params"]).to(device)

if config["read_weights"] is not None:
    print("Reading weights from:", config["read_weights"])
    model.load_state_dict(torch.load(config["read_weights"], weights_only=True))

criterion = nn.L1Loss()
optimizer = torch.optim.AdamW(model.parameters(), **config["optimizer"])  # hparams
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', **config["scheduler"]["params"])

best_val_loss = 10000.0
epochs_wo_improvement = 0
t0 = time.time()  # Measure staring time

for epoch in range(1, config["max_epochs"] + 1):

    # Train
    model.train()
    train_losses = []
    for step, (inputs, targets) in enumerate(dltrain):

        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()

        with torch.autocast(device_type="cuda"):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        if step % config["print_freq"] == config["print_freq"] - 1 or step == len(dltrain) - 1:
            trn_loss = np.mean(train_losses)
            t1 = format_time(time.time() - t0)
            free, total = torch.cuda.mem_get_info(device=0)
            mem_used = (total - free) / 1024**3
            lr = optimizer.param_groups[-1]['lr']
            print(
                f"Epoch: {epoch:02d}  Step {step+1}/{len(dltrain)}  Trn Loss: {trn_loss:.2f}  LR: {lr:.2e}  GPU Usage: {mem_used:.2f}GB  Elapsed Time: {t1}",
                flush=True,
            )

    # Valid
    model.eval()
    valid_losses = []
    for inputs, targets in dlvalid:
        inputs = inputs.to(device)
        targets = targets.to(device)

        with torch.inference_mode():
            with torch.autocast(device_type="cuda"):
                outputs = model(inputs)

        loss = criterion(outputs, targets)

        valid_losses.append(loss.item())

    t1 = format_time(time.time() - t0)
    trn_loss = np.mean(train_losses)
    val_loss = np.mean(valid_losses)

    free, total = torch.cuda.mem_get_info(device=0)
    mem_used = (total - free) / 1024**3

    print(
        f"\nEpoch: {epoch:02d}  Trn Loss: {trn_loss:.2f}  Val Loss: {val_loss:.2f}  GPU Usage: {mem_used:.2f}GB  Elapsed Time: {t1}",
        flush=True,
    )
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_wo_improvement = 0
        torch.save(model.state_dict(), "best_model.pth")
        print(f"\nNew best val_loss: {val_loss:.2f}\n", flush=True)
    else:
        epochs_wo_improvement += 1
        print(f"\nEpochs without improvement: {epochs_wo_improvement}\n", flush=True)

    if epochs_wo_improvement == config["es_epochs"]:
        break

    scheduler.step(val_loss)
