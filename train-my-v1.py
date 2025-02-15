"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
from contextlib import nullcontext

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer

from model import ModelM
from dataloader import ProblemAnswerDataset, CollateFn 

# ----------------------------------------------------------------------------- 
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
checkpoint_name = f"ckpt_{time.strftime('%Y%m%d_%H%M%S')}.pt"
eval_interval = 10 # better than 5
log_interval = 1
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = False # if True, always save a checkpoint after each eval
init_from = ('gpt2', 'gpt2', 'gpt2') # 'scratch' or 'resume' or 'gpt2*' for encoder/think_model/decoder
num_iterations = 5
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' + str(time.time()) # 'run' + str(time.time())
# data
gradient_accumulation_steps = 8 # used to simulate larger batch sizes
batch_size = 8 # if gradient_accumulation_steps > 1, this is the micro-batch size
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.2 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 100 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 10 # how many steps to warm up for
lr_decay_iters = 100 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda:4' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = False # use PyTorch 2.0 to compile the model to be faster
# ----------------------------------------------------------------------------- 
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# ----------------------------------------------------------------------------- 

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

def compute_loss(model, batch):
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    targets = [t.to(device) for t in batch['targets']]
    loss_mask = [t.to(device) for t in batch['loss_masks']]

    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    neuron_matrixes = model(input_ids, attention_mask=attention_mask)
    loss_all = 0
    no_mask_counts = 0
    for i in range(len(neuron_matrixes)):
        logits = model.decode(neuron_matrixes[i], targets[i])
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = targets[i][:, 1:]
        shift_mask = loss_mask[i][:, 1:].contiguous()
        loss = loss_fn(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))
        loss = loss.view(shift_labels.size()) * shift_mask
        loss_all += loss.sum() 
        no_mask_counts += shift_mask.sum()
    return loss_all / no_mask_counts

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
if init_from[0] == 'resume':
    print(f"Resuming training from {out_dir}")
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    encoder_name, think_name, decoder_name = checkpoint['encoder_name'], checkpoint['think_name'], checkpoint['decoder_name']
    model = ModelM(tokenizer, init_from=(encoder_name, think_name, decoder_name), neuron_dim=100, num_iterations=num_iterations,batch_size = batch_size)
    model.load_state_dict(checkpoint['model'])
elif init_from[0].startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: encoder-{init_from[0]}, thinker-{init_from[1]}, decoder-{init_from[2]}")
    model = ModelM(tokenizer, init_from=init_from, neuron_dim=100, num_iterations=num_iterations,batch_size = batch_size)

model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(beta1, beta2), weight_decay=weight_decay)
if init_from[0] == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

train_dataset = ProblemAnswerDataset('/data/jyliu/transformer_project/nanoGPT-master/data/gsm8k/train.jsonl', tokenizer)
val_dataset = ProblemAnswerDataset('/data/jyliu/transformer_project/nanoGPT-master/data/gsm8k/test.jsonl', tokenizer)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=CollateFn(tokenizer.eos_token_id))
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=CollateFn(tokenizer.eos_token_id))

best_val_loss = 1e9

# training loop
t0 = time.time()
local_iter_num = 0
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0
for iter_num in range(max_iters):
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        model.eval()
        with torch.no_grad():
            train_loss = sum(compute_loss(model, batch).item() for batch in train_loader) / len(train_loader)
            val_loss = sum(compute_loss(model, batch).item() for batch in val_loader) / len(val_loader)
        print(f"step {iter_num}: val loss {val_loss:.4f}")
        if wandb_log:
            # wandb.log({"iter": iter_num, "train/loss": train_loss, "val/loss": val_loss, "lr": lr, "mfu": running_mfu * 100})
            wandb.log({"iter": iter_num, "train/loss": train_loss, "val/loss": val_loss, "lr": lr})
        if val_loss < best_val_loss or always_save_checkpoint:
            best_val_loss = val_loss
            checkpoint = {
                'model': raw_model.state_dict(),
                'encoder_name': init_from[0], 
                'think_name': init_from[1], 
                'decoder_name': init_from[2],
                'optimizer': optimizer.state_dict(),
                'iter_num': iter_num,
                'best_val_loss': best_val_loss,
                'config': config,
            }
            print(f"saving checkpoint to {out_dir}")
            torch.save(checkpoint, os.path.join(out_dir, checkpoint_name))

    if eval_only:
        break
    model.train()
    
    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    micro_step = 0  # track accumulation steps
    for batch in train_loader:
        if micro_step == 0:
            optimizer.zero_grad(set_to_none=True)

        if ddp and micro_step < gradient_accumulation_steps - 1:
            with model.no_sync():
                loss = compute_loss(model, batch) / gradient_accumulation_steps
                scaler.scale(loss).backward()
        else:
            loss = compute_loss(model, batch) / gradient_accumulation_steps
            scaler.scale(loss).backward()

        micro_step += 1

        if micro_step == gradient_accumulation_steps:
            if grad_clip != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            scaler.step(optimizer)
            scaler.update()

            micro_step = 0

    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        lossf = loss.item() * gradient_accumulation_steps
        # if local_iter_num >= 5:
        #     mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
        #     running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
        # print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms")
    local_iter_num += 1

if ddp:
    destroy_process_group()
