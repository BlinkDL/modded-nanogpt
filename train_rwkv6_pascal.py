import os
import sys
with open(sys.argv[0]) as f:
    code = f.read() # read the code of this file ASAP, for logging
import uuid
import glob
import time, datetime, random
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
import torch._inductor.config as config
from torch.nn.parallel import DistributedDataParallel as DDP
import wandb
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--headsz', type=int, default=64) # increase to 96/128/192 for better loss (slow in inefficient implementation, fast after optimization)
parser.add_argument('--muon_lr', type=float, default=0.00036)
parser.add_argument('--adam_lr', type=float, default=0.0020) # adam lr for misc weights (lora, time, etc.)
parser.add_argument('--emb_scale', type=float, default=1.5) # rescale embedding to boost its adam lr
parser.add_argument('--device_bsz', type=int, default=64)
parser.add_argument('--bsz', type=int, default=8*64)
parser.add_argument('--random_seed', type=int, default=-1)
cmd_args = parser.parse_args()

if cmd_args.random_seed != -1:
    random.seed(cmd_args.random_seed)
    np.random.seed(cmd_args.random_seed)
    torch.manual_seed(cmd_args.random_seed)
    torch.cuda.manual_seed_all(cmd_args.random_seed)

'''
Based on the GPT code in "101424_ModernArch" folder (please diff it to see the changes)

Changes:
*) CausalSelfAttention => RWKV-6
*) FFN 4x => 3.5x (to keep params count)
*) rms_norm => LayerNorm
*) use Adam for misc weights (lora, time, etc.)

Note:
Currently runs at 65% GPT speed due to:
*) Inefficient RWKV-6 kernel.
*) The loras can be further fused.

I think we can get it to 90% GPT speed @ ctxlen 1024 (can be faster than GPT @ ctxlen 2048) after more work.
'''

# -----------------------------------------------------------------------------
# RWKV-6 kernel

HEAD_SIZE = cmd_args.headsz
sequence_length = 1024

if True:       
    from torch.utils.cpp_extension import load
    load(name="wkv6", sources=["rwkv_cuda_pascal/wkv6_op.cpp", f"rwkv_cuda_pascal/wkv6_cuda.cu"], is_python_module=False,
                    verbose=True, extra_cuda_cflags=["-res-usage", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization", f"-D_N_={HEAD_SIZE}", f"-D_T_={sequence_length}"])
        
    class WKV_6(torch.autograd.Function):
        @staticmethod
        def forward(ctx, r, k, v, w, u):
            with torch.no_grad():
                B, T, C = r.size()
                H = C // HEAD_SIZE
                assert C % HEAD_SIZE == 0
                assert all(i.dtype == torch.float for i in [r,k,v,w,u])
                ctx.B = B
                ctx.T = T
                ctx.C = C
                ctx.H = H
                r,k,v,w,u = [i.contiguous() for i in [r,k,v,w,u]]
                ctx.save_for_backward(r, k, v, w, u)
                y = torch.empty((B, T, C), device=r.device, dtype=torch.float, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                torch.ops.wkv6.forward(B, T, C, H, r, k, v, w, u, y)
                return y

        @staticmethod
        def backward(ctx, gy):
            with torch.no_grad():
                assert gy.dtype == torch.float
                B = ctx.B
                T = ctx.T
                C = ctx.C
                H = ctx.H
                gy = gy.contiguous()
                r, k, v, w, u = ctx.saved_tensors
                gr = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.float, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                gk = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.float, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                gv = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.float, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                gw = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.float, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                gu = torch.empty((B, C), device=gy.device, requires_grad=False, dtype=torch.float, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                torch.ops.wkv6.backward(B, T, C, H, r, k, v, w, u, gy, gr, gk, gv, gw, gu)
                gu = torch.sum(gu, 0).view(H, C//H)
                return (gr, gk, gv, gw, gu)

    def RUN_CUDA_RWKV6(r, k, v, w, u):
        return WKV_6.apply(r, k, v, w, u)

# -----------------------------------------------------------------------------
# Muon optimizer

def zeropower_via_svd(G, steps=None):
    U, S, V = G.svd()
    return U @ V.T

@torch.compile
def zeropower_via_newtonschulz5(G, steps=10, eps=1e-7):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' \\sim Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G
    X /= (X.norm() + eps) # ensure top singular value <= 1
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = A @ X
        X = a * X + b * B + c * A @ B
    if G.size(0) > G.size(1):
        X = X.T
    return X

zeropower_backends = dict(svd=zeropower_via_svd, newtonschulz5=zeropower_via_newtonschulz5)

class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in float on the GPU.

    Some warnings:
    - This optimizer assumes that all parameters passed in are 2D.
    - It should not be used for the embedding layer, the final fully connected layer, or any {0,1}-D
    parameters; those should all be optimized by a standard method (e.g., AdamW).
    - To use it with 4D convolutional filters, it works well to just flatten their last 3 dimensions.
    - We believe it is unlikely to work well for training with small batch size.
    - We believe it may not work well for finetuning pretrained models, but we haven't tested this.
    - We have not yet tried this optimizer for training scenarios larger than NanoGPT (124M).

    Arguments:
        lr: The learning rate used by the internal SGD.
        momentum: The momentum used by the internal SGD.
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        backend: The chosen backend for the orthogonalization step. (recommended: 'newtonschulz5')
        backend_steps: The number of iteration steps to use in the backend, if it is iterative.
    """
    def __init__(self, params, lr=3e-4, momentum=0.95, nesterov=True, backend='newtonschulz5', backend_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, backend=backend, backend_steps=backend_steps)
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            zeropower_backend = zeropower_backends[group['backend']]
            for p in group['params']:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g)
                if group['nesterov']:
                    g = g.add(buf, alpha=momentum)
                if g.size(0) == 3 * g.size(1): # split grouped QKV parameters
                    g = torch.cat([zeropower_backend(g1, steps=group['backend_steps']) for g1 in g.split(g.size(1))])
                    scale = g.size(1)**0.5
                else:
                    g = zeropower_backend(g, steps=group['backend_steps'])
                    scale = max(g.size(0), g.size(1))**0.5 # scale to have update.square().mean() == 1
                p.data.add_(g, alpha=-lr * scale)

# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the RWKV-6 model (optimized for 123M params performance)

class RWKV6(nn.Module):

    def __init__(self, config, layer_id):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0

        with torch.no_grad():            
            ratio_0_to_1 = layer_id / (config.n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / config.n_layer)  # 1 to ~0            
            ddd = torch.ones(1, 1, self.n_embd)
            for i in range(self.n_embd):
                ddd[0, 0, i] = i / self.n_embd
            
            # initialization comes from fitting my RWKV-6 7B runs
            self.time_maa_x = nn.Parameter(1.0 - torch.pow(ddd, 0.6 * ratio_1_to_almost0 ** 0.9))
            self.time_maa_w = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.time_maa_k = nn.Parameter(1.0 - (torch.pow(ddd, 0.9 * ratio_1_to_almost0) + 0.4 * ratio_0_to_1))
            self.time_maa_v = nn.Parameter(1.0 - (torch.pow(ddd, 0.4 * ratio_1_to_almost0) + 0.6 * ratio_0_to_1))
            self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))
            self.time_maa_g = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))

            decay_speed = torch.ones(self.n_embd)
            for n in range(self.n_embd):
                decay_speed[n] = -6 + 5.5 * (n / (self.n_embd - 1)) ** (0.85 + 1.0 * ratio_0_to_1 ** 0.5)
            self.time_decay = nn.Parameter(decay_speed.reshape(1,1,self.n_embd))

            tmp = torch.zeros(self.n_embd)
            for n in range(self.n_embd):
                zigzag = ((n + 1) % 3 - 1) * 0.1
                tmp[n] = ratio_0_to_1 * (2.5 - (n / (self.n_embd - 1))) + zigzag

            self.time_faaaa = nn.Parameter(tmp.reshape(self.n_head, self.head_dim))

            D_MIX_LORA = 32
            self.time_maa_w1 = nn.Parameter(torch.zeros(self.n_embd, D_MIX_LORA*5))
            self.time_maa_w2 = nn.Parameter(torch.zeros(5, D_MIX_LORA, self.n_embd).uniform_(-0.01, 0.01))
            
            D_DECAY_LORA = 64
            self.time_decay_w1 = nn.Parameter(torch.zeros(self.n_embd, D_DECAY_LORA))
            self.time_decay_w2 = nn.Parameter(torch.zeros(D_DECAY_LORA, self.n_embd).uniform_(-0.01, 0.01))

            self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

            self.receptance = nn.Linear(self.n_embd, self.n_embd, bias=False)
            self.key = nn.Linear(self.n_embd, self.n_embd, bias=False)
            self.value = nn.Linear(self.n_embd, self.n_embd, bias=False)
            self.output = nn.Linear(self.n_embd, self.n_embd, bias=False)

            self.receptance.weight.data.uniform_(-0.5/(self.n_embd**0.5), 0.5/(self.n_embd**0.5))
            self.key.weight.data.uniform_(-0.05/(self.n_embd**0.5), 0.05/(self.n_embd**0.5))
            self.value.weight.data.uniform_(-0.5/(self.n_embd**0.5), 0.5/(self.n_embd**0.5))
            self.output.weight.data.zero_()

            D_GATE_LORA = 160 # using lora gate to save params
            self.gate_w1 = nn.Parameter(torch.zeros(self.n_embd, D_GATE_LORA).uniform_(-0.01, 0.01))
            self.gate_w2 = nn.Parameter(torch.zeros(D_GATE_LORA, self.n_embd).uniform_(-0.01, 0.01))

            self.ln_x = nn.GroupNorm(self.n_head, self.n_embd, eps=64e-5)


    def forward(self, x):

        B, T, C = x.size()

        xx = self.time_shift(x) - x

        xxx = x + xx * self.time_maa_x
        xxx = torch.tanh(xxx @ self.time_maa_w1).view(B*T, 5, -1).transpose(0, 1)
        xxx = torch.bmm(xxx, self.time_maa_w2).view(5, B, T, -1)
        mw, mk, mv, mr, mg = xxx.unbind(dim=0)

        xw = x + xx * (self.time_maa_w + mw)
        xk = x + xx * (self.time_maa_k + mk)
        xv = x + xx * (self.time_maa_v + mv)
        xr = x + xx * (self.time_maa_r + mr)
        xg = x + xx * (self.time_maa_g + mg)

        r = self.receptance(xr)
        k = self.key(xk)
        v = self.value(xv)
        g = torch.tanh(xg @ self.gate_w1) @ self.gate_w2

        ww = torch.tanh(xw @ self.time_decay_w1) @ self.time_decay_w2
        w = self.time_decay + ww

        x = RUN_CUDA_RWKV6(r, (k * torch.clamp(w, max=0).exp()), v, w, u=self.time_faaaa)

        x = x.view(B * T, C)
        x = self.ln_x(x).view(B, T, C)

        x = self.output(x * g)

        return x


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 7 * config.n_embd // 2, bias=False)
        self.c_proj  = nn.Linear(7 * config.n_embd // 2, config.n_embd, bias=False)
        self.c_proj.weight.data.zero_() # zero init suggested by @Grad62304977

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square() # https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config, layer_id):
        super().__init__()
        self.attn = RWKV6(config, layer_id)
        self.mlp = MLP(config)

        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

# -----------------------------------------------------------------------------
# The main GPT-2 model

@dataclass
class GPTConfig:
    vocab_size : int = 50304
    n_layer : int = 12
    n_head : int = 768//HEAD_SIZE
    n_embd : int = 768

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            h = nn.ModuleList([Block(config, layer_id) for layer_id in range(config.n_layer)]),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        self.ln_out = nn.LayerNorm(config.n_embd)

    def forward(self, idx, targets=None, return_logits=True):

        # forward the GPT model itself
        x = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        x = x * cmd_args.emb_scale # rescale embedding
        for block in self.transformer.h:
            x = block(x)
        x = self.ln_out(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            logits = logits # use tf32/fp32 for logits
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            logits = logits # use tf32/fp32 for logits
            loss = None

        # there are performance reasons why not returning logits is prudent, if not needed
        if not return_logits:
            logits = None

        return logits, loss

# -----------------------------------------------------------------------------
# Our own simple Distributed Data Loader

def _peek_data_shard(filename):
    # only reads the header, returns header data
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256*4), dtype=np.int32)
    if header[0] != 20240520:
        print("ERROR: magic number mismatch in the data .bin file!")
        print("---> HINT: Are you passing in a correct file with --input_bin?")
        print("---> HINT: Dataset encoding changed recently, re-run data prepro or refer again to README")
        print("---> HINT: For example re-run: `python dev/data/tinyshakespeare.py`, then re-try")
        exit(1)
    assert header[1] == 1, "unsupported version"
    ntok = header[2] # number of tokens (claimed)
    return ntok # for now just return the number of tokens

def _load_data_shard(filename):
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256*4), dtype=np.int32)
        assert header[0] == 20240520, "magic number mismatch in the data .bin file"
        assert header[1] == 1, "unsupported version"
        ntok = header[2] # number of tokens (claimed)
        # the rest of it are tokens, stored as uint16
        tokens = np.frombuffer(f.read(), dtype=np.uint16)
    assert len(tokens) == ntok, "number of tokens read does not match header?"
    return tokens

class DistributedDataLoader:
    def __init__(self, filename_pattern, B, T, process_rank, num_processes):
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.B = B
        self.T = T

        # glob files that match the pattern
        self.files = sorted(glob.glob(filename_pattern))
        assert len(self.files) > 0, f"did not find any files that match the pattern {filename_pattern}"

        # load and validate all data shards, count number of tokens in total
        ntok_total = 0
        for fname in self.files:
            shard_ntok = _peek_data_shard(fname)
            assert shard_ntok >= num_processes * B * T + 1
            ntok_total += int(shard_ntok)
        self.ntok_total = ntok_total

        # kick things off
        self.reset()

    def reset(self):
        self.current_shard = 0
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def advance(self): # advance to next data shard
        self.current_shard = (self.current_shard + 1) % len(self.files)
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def next_batch(self):
        B = self.B
        T = self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        buf = torch.tensor(buf.astype(np.int32), dtype=torch.long)
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance current position and load next shard if necessary
        self.current_position += B * T * self.num_processes
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.advance()
        return x.cuda(), y.cuda()

# -----------------------------------------------------------------------------
# int main

@dataclass
class Hyperparameters:
    # data hyperparams
    input_bin : str = 'data/fineweb10B/fineweb_train_*.bin' # input .bin to train on
    input_val_bin : str = 'data/fineweb10B/fineweb_val_*.bin' # input .bin to eval validation loss on
    # optimization hyperparams
    batch_size : int = cmd_args.bsz # batch size, in sequences, across all devices
    device_batch_size : int = cmd_args.device_bsz # batch size, in sequences, per device
    sequence_length : int = sequence_length # sequence length, in tokens
    num_iterations : int = 5100 # number of iterations to run
    learning_rate : float = -1
    warmup_iters : int = 0
    warmdown_iters : int = 1450 # number of iterations of linear warmup/warmdown for triangular or trapezoidal schedule
    weight_decay : float = 0
    # evaluation and logging hyperparams
    val_loss_every : int = 125 # every how many steps to evaluate val loss? 0 for only at the end
    val_tokens : int = 10485760 # how many tokens of validation data? it's important to keep this fixed for consistent comparisons
    save_every : int = 0 # every how many steps to save the checkpoint? 0 for only at the end
args = Hyperparameters()

args.headsz = cmd_args.headsz
args.muon_lr = cmd_args.muon_lr
args.adam_lr = cmd_args.adam_lr
args.emb_scale = cmd_args.emb_scale

# set up DDP (distributed data parallel). torchrun sets this env variable
assert torch.cuda.is_available()
dist.init_process_group(backend='nccl')
ddp_rank = int(os.environ['RANK'])
ddp_local_rank = int(os.environ['LOCAL_RANK'])
ddp_world_size = int(os.environ['WORLD_SIZE'])
device = f'cuda:{ddp_local_rank}'
torch.cuda.set_device(device)
print(f"using device: {device}")
master_process = (ddp_rank == 0) # this process will do logging, checkpointing etc.

# convenience variables
B, T = args.device_batch_size, args.sequence_length
# calculate the number of steps to take in the val loop.
assert args.val_tokens % (B * T * ddp_world_size) == 0
val_steps = args.val_tokens // (B * T * ddp_world_size)
# calculate the steps of gradient accumulation required to attain the desired global batch size.
assert args.batch_size % (B * ddp_world_size) == 0
train_accumulation_steps = args.batch_size // (B * ddp_world_size)

# load tokens
train_loader = DistributedDataLoader(args.input_bin, B, T, ddp_rank, ddp_world_size)
val_loader = DistributedDataLoader(args.input_val_bin, B, T, ddp_rank, ddp_world_size)
if master_process:
    print(cmd_args)
    print(f"Training DataLoader: total number of tokens: {train_loader.ntok_total} across {len(train_loader.files)} files")
    print(f"Validation DataLoader: total number of tokens: {val_loader.ntok_total} across {len(val_loader.files)} files")
x, y = train_loader.next_batch()

# there are only 50257 unique GPT-2 tokens; we extend to nearest multiple of 128 for efficiency. suggested to me by @Grad62304977.
# this originates from Karpathy's experiments.
num_vocab = 50304
model = GPT(GPTConfig(vocab_size=num_vocab, n_layer=12, n_head=768//HEAD_SIZE, n_embd=768))
model = model.cuda()
torch._dynamo.config.optimize_ddp = False # otherwise compiler will complain
if hasattr(config, "coordinate_descent_tuning"):
    config.coordinate_descent_tuning = True # suggested by @Chillee
# config.max_autotune = True # faster, but VERY slow to compile
model = torch.compile(model, fullgraph=True)
# here we wrap model into DDP container
model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module # always contains the "raw" unwrapped model
ctx = torch.amp.autocast(device_type='cuda', dtype=torch.float)

# init the optimizer(s)
opt_group1 = []
opt_group2 = []
opt_name1 = []
opt_name2 = []
n_params = 0
for n,p in raw_model.named_parameters():
    if ".attn.receptance." in n or 'attn.key.' in n or 'attn.value.' in n or 'attn.gate.' in n or 'attn.output.' in n or '.mlp.' in n:# or 'gate_w' in n:# or '_w1' in n or '_w2' in n:
        opt_group2.append(p)
        if '.1.' not in n and '.2.' not in n and '.3.' not in n and '.4.' not in n and '.5.' not in n and '.6.' not in n and '.7.' not in n and '.8.' not in n and '.9.' not in n and '.10.' not in n and '.11.' not in n:
            opt_name2.append(n)
    else:
        opt_group1.append(p)
        if '.1.' not in n and '.2.' not in n and '.3.' not in n and '.4.' not in n and '.5.' not in n and '.6.' not in n and '.7.' not in n and '.8.' not in n and '.9.' not in n and '.10.' not in n and '.11.' not in n:
            opt_name1.append(n)
    n_params += p.numel()
if master_process:
    print('model params', n_params)
    print(f'Adam\n{opt_name1}...')
    print(f'Muon\n{opt_name2}...')

# init the optimizer(s)
optimizer1 = torch.optim.AdamW(opt_group1, lr=cmd_args.adam_lr, betas=(0.9, 0.95),
                               weight_decay=args.weight_decay, fused=True)
optimizer2 = Muon(opt_group2, lr=cmd_args.muon_lr, momentum=0.95)
optimizers = [optimizer1, optimizer2]
# learning rate decay scheduler (linear warmup and warmdown)
def get_lr(it):
    assert it <= args.num_iterations
    # 1) linear warmup for warmup_iters steps
    if it < args.warmup_iters:
        return (it+1) / args.warmup_iters
    # 2) constant lr for a while
    elif it < args.num_iterations - args.warmdown_iters:
        return 1.0
    # 3) linear warmdown
    else:
        decay_ratio = (args.num_iterations - it) / args.warmdown_iters
        return decay_ratio
schedulers = [torch.optim.lr_scheduler.LambdaLR(opt, get_lr) for opt in optimizers]

# begin logging
if master_process:
    run_id = datetime.datetime.today().strftime("%Y-%m-%d-%H-%M-%S")
    run_prefix = 'v6fla' if cmd_args.fla_cuda else 'v6'
    if cmd_args.random_seed != -1:
        run_prefix += f' seed{cmd_args.random_seed}'
    wandb.init(
        project='fast-nanogpt',
        name=f'{run_prefix} {cmd_args.adam_lr}/{cmd_args.emb_scale} {run_id}',
        config=args,
        save_code=False,
    )

    logdir = 'logs/%s/' % run_id
    os.makedirs(logdir, exist_ok=True)
    logfile = 'logs/%s.txt' % run_id
    # create the log file
    with open(logfile, "w") as f:
        # begin the log by printing this file (the Python code)
        f.write(str(cmd_args) + '\n')
        f.write('='*100 + '\n')
        f.write(code)
        f.write('='*100 + '\n')
        # log information about the hardware/software environment this is running on
        # and print the full `nvidia-smi` to file
        f.write(f"Running pytorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}\nnvidia-smi:\n")
        import subprocess
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        f.write(f'{result.stdout}\n')
        f.write('='*100 + '\n')

training_time_ms = 0
# start the clock
torch.cuda.synchronize()
t0 = time.time()
# begin training
train_loader.reset()
for step in range(args.num_iterations + 1):
    last_step = (step == args.num_iterations)
    # This effectively ignores timing first 10 steps, which are slower for weird reasons.
    # Alternately, and slightly more correctly in terms of benchmarking, we could do 10
    # steps with dummy data first, and then re-initialize the model and reset the loader.
    if step == 10:
        training_time_ms = 0
        t0 = time.time()
    timed_steps = float('nan') if step <= 11 else (step - 10) + 1 # <= 11 to avoid bug in val

    # once in a while evaluate the validation dataset
    if (last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)):
        # stop the clock
        torch.cuda.synchronize()
        training_time_ms += 1000 * (time.time() - t0)
        # run validation batches
        model.eval()
        val_loader.reset()
        val_loss = 0.0
        for _ in range(val_steps):
            x_val, y_val = val_loader.next_batch()
            with ctx: # of course, we'd like to use no_grad() here too, but that creates a torch.compile error for some reason
                _, loss = model(x_val, y_val, return_logits=False)
                val_loss += loss.detach()
                del loss
        dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
        val_loss /= val_steps
        # log val loss to console and to logfile
        if master_process:
            print(f'step:{step}/{args.num_iterations} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/(timed_steps-1):.2f}ms')
            with open(logfile, "a") as f:
                f.write(f'step:{step}/{args.num_iterations} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/(timed_steps-1):.2f}ms\n')
            wandb.log({"loss_val": val_loss.item()}, step=int(step+1))
        # start the clock again
        torch.cuda.synchronize()
        t0 = time.time()

    if master_process and (last_step or (args.save_every > 0 and step % args.save_every == 0)):
        # stop the clock
        torch.cuda.synchronize()
        training_time_ms += 1000 * (time.time() - t0)
        # save the state of the training process
        log = dict(step=step, code=code, model=raw_model.state_dict(), optimizers=[opt.state_dict() for opt in optimizers])
        torch.save(log, 'logs/%s/state_step%06d.pt' % (run_id, step))
        # start the clock again
        torch.cuda.synchronize()
        t0 = time.time()

    # bit confusing: we want to make sure to eval on 0th iteration
    # but also after the very last iteration. so we loop for step <= num_iterations
    # instead of just < num_iterations (one extra due to <=), only to do
    # the validation/sampling one last time, and then we break right here as we're done.
    if last_step:
        break

    # --------------- TRAINING SECTION BEGIN -----------------
    model.train()
    for i in range(1, train_accumulation_steps+1):
        # forward pass
        with ctx:
            _, loss = model(x, y, return_logits=False)
            train_loss = loss.detach()
        # advance the dataset for the next batch
        x, y = train_loader.next_batch()
        # backward pass
        if i < train_accumulation_steps:
            with model.no_sync(): # there's no need to sync gradients every accumulation step
                loss.backward()
        else:
            loss.backward() # just sync on the last step
    for p in model.parameters():
        p.grad /= train_accumulation_steps
    # step the optimizers and schedulers

    lr = []
    for opt, sched in zip(optimizers, schedulers):
        opt.step()
        sched.step()
        lr.append(sched.get_last_lr())

    # null the gradients
    model.zero_grad(set_to_none=True)
    # --------------- TRAINING SECTION END -------------------
    # everything that follows now is just diagnostics, prints, logging, etc.

    #dist.all_reduce(train_loss, op=dist.ReduceOp.AVG) # all-reducing the training loss would be more correct in terms of logging, but slower
    if master_process:
        approx_time = training_time_ms + 1000 * (time.time() - t0)
        print(f"step:{step+1}/{args.num_iterations} train_loss:{train_loss.item():.4f} train_time:{approx_time:.0f}ms step_avg:{approx_time/timed_steps:.2f}ms")
        with open(logfile, "a") as f:
            f.write(f"step:{step+1}/{args.num_iterations} train_loss:{train_loss.item():.4f} train_time:{approx_time:.0f}ms step_avg:{approx_time/timed_steps:.2f}ms\n")
        wandb.log({"loss": train_loss.item(), "lr1": float(lr[0][0]), "lr2": float(lr[1][0]), "step_t": approx_time/timed_steps}, step=int(step+1))

if master_process:
    print(f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")

# -------------------------------------------------------------------------
# clean up nice
dist.destroy_process_group()
