from dataclasses import dataclass
from multiprocessing import Lock, Process, Queue, current_process
from torch import manual_seed

from torch import Tensor, zeros, save
from simplellm.tokenizers import SPTokenizer
from simplellm.llama import LLamaFirstStage, LLamaStage

from torch.optim import Adam
from torch import cuda, no_grad
import traceback
import torch
from contextlib import redirect_stdout
from simplellm.dataloaders import Wikipedia_Dataset, TinyStories
import torch.nn.functional as F
from time import time
from simplellm.losses import causalLLMLoss
import pickle
# Messages Exchanged by the processes
@dataclass
class Forward:
    tag: int
    frm: int
    to: int
    originator: int
    data: Tensor
@dataclass
class Backward:
    tag: int
    frm: int
    to: int
    originator: int
    data: Tensor


@dataclass
class Gradients:
    frm: int
    data: Tensor

@dataclass
class SendGradients:
    frm: int
    data: Tensor

@dataclass
class Weights:
    frm: int
    ldata: Tensor
    rdata: Tensor
@dataclass
class SendWeights:
    frm: int
    data: Tensor

@dataclass
class Start:
    tag: int
    to: int
    originator: int

@dataclass
class Deferred:
    tag: int

@dataclass
class Loss:
    tag: int
    frm: int
    to: int
    originator: int
    data: Tensor



@dataclass
class Aggregate:
    epoch: int

def run_p(queue_in: Queue, queue_out: Queue, node_id: int = 0, stage: int = 0, seq_l: int = 256, n_layers = 4, 
                    batch_size = 16, dmodel = 256, num_heads = 16, mb_count = 6,
                    device = "cuda"):
    manual_seed(0)

    if stage == 0:
        tkns = SPTokenizer()
        ts = Wikipedia_Dataset(tkns,batch_size = batch_size, seq_l=seq_l)
        net = LLamaFirstStage(tkns.vocab_size, dmodel, num_heads, 0, ctx_size= seq_l,device=device)
        
        optimizer = Adam(net.parameters(),6e-4/3)
        with open(f'log{node_id}.txt', 'a') as file, redirect_stdout(file):
            loc =  SubP(queue_in,queue_out,net,optimizer,node_id,stage,ts,device=device, mb_count=mb_count)
            loc.start()
    else:
        net = LLamaStage(ctx_size=seq_l, dmodel=dmodel,num_heads=num_heads,n_layers=n_layers,device=device)
        
        optimizer = Adam(net.parameters(),6e-4/3)
        
        loc =  SubP(queue_in,queue_out,net,optimizer,node_id,stage,None,device=device,  mb_count=mb_count)
        loc.start()


class SubP(object):
    def __init__(self,queue_in: Queue, queue_out: Queue, net, optimizer, node_id = 0, stage = 0, ds = None,
                    device = "cuda", mb_count = 12) -> None:
        self.net = net
        
        
        self.device = device
        self.queue_in: Queue = queue_in
        self.queue_out: Queue = queue_out
        self.optimizer = optimizer
        
        self.node_id = node_id
        self.buffer_in = {}
        self.buffer_out = {}
        self.receives = []
        self.iteration = 0
        self.mb_count = mb_count
        self.started = True
        
        self.mbs = []
        self.epoch = 0
        self.len_sizes = []
        self.sizes = []
        for param in self.net.parameters():
            self.sizes.append(param.shape)
            self.len_sizes.append(len(param.view(-1)))
        if stage == 0:
            self.ds = ds
            self.dl = iter(ds)
            
            self.target = {}
        
        
        


    def start(self):
        try:
            while self.started:
                
                while self.queue_in.empty() and self.started:
                    continue
                if not self.started:
                    break
                
                task = self.queue_in.get(True)
                if isinstance(task, Start):
                    
                    with open(f"log_stats_proj_2_{self.node_id}.txt", "a") as log:
                        
                        log.write(f"=======NEW MB:======== {task.tag} {time()}\n")
                    
                    
                    x = next(self.dl)
                    self.target[task.tag] = x.detach().clone()
                    with no_grad():
                        x = x.to(self.device)

                    self.buffer_in[task.tag] = x
                    x = self.net.embed(x)
                    x.retain_grad()
                    self.buffer_out[task.tag] = x
                    ret = pickle.dumps(x)
                    
                    
                    self.queue_out.put(Forward(task.tag, self.node_id, task.to, task.originator, ret), True)
                    
                    
                    continue
                elif isinstance(task, Loss):
                    
                    
                    
                    x = pickle.loads(task.data)
                    with no_grad():
                        x = x.to(self.device)
                    x.requires_grad = True
                    x.retain_grad()
                    y = self.target[task.tag].to(self.device)
                    ret = self.net.forward_end(x)

                    loss = causalLLMLoss(ret,y,vocab_size=self.ds.tokenizer.vocab_size)
                    loss_report = loss.item()
                    loss = loss / self.mb_count
                    loss.backward()
                    
                    
                    ret = x.grad
                    with open(f"log_stats_proj_2_{self.node_id}.txt", "a") as log:
                        log.write(f"LOSS {task.tag}:{loss_report} \n") #
                    # ret = ret.to("cpu")
                    
                    self.queue_out.put(Loss(task.tag, task.frm, task.to,  task.originator, pickle.dumps(ret)), True)
                    
                elif isinstance(task, Forward):
                    
                    with open(f"log_stats_proj_2_{self.node_id}.txt", "a") as log:
                        log.write(f"Processing forward to {task.to} {time()}\n")
                    
                    x = pickle.loads(task.data)
                    
                    with no_grad():
                        x = x.to(self.device)
                    x.requires_grad = True
                    x.retain_grad()

                    self.buffer_in[task.tag] = x
                    
                    
                    x = self.net(x)
                    x.retain_grad()
                    self.buffer_out[task.tag] = x

                    ret = pickle.dumps(x)
                    
                    
                    self.queue_out.put(Forward(task.tag, task.frm, task.to,  task.originator, ret), True)
                    
                    continue
                    
                elif isinstance(task, Backward):
                    output = pickle.loads(task.data)
                    
                    
                    with no_grad():
                        output = output.to(self.device)
                        
                    inp_batch = self.buffer_out[task.tag].to(self.device)
                    
                    inp_batch.backward(output)
                   
                    
        
                    
                    if task.to != -1:
                        ret = self.buffer_in[task.tag].grad
                        self.queue_out.put(Backward(task.tag, task.frm, task.to, task.originator, pickle.dumps(ret)),True)

                    else:
                        self.queue_out.put(Backward(task.tag, task.frm, task.to, task.originator, None),True)

                    del self.buffer_in[task.tag]
                    del self.buffer_out[task.tag] 
                    cuda.empty_cache()
                    

                    
                elif isinstance(task, Gradients):
                    data = pickle.loads(task.data)
                    tmp = torch.split(data, self.len_sizes)
                    for i, param in enumerate(self.net.parameters()):
                        if param.grad == None:
                            param.grad = tmp[i].view(self.sizes[i]).to(param.device).to(param.data.dtype)
                            continue
                        param.grad += tmp[i].view(self.sizes[i]).to(param.device).to(param.data.dtype)
                        
                    
                elif isinstance(task, SendGradients):
                    with open(f"log_stats_proj_2_{self.node_id}.txt", "a") as log:
                        log.write(f"=PREPARING GRADIENTS= {time()}\n")
                    tmp = []
                    for param in self.net.parameters():
                        if param.grad == None:
                            tmp.append(zeros_like(param).view(-1))
                                        
                            continue
                        
                        tmp.append(param.grad.view(-1))
                    
                    prev_grad = torch.cat(tmp).to("cpu")
                    self.queue_out.put(SendGradients(task.frm,pickle.dumps(prev_grad)), True)

                elif isinstance(task, Weights):
                    if task.rdata == None:
                        data = pickle.loads(task.ldata)
                        tmp = torch.split(data, self.len_sizes)
                        for i, param in enumerate(self.net.parameters()):
                            param.data = tmp[i].view(self.sizes[i]).to(param.device).to(param.data.dtype)
                    else:
                        datal = pickle.loads(task.ldata)
                        tmpl = torch.split(datal, self.len_sizes)

                        datar = pickle.loads(task.rdata)
                        tmpr = torch.split(datar, self.len_sizes)
                        for i, param in enumerate(self.net.parameters()):
                            param.data = 0.5*tmpl[i].view(self.sizes[i]).to(param.device).to(param.data.dtype) + 0.5 * tmpr[i].view(self.sizes[i]).to(param.device).to(param.data.dtype)

                    with open(f"log_stats_proj_2_{self.node_id}.txt", "a") as log:
                        log.write(f"MODEL READY {time()}\n")
                    
                elif isinstance(task, SendWeights):
                    tmp = []
                    for param in self.net.parameters():
                        if param == None:
                            tmp.append(zeros_like(param).view(-1))
                                        
                            continue
                        
                        tmp.append(param.view(-1))
                    
                    prev_grad = torch.cat(tmp).to("cpu")
                    self.queue_out.put(SendWeights(task.frm,pickle.dumps(prev_grad)), True)
                
                elif isinstance(task, Aggregate):
                    
                    
                    self.buffer_in.clear()
                    self.buffer_out.clear()
                    cuda.empty_cache()
                    self.iteration += 1
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    
                    cuda.empty_cache()
                    self.queue_out.put(Aggregate(0), True)
                    with open(f"log_stats_proj_2_{self.node_id}.txt", "a") as log:
                        log.write(f"===AGGREGATING==== {self.iteration} {time()}\n")
                    
                    
        except Exception:
            with open(f"log_stats_proj_2_{self.node_id}.txt", "a") as log:
                log.write(f"{traceback.format_exc()}\n")
            
            exit()