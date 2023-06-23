#-
import torch
from time import time
from utils.embed import tokenize_texts, embed_tokens, embed_tokens_tensor

#-- 
def cpu_test(texts, tokenizer, model):
  tokens = tokenize_texts(texts, tokenizer)
  embeds = embed_tokens(tokens, model)
  del(tokens); del(embeds)

#--
def gpu_test(texts, tokenizer, model, device):
  model = model.to(device)
  tokens = tokenize_texts(texts, tokenizer)
  tokens = torch.Tensor(tokens).long().to(device)
  embeds = embed_tokens_tensor(tokens, model)
  del(tokens); del(embeds)
  model = model.cpu()

#--
def gpu_batch_limit(max_batch, max_seq, model, device):
  model = model.to(device)
  limit = 1
  for i in range(1, max_batch):
    try:
      tokens = torch.ones((i, max_seq)).long().to(device)
      embeds = embed_tokens_tensor(tokens, model)
    except: limit = i-1; break
  del(tokens); del(embeds)
  model = model.cpu()
  torch.cuda.empty_cache()
  return limit

#--
def gpu_batch_runtime(max_batch, max_seq, model, device, samples=100):
  model = model.to(device)
  ts = []
  for i in range(samples):
    tokens = torch.ones((max_batch, max_seq)).long().to(device)
    t = time()
    embeds = embed_tokens_tensor(tokens, model)
    ts.append(time()-t)
    del(tokens); del(embeds)
    torch.cuda.empty_cache()
  model = model.cpu()
  torch.cuda.empty_cache()
  return sum(ts)/samples

#--
def benchmark_llm(nme, read_llm, batch_limit, max_seq, texts, context):

  #- LLM Load
  tokenizer, model = read_llm()
  params = sum(p.numel() for p in model.parameters())
  print("rank-{}: {}: Number Trainable Params: {}".format(context.rank, nme, params))
  context.barrier()

  #- LLM CPU Test
  cpu_test(texts, tokenizer, model)
  print("rank-{}: {}: CPU Test Passed.".format(context.rank, nme))
  context.barrier()

  #- LLM GPU Test
  gpu_test(texts, tokenizer, model, context.device)
  print("rank-{}: {}: GPU Test.".format(context.rank, nme))
  context.barrier()

  #- LLM Throughput Test
  max_batch = gpu_batch_limit(batch_limit, max_seq, model, context.device)
  runtime = gpu_batch_runtime(max_batch, max_seq, model, context.device)
  print("rank-{}: {}: Inference on tensor ({}, {}): {} sec".format(
    context.rank, 
    nme, 
    max_batch, 
    max_seq, 
    runtime))
  context.barrier()