#-
import torch

#--
def tokenize_texts(texts, tokenizer):
  batch = []
  for text in texts:
    tokens = tokenizer(text)["input_ids"]
    batch.append(tokens)
  return batch

#--
def embed_tokens(tokens, model, layer_id=-1, agg=["sum", "mean"][0]):
  batch = []
  for tok in tokens:
    ids = torch.Tensor([tok]).long()
    embed = model(input_ids=ids, output_hidden_states=True, return_dict=True)
    if "hidden_states" in embed: embed = embed["hidden_states"][layer_id]
    elif "encoder_hidden_states" in embed: embed = embed["encoder_hidden_states"][layer_id]
    if agg == "sum": embed = embed.sum(1)
    if agg == "mean": embed = embed.mean(1)
    batch.append(embed)
  else: return batch

#--
def embed_tokens_tensor(tensor, model, layer_id=-1, agg=["sum", "mean"][0]):
  embed = model(input_ids=tensor, output_hidden_states=True, return_dict=True)
  if "hidden_states" in embed: embed = embed["hidden_states"][layer_id]
  elif "encoder_hidden_states" in embed: embed = embed["encoder_hidden_states"][layer_id]
  if agg == "sum": return embed.sum(1)
  if agg == "mean": return embed.mean(1)
  return batch
