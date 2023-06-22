"""
Author: Rafael Zamora-Resendiz (LBNL)

Short script defining some distributed Pytorch contexts
for multi-gpu communication. Implemented for SLURM and LSF
scheduling environment when working on KDI and SUMMIT respectively.

"""

#--
import os, torch, random
import numpy as np
import torch.nn as nn
import torch.distributed as distr

#-
SLURM_GLOBALVARS = ["MASTER_ADDR", "MASTER_PORT", "SLURM_PROCID", "WORLD_SIZE"]
LSF_GLOBALVARS = ["WORLD_SIZE", "PMIX_RANK", "LSB_MCPU_HOSTS"]

#-
def set_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)

#---
class LocalCPUTorch():

  #--
  def __init__(self, rank, world_size, seed=666142):
    self.device = torch.device("cpu")
    self.gidx = None
    self.rank = rank
    self.world_size = world_size
    self.seed = seed
    set_seed(seed)

  #--
  def __enter__(self):
    return self

  #--
  def __exit__(self, _type, _value, _traceback):
    pass

#---
class SLURMDistributedTorch():

  #--
  def __init__(self, seed=666142):
    self.device = None
    self.gidx = None
    self.rank = None
    self.world_size = None
    self.seed = seed
    set_seed(seed)

  #--
  def __enter__(self):
    env = {key:os.environ[key] for key in SLURM_GLOBALVARS}
    os.environ["RANK"] = env["SLURM_PROCID"]
    distr.init_process_group(backend="nccl")
    self.gidx = distr.get_rank() % torch.cuda.device_count()
    torch.cuda.set_device(self.gidx)
    self.device = torch.device("cuda", self.gidx)
    self.rank = int(env["SLURM_PROCID"])
    self.world_size = int(env["WORLD_SIZE"])
    return self

  #--
  def __exit__(self, _type, _value, _traceback):
    distr.destroy_process_group()

  #-
  def barrier(self): distr.barrier()

#---
class LSFDistributedTorch():

  #--
  def __init__(self, seed=666142):
    self.device = None
    self.gidx = None
    self.rank = None
    self.world_size = None
    self.seed = seed
    set_seed(seed)

  #--
  def __enter__(self):
    env = {key:os.environ[key] for key in LSF_GLOBALVARS}
    master = list(sorted(env["LSB_MCPU_HOSTS"].split()[2::2]))[0]
    with open("/etc/hosts", "r") as f: addrs = f.readlines()
    master_addr = [line.split()[0] for line in addrs if line.strip().endswith(master)][0]
    self.world_size = int(env["WORLD_SIZE"])
    self.rank = int(env["PMIX_RANK"])
    os.environ["RANK"] = str(self.rank)
    os.environ["WORLD_SIZE"] = str(self.world_size)
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = "51234"
    distr.init_process_group(backend="nccl")
    self.gidx = distr.get_rank() % torch.cuda.device_count()
    torch.cuda.set_device(self.gidx)
    self.device = torch.device("cuda", self.gidx)
    return self

  #--
  def __exit__(self, _type, _value, _traceback):
    distr.destroy_process_group()

  #-
  def barrier(self): distr.barrier()
