# 
module load python
conda remove --name llms --all
conda create --name llms
conda activate llms
conda install jaxlib=*=*cuda* jax cuda-nvcc -c conda-forge -c nvidia
pip install torch transformers flax sacremoses pandas