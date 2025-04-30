# Fractional Derivatives for Residual Neural Network Backpropagation
## Setup
```
module load anaconda/2024.02-py311
conda create -n FRN python=3.11 ipython ipykernel -y
conda activate FRN
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install matplotlib
```

## Experiments
### Base model
`python main.py --method base --num-epochs 80 --lr 0.001`

[Log](/slurm_logs/base.txt)

![image](/plots/base_a1.png)

### 𝛼=0.7
`python main.py --method GL --alpha 0.7 --num-epochs 80 --max-history 15 --lr 0.001`

[Log](/slurm_logs/GL_a0_7.txt)

![image](/plots/GL_a0_7.png)

### 𝛼=0.8
`python main.py --method GL --alpha 0.8 --num-epochs 80 --max-history 15 --lr 0.001`

[Log](/slurm_logs/GL_a0_8.txt)

![image](/plots/GL_a0_8.png)

### 𝛼=0.9
`python main.py --method GL --alpha 0.9 --num-epochs 80 --max-history 15 --lr 0.001`

[Log](/slurm_logs/GL_a0_9.txt)

![image](/plots/GL_a0_9.png)

### 𝛼=1.0
`python main.py --method GL --alpha 1.0 --num-epochs 80 --max-history 15 --lr 0.001`

[Log](/slurm_logs/GL_a1_0.txt)

![image](/plots/GL_a1_0.png)

### 𝛼=1.1
`python main.py --method GL --alpha 1.1 --num-epochs 80 --max-history 15 --lr 0.001`

[Log](/slurm_logs/GL_a1_1.txt)

![image](/plots/GL_a1_1.png)
