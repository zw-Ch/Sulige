# Sulige

## Installation
Create a virtual environment <br>
```
conda create -n your-env-name python=3.11
```
Then, install some Python packages<br>
```
conda install pytorch=2.1.0 torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install pyg -c pyg
conda install pandas
conda install matplotlib
conda install xlrd
```

## Predict Daily Oil Production
<!--Go to [pred](https://github.com/zw-Ch/Sulige/tree/main/pred) page to predict daily oil production -->

### 1. Autoregressive Model
If you plan to analyze one well, that is:
- **Data**:<br>
Single sequences, $\boldsymbol{x}=[x_{0}, x_{1}, ..., x_{N-2}, x_{N-1}]$

- **Training Set**:<br>
Previous part of sequences, $\boldsymbol{x} = [x_{0}, x_{1}, ..., x_{t-1}, x_{t}]$

- **Test Set**:<br>
Latter part of sequences, $\boldsymbol{x} = [x_{t+1}, x_{t+2}, ..., x_{N-2}, x_{N-1}]$
```
cd pred_one
python pred_gnn_one_well.py       // Predict Production in multiple Blocks
python pred_gnn_one_well.py       // Predict Production in single Block
python pred_well_mul.py        // Predict Production in multiple Wells
python pred_well_one.py        // Predict Production in single Well 
```

### 2.  
If you plan to analyze multiple wells, that is:
- **Data**:<br>
Multiple sequences, $\boldsymbol{X}=[\boldsymbol{x}\_{0}, \boldsymbol{x}\_{1}, ..., \boldsymbol{x}\_{N-2}, \boldsymbol{x}\_{N-1}]$

- **Training Set**:<br>
Several sequences, $\boldsymbol{X}=[\boldsymbol{x}\_{0}, \boldsymbol{x}\_{1}, ..., \boldsymbol{x}\_{t-1}, \boldsymbol{x}\_{t}]$

- **Test Set**:<br>
Other sequences, $\boldsymbol{X}=[\boldsymbol{x}\_{t+1}, \boldsymbol{x}\_{t+2}, ..., \boldsymbol{x}\_{N-2}, \boldsymbol{x}\_{N-1}]$
```
cd pred_mul
python pred_block_mul.py       // Predict Production in multiple Blocks
python pred_block_one.py       // Predict Production in single Block
python pred_well_mul.py        // Predict Production in multiple Wells
python pred_well_one.py        // Predict Production in single Well 
```
