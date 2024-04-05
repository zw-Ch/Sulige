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
Go to [pred](https://github.com/zw-Ch/Sulige/tree/main/pred) page to predict daily oil production

### 1.Autoregressive Model
- **Data**:<br>
Single Time Series, $x=[x_{0},x_{1},...,x_{N-2},x_{N-1}]$

- **Training Set**:<br>
Previous part of time series,

- **Test Set**:<br>
Latter part of times series, 

### 2. 

```
cd pred
python pred_block_mul.py       // Predict Production in multiple Blocks
python pred_block_one.py       // Predict Production in single Block
python pred_well_mul.py        // Predict Production in multiple Wells
python pred_well_one.py        // Predict Production in single Well 
```
