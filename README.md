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

## Program Description
### 1. Predict Daily Gas Production
Go to [pred](https://github.com/zw-Ch/Sulige/tree/main/pred) page, and predict time series 
```
cd pred
python pred_block_mul.py       // Predict Production in multiple Blocks
python pred_block_one.py       // Predict Production in single Block
python pred_well_mul.py        // Predict Production in multiple Wells
python pred_well_one.py        // Predict Production in single Well 
```
