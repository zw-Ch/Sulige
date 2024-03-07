"""
"""
import os.path as osp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
import func.process as pro


file_ad = '../dataset/product_data'
df = pd.read_excel(osp.join(file_ad, '47(2).xls'))


print()
plt.show()
print()
