import os
import os.path as osp
import sys
sys.path.append('..')
import func.well as well


root = "../dataset"
block_name = "120（1）"

for block_name in os.listdir(osp.join(root, "product_data")):
    block_name = block_name.split('.')[0]
    block = well.Block(root, block_name)


print()
