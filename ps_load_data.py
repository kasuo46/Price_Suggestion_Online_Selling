import pandas as pd
from print_df_info import *

data_dir = 'Data/'
train = pd.read_csv(data_dir + 'train.tsv', delimiter='\t', encoding='utf-8')
print_df_info(train, info=True, head=True, describe=True, lines=20)
