# Importing the libraries
import numpy as np    #library that contains mathematical tools.
import matplotlib.pyplot as plt   #going to help us plot nice charts.
import pandas as pd #best library to import data sets and manage data sets.

# Importing the dataset
big_case_train = pd.read_csv('big_case_train.csv')
puzzle_train_dataset = pd.read_csv('puzzle_train_dataset.csv')
puzzle_test_dataset = pd.read_csv('puzzle_test_dataset.csv')

# Fill empty valyes of column default as false
puzzle_train_dataset['default'] = puzzle_train_dataset['default'].fillna('FALSE')

# Creating new columns in big_case_train
# cel = Comapny's earning or loss
# cc Company's cost
big_case_train['cel'] = big_case_train.apply(lambda row: (row.spends*0.05) + (row.revolving_balance*0.17), axis=1 )
big_case_train['cc'] = big_case_train.apply(lambda row: (row.card_request*10) + (row.minutes_cs*2.5), axis=1 )

# Adding Company's cost of customers
tcc = big_case_train.groupby('ids')['cc'].sum().reset_index()

# inner joint on puzzle_train_datase and big_case_train

result = pd.merge(puzzle_train_dataset[['ids', 'default']], big_case_train[['ids','month','cel']],on = 'ids', how = 'inner')
non_default=result.loc[result['default'] == 'FALSE']
non_default_ce = non_default.groupby('ids')['cel'].sum().reset_index()
non_default_ce['loss'] = 0

default = result.loc[result['default'] != 'FALSE']
default_cel = default.groupby('ids')['cel'].sum().reset_index()
loss_df1 = default.groupby('ids')['cel'].max().reset_index()
loss_df1.columns = ['ids' ,'loss']

result2 = pd.merge(loss_df1, default_cel ,on = 'ids') # ids , cel, loss
result2['ce'] = result2.apply(lambda row: row.cel - row.loss, axis=1 )
default_ce_l =  result2[['ids' , 'ce', 'loss']]

# Total company's earning and loss 
Total_ce_l = pd.concat([non_default_ce, default_ce_l])

# Merging total company's cost earning and loss
final = pd.merge(tcc,Total_ce_l, on = 'ids')
final['net_profit'] = final.apply(lambda row: (row.ce) - (row.cc) - (row.loss) , axis=1 )
final['profit'] = np.where(final['net_profit'] > 0 , 1 , 0 )
# final['profit'] = final.apply(lambda row: (row.profit) = 0 if final[net_profit]>0,  axis=1 )
final2 = final[['ids','profit']]
new_puzzle_train_dataset = pd.merge(puzzle_train_dataset, final2 ,on = 'ids', how = 'outer')
new_puzzle_train_dataset.to_csv('new_puzzle_train_dataset.csv')



