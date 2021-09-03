"""
Implementation of data loader and noiid sampling

Author: Kai Zhang (www.kaizhang.us)
https://github.com/taokz
"""

import pandas as pd 
import numpy as np
import random

def dataloader_adult(train_path = "data/adult.data", test_path = "data/adult.test"):
	
	train_set = pd.read_csv(train_path)#, header = None)
	test_set = pd.read_csv(test_path)#, header = None)

	# assign columns' names
	col_labels = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status',
              'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss',
              'hours_per_week', 'native_country', 'wage_class']
	train_set.columns = col_labels
	test_set.columns = col_labels

	# deal with missing values
	train_set = train_set.replace('?', np.nan).dropna()
	test_set = test_set.replace('?', np.nan).dropna()

	# replace the value of 'wage_class' in test_set with the identical ones in the train_set 
	test_set['wage_class'] = test_set.wage_class.replace({' <=50K.': ' <=50K', ' >50K.': ' >50K'})

	# Encode categorical features
	combined_set = pd.concat([train_set, test_set], axis = 0)
	for feature in combined_set.columns:
		if combined_set[feature].dtype == 'object':
			combined_set[feature] = pd.Categorical(combined_set[feature]).codes

	combined_set.rename(columns = {'wage_class':'target'}, inplace = True)

	train_set = combined_set[:train_set.shape[0]]
	test_set = combined_set[train_set.shape[0]:]

	return train_set, test_set

def data_noniid(dataset, num_clients):
	"""
	sample non-iid client data from dataset
	"""
	lens = len(dataset)
	num_shards = cal_num_shards(lens, num_clients)
	num_data = int(lens / num_shards)
	idx_shard = [i for i in range(num_shards)]
	dict_clients = {i: np.array([], dtype = 'int64') for i in range(num_clients)}
	idxs = np.arange(num_shards*num_data)
	labels = dataset.target.to_numpy()

	# sort labels
	idxs_labels = np.vstack((idxs, labels))
	idxs_labels = idxs_labels[:, idxs_labels[1,:].argsort()]
	idxs = idxs_labels[0,:]

	# divide and assign
	count = int(num_shards / num_clients) 
	for i in range(num_clients):
		rand_set = set(np.random.choice(idx_shard, count, replace = False))
		idx_shard = list(set(idx_shard) - rand_set)
		for rand in rand_set:
			dict_clients[i] = np.concatenate((dict_clients[i], idxs[rand*num_data:(rand+1)*num_data]), axis = 0)

	return dict_clients

def cal_num_shards(lens, num_clients):
	
	flag = True
	count = 3 # start from 3
	
	while (flag):
		if (lens % (count*num_clients)) == 0:
			flag = False
		else:
			count = count + 1

	num_shards = int(count*num_clients)

	return  num_shards

def load_noniid(num_clients, train_set):
	# split training data set into multiple non-iid data sets
	train_data = train_set.values
	non_iid = []
	client_dict = data_noniid(train_set, num_clients)
	for i in range(num_clients):
		idx = client_dict[i]
		d = train_data[idx]
		non_iid.append(d)
	return non_iid

