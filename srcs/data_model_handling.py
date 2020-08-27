import os
import pandas as pd
import numpy as np
import math

import pickle

def import_data(directory, only_main_ipc):
	"""Imports preprocessed patent files with patent text and ipcs from txt files in given hierarchical directory.
	
	Parameters
	----------
	directory : string
		Path of hierarchical directory with preprocessed patent txt files
	
	Returns
	-------
	result_df : pandas DataFrame
		Two-column DataFrame with a 'document' (string) column and an 'ipcs' (string) column
	"""
	
	data_columns = ['document', 'ipcs']
	result_df = pd.DataFrame(columns = data_columns)
	
	counter = 0
	
	for root, dirs, files in os.walk(directory):

		for file_ in files:
			
			if '.txt' in file_:
				
				counter += 1
				
				if counter%5000 == 0:
					print(counter)
					
				path = os.path.join(root, file_)
				
				document = open(path, encoding="utf8", errors='ignore')
				
				# read patent ipcs from first line & preprocessed patent description from remaining lines
				document_ipcs = document.readline().split()
				document_text = document.read()
				
				# if true only use the main ipc
				if only_main_ipc:
					document_ipcs = [document_ipcs[0]]
					
				label_list = list(set(document_ipcs))
				label_str = ' '.join(label_list)
				
				df_entry = {'document': document_text, 'ipcs': label_str}
				
				result_df = result_df.append(df_entry, ignore_index=True)
			
	return result_df
	
def get_level_data(raw_data, label_level, ipc_content):
	"""Modifies label in given DataFrame based on given label level and ipc filter
	
	This method has to main functionalities:
	1. Cutting all ipc labels to the length that corresponds to the provided label-level
	2. Filtering all patents and ipcs based on a provided ipc-content. This allows to create sub-datasets from the original raw dataset just by
	providing the level and the name of a node in the hierarchy 
	
	Parameters
	----------
	raw_data : pandas DataFrame
		DataFrame with patent text and raw ipcs
	
	label_level : integer
		Determines the length by which all ipcs are trimmed and combined if not unique
	
	ipc_content : string or None
		Used to filter out all entries that don't have any IPCS containing the ipc_content string
	
	Returns
	-------
	level_data : pandas DataFrame
		Filtered two-column DataFrame with a 'document' (string) column and an modified 'ipcs' (string) column
	"""

	# only use patents that contain the provided ipc_content if provided
	if ipc_content is not None:
		level_data = raw_data[raw_data['ipcs'].str.contains(ipc_content.upper())].copy()
	else:
		level_data = raw_data.copy()
		
	
	# define character index/ string length for ipc labels that corresponds to the hierarchy level
	label_level_split_mapping = {'1':1, '2':3, '3':4, '4':7}
	
	for index, row in level_data.iterrows():
		document_ipcs = row['ipcs'].split()
		
		# cut all ipcs to the respective ipc label length
		document_ipcs = [ipc[:label_level_split_mapping[str(label_level)]] for ipc in document_ipcs]
		
		# convert into set to drop all redundant labels after cutting to get a list of unique labels
		label_list = list(set(document_ipcs))
		
		# only use labels that contain the provided ipc_content if not None
		if ipc_content is None:
			label_list_clean = label_list
		else:
			label_list_clean = []
			for ipc in label_list:
				if ipc_content.lower() in ipc.lower():
					label_list_clean.append(ipc)
				
		label_str = ' '.join(label_list_clean)
		row['ipcs'] = label_str
		
	return level_data
	
	
def save_object(obj, filename):
	"""Saves object to given filename path.
	
	Parameters
	----------
	obj : Python object
		Any python object
	
	filename : string
		Filename path location for storing the given python object
	"""
	
	with open(filename, 'wb') as output:  # Overwrites any existing file
		pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def load_object(filename):
	"""Loads object from given filename path.
	
	Parameters
	----------
	filename : string
		Filename path location for loading the given python object
	
	Returns
	-------
	obj : Python object
		Loaded python object
	"""
	
	with open(filename, 'rb') as input:
		obj = pickle.load(input)
	return obj
		
		
	
class TransformerPipeline:
	"""Transformer Pipeline
	
	This class serves as a container for transformer objects that are fit to the training data and are supposed to be applied one
	after another to any other data.
	
	Parameters
	----------
	transformer_list : list of transformer objcts
		List with all transformer objects that are already fit to the training data
	"""
	
	def __init__(self, transformer_list):
		self.transformer_list = transformer_list
		
	def transform(self, data):
		"""Transform data with list of transformer objects
	
		Parameters
		----------
		data : pandas DataFrame
			Input data
		
		Returns
		-------
		data : numpy array
			Data with all transformer object's transform method applied
		
		"""
		
		for t in self.transformer_list:
			data = t.transform(data)
			
		return data