from data_model_handling import import_data, get_level_data, TransformerPipeline

import os
import pandas as pd
import numpy as np
import math

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score, precision_score, recall_score



def create_classification_hierarchy(root_path, levels, cls_type, only_main_ipc):
	"""Create a classification hierarchy
	
	This method creates a classification hierarchy by iteratively creating all nodes level per level. Since each node carries the 
	references to its sub nodes, the whole hierarchy can be accessed by simply referencing the seed node.
	
	Parameters
	----------
	root_path : string
		Directory path of the preprocessed training data
	
	levels : integer
		Depth of the hierarchy
	
	cls_type : python module
		Python module representing the type of hierarchical classification (LCN or LCPN) that will be used
	
	Returns
	-------
	seed_node : ParentNode
		Seed node of the hierarchy
	"""
	
	# Import raw data
	raw_data = import_data(root_path, only_main_ipc)
	
	# Start with seed parent node
	seed_node = cls_type.get_classification_node(None, 1, raw_data)
	
	print(seed_node.sub_classes)
	
	if levels > 1:
		for c in seed_node.sub_classes:
			print("### CREATING SUBCLASS", c)
			
			try:
				child_node = cls_type.get_classification_node(c, 2, raw_data)
				
				# save reference on child node in parent node object
				seed_node.add_child_node(child_node)
				print(child_node.sub_classes)
				
			except Exception:
				print("Exception @ Sub Model Creation.")
			
			if levels > 2:
				for  sc in child_node.sub_classes:
			
					print("### CREATING SUBSUBCLASS", sc)
			
					try:
						sub_child_node = cls_type.get_classification_node(sc, 3, raw_data)
						
						# save reference on child node in parent node object
						child_node.add_child_node(sub_child_node)
						print(sub_child_node.sub_classes)
						
					except Exception:
						print("Exception @ SubSub Model Creation.")
	
	return seed_node
	
def match_predictions(y_predicted, y_target_names, pred_node_predicted, pred_node_target_names):
	"""Add final level node prediction results to final level-wide prediction result
	
	This method matches the node prediction result, which is only a small fraction of the whole level prediction, with the level-wide
	prediction result matrix. Since both matrices have the same number of rows, only columns have to be matched. This is achieved by
	matching the given names of the target columns.
	
	Parameters
	----------
	y_predicted : numpy array
		Level wide prediction
	
	y_target_names : list of strings
		Ordered list of names of the y target classes/ labels corresponding to y_predicted
	
	pred_node_predicted : numpy array
		Node prediction
	
	pred_node_target_names : list of strings
		Ordered list of names of the y target classes/ labels corresponding to pred_node_predicted
	
	Returns
	-------
	y_predicted : numpy array
		Updated level wide prediction
	"""
	
	# iterate through all columns in node prediction and add predictions to y_predicted based on a target name match
	for pred_index, pred_column in enumerate(pred_node_target_names):

		try:
			match_column_index = y_target_names.index(pred_column)
			y_predicted[:, match_column_index] = y_predicted[:, match_column_index] + pred_node_predicted[:, pred_index]
		except Exception:
			continue
		
	return y_predicted
	
	
def predict_with_hierarchy(seed_node, levels, test_directory, only_main_ipc, binary_pred):
	"""Predict with hierarchy
	
	This method provides multilabel prediction results using the provided trained classification hierarchy by iteratively
	predicting labels with every node for always the whole provided test data, matching the prediction results between neighboring
	levels and aggregating all fractioned node predictions on the final level to a single level prediction result array.
	
	Parameters
	----------
	seed_node : ParentNode
		Seed node of the trained classification hierarchy
	
	levels : integer
		Number of hierarchy levels
	
	test_directory : string
		Directory path with all preprocessed test data
	
	Returns
	-------
	y_predicted: numpy array
		Prediction of hierarchy on provided level
	
	y_test: numpy array
		True data on provided level
	"""
	# import data from test directory and adjust them to provided level
	df_test = import_data(test_directory, only_main_ipc)
	df_test = get_level_data(df_test, levels, None)
	
	# get true_y at provided level
	count_vec_y = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None, token_pattern = r"(?u)\b\w+\b")
	y_test = count_vec_y.fit_transform(df_test['ipcs'])
	y_test_target_names = count_vec_y.get_feature_names()

	# create empty result matrix
	y_predicted = np.zeros(y_test.shape)

	# get predictions on whole level by node classifiers
	test_pred = seed_node.predict(df_test)
	
	# transform probability predictions into binary predictions if true
	if binary_pred:
		test_pred = (test_pred > 0.5)
	
	if levels == 1:
		# match predictions with empty result matrix
		y_predicted = match_predictions(y_predicted, y_test_target_names, test_pred, seed_node.sub_classes)
	
	if levels > 1:
		
		# iterate through all child_nodes of seed_node to generate their predictions
		for sn in seed_node.child_nodes:
			sn_test_pred = sn.predict(df_test, seed_node, test_pred)
			if binary_pred:
				sn_test_pred = (sn_test_pred > 0.5)
		
			if levels == 2:
				# match predictions with empty result matrix
				y_predicted = match_predictions(y_predicted, y_test_target_names, sn_test_pred, sn.sub_classes)
	
			if levels > 2:
				
				# iterate through all child_nodes of child_nodes of seed_node to generate their predictions
				for ssn in sn.child_nodes:
					ssn_test_pred = ssn.predict(df_test, sn, sn_test_pred)
					if binary_pred:
						ssn_test_pred = (ssn_test_pred > 0.5)
			
					if levels == 3:
						# match predictions with empty result matrix
						y_predicted = match_predictions(y_predicted, y_test_target_names, ssn_test_pred, ssn.sub_classes)
	
	return y_predicted, y_test