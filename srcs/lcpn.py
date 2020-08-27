from data_model_handling import import_data, get_level_data, TransformerPipeline

import os
import pandas as pd
import numpy as np
import math

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import f1_score
import tensorflow
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, Flatten, GlobalMaxPool1D, Dropout, Conv1D
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.losses import binary_crossentropy
from keras.optimizers import Adam
from keras.models import load_model
from sklearn.metrics import f1_score, precision_score, recall_score

import pickle


os.environ['KMP_DUPLICATE_LIB_OK']='True'

class ParentNode:
	
	"""ParentNode
	
	This class represents a parent node in the classification hierarchy of a Local Classifier per Parent Node (LCPN) 
	concept based on the definition of Silla and Freitas [1].
	Besides carrying the name and level of the respective node, it serves as a container to save the data transformers 
	for X and y data that are fit on training data and ready to be applied on test data.
	Furthermore, it carries the name of all sub classes and references the respective child nodes, which allows to 
	iteratively build a hierarchy from top to bottom. 
	
	Once the hierarchy is built, this implementation design allows to access the hierarchy with all nodes only by 
	referencing the seed node. In addition, this design allows to merge predictions of different levels by matching
	the child node's name with the list of sub classes of the parent node.
	
	Parameters
	----------
	name : string
		Name of the node
	
	level : integer
		Level of the node within the hierarchy
	
	transformer_x : TransformerPipeline object
		Transformer pipeline fit on training X data
	
	count_vec_y : CountVectorizer object
		Count vectorizer fit on training y data
	
	sub_classes : list of strings
		List with all names of target/ child classes
	
	"""
	
	def __init__(self, name, level, model, transformer_x, count_vec_y, sub_classes):
		self.name = name
		self.level = level
		self.model = model
		self.transformer_x = transformer_x
		self.count_vec_y = count_vec_y
		self.sub_classes = sub_classes
		self.child_nodes = []
		
		
	def add_child_node(self, child_node):
		"""Add a child node to the child_nodes list
	
		Parameters
		----------
		child_node : ParentNode
			An object of the class ParentNode
		"""
		self.child_nodes.append(child_node)
		
	def predict(self, df_test, parent_node = None, parent_pred = None):
		"""Predict labels for all subclasses
		
		After transforming the input data, this method predicts labels using the trained model. Subsequently, if a parent_node and parent_pred
		is provided, the resulting multi-column array is matched against the respective parent_pred column in order to transfer label
		predictions between neighboring levels.
	
		Parameters
		----------
		df_test : pandas DataFrame
			Input data to predict from
		
		parent_node : ParentNode, None
			Parent node that carries a string list with all subclass names, which is used to get the respective subclass index in the parent_pred
			array
		
		parent_pred : numpy array
			Prediction results on the same input data by parent node.
		
		Returns
		-------
		test_pred : numpy array
			Prediction result on input data by this node, that has been matched against the upper level prediction results if provided.
		"""
		
		# transform test data and create empty prediction matrix
		X_test = self.transformer_x.transform(df_test['document'])

		# get model prediction
		test_pred = self.model.predict_proba(X_test)
		
		# if parent node extistent, incorporate parent node prediction that corresponds to this node based on a name matching
		if parent_node is not None: 

			parent_column_index = parent_node.sub_classes.index(self.name)
			parent_column = parent_pred[:, parent_column_index]

			test_pred = test_pred * parent_column[:, np.newaxis]
		
		return test_pred
		
def get_classification_node(name, level, df_raw):
	"""Create a ParentNode object
	
	This method creates a ParentNode object that carries all revelant information, transformer objects and all trained sub models.
	
	Parameters
	----------
	name : string
		Name of the resulting parent node

	level : integer
		Hierarchy level of the resulting parent node which is required to cut the training data labels to the correct length
	
	df_raw : pandas DataFrame
		Raw training data without any level adjustments of labels or vectorization applied
	
	Returns
	-------
	node_model : ParentNode
		Resulting parent node with trained model
	"""
	
	# get training data
	df_train = get_level_data(df_raw, level, name)
	# shuffle training data
	df_train = df_train.sample(frac=1)
	
	input_dim = min(3000, df_train.shape[0])
	
	# process data
	X_train = df_train['document']
	
	count_vec = CountVectorizer()
	tfidf = TfidfTransformer()
	svd = TruncatedSVD(n_components = input_dim)
	
	X_train_counts = count_vec.fit_transform(X_train)
	X_train_tfidf = tfidf.fit_transform(X_train_counts)
	X_train_svd = svd.fit_transform(X_train_tfidf)
	
	# save Transfomer Pipeline for applying it on test data later
	transformer_x = TransformerPipeline([count_vec, tfidf, svd])
	
	# vectorize target values
	count_vec_y = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None, token_pattern = r"(?u)\b\w+\b")
	y_train = count_vec_y.fit_transform(df_train['ipcs'])
	y_target_names = count_vec_y.get_feature_names()
	
	# define & train model
	model = Sequential()
	model.add(Dense(600, input_dim = input_dim, activation='relu'))
	model.add(Dropout(0.2))
	model.add(Dense(600, activation='relu'))
	model.add(Dropout(0.2))
	model.add(Dense(len(y_target_names), activation='sigmoid'))

	model.compile(optimizer=Adam(0.015), loss='binary_crossentropy', metrics=['categorical_accuracy'])
	
	if name == None:
		name = "Seed"
	
	# define callbacks	
	callbacks = [
	    ReduceLROnPlateau(),
	    EarlyStopping(patience=7)
	]

	# fit model with provided training parameters
	history = model.fit(X_train_svd, y_train,
	                    epochs=15,
	                    batch_size=32,
	                    validation_split=0.1,
	                    callbacks=callbacks)
						

	# create ParentNode object with processed information and model
	parent_node = ParentNode(name, level, model, transformer_x, count_vec_y, y_target_names)
	
	return parent_node