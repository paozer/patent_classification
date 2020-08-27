from data_model_handling import import_data, get_level_data, TransformerPipeline

import os
import pandas as pd
import numpy as np
import math
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler
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
	
	This class represents a parent node in the classification hierarchy of a Local Classifier per Node (LCN) 
	concept based on the definition of Silla and Freitas [1].
	Besides carrying the name and level of the respective node, it serves as a container to save the data transformers 
	for X and y data that are fit on training data and ready to be applied on test data.
	Furthermore, it carries the name of all sub classes, the respective binary child models and the respective child 
	nodes (which are a ParentNode again). This design which allows to iteratively build a hierarchy from top to bottom. 
	
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
	
	def __init__(self, name, level, transformer_x, count_vec_y, sub_classes):
		self.name = name
		self.level = level
		self.transformer_x = transformer_x
		self.count_vec_y = count_vec_y
		self.sub_classes = sub_classes
		self.child_models = []
		self.child_nodes = []
		
	def add_child_model(self, child_model):
		"""Add a child model classifier to the child_models list
	
		Parameters
		----------
		child_model : ChildModel
			An object of the class ChildModel
		"""
		
		
		self.child_models.append(child_model)
		
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
		
		After transforming the input data, this method iterates through the list of all child models (binary classifiers) and calls their predict-method which returns 
		a one-column numpy-array with a prediction regarding their belonging to the respective subclass. Subsequently, all predictions
		are aggregated on the parent node level which results in a multi-column numpy array. Furthermore, if a parent_node and parent_pred
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
		test_pred = np.zeros(shape=(X_test.shape[0], len(self.child_models)))
		
		# iterate through all binary child_models and add their prediction to the prediction matrix
		for index, cm in enumerate(self.child_models):
		
			cm_test_pred = cm.model.predict_proba(X_test)
			test_pred[:, index] = cm_test_pred[:, 1]
		
		# if parent node extistent, incorporate parent node prediction that corresponds to this node based on a name matching
		if parent_node is not None:
			parent_column_index = parent_node.sub_classes.index(self.name)
			parent_column = parent_pred[:, parent_column_index]

			test_pred = test_pred * parent_column[:, np.newaxis]
		
		return test_pred
		
class ChildModel:
	"""ChildModel
	
	This class simply serves as container for a binary classifier model, that is used to generate predictions regarding one subclass
	of the respective ParentNode.
	
	Parameters
	----------
	name : string
		Name of the subclass the model represents
	
	level : integer
		Level of the parent node within the hierarchy
	
	model : classifier object
		Classifier object that is trained and executes the actual binary prediction
	"""
	
	
	def __init__(self, name, level, model):
		self.name = name
		self.level = level
		self.model = model
		

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
	parent_node: ParentNode
		Resulting parent node with all trained child models
	"""
	
	
	# get training data
	df_train = get_level_data(df_raw, level, name)
	# shuffle training data
	df_train = df_train.sample(frac=1)
	
	input_dim = min(300, df_train.shape[0])
	
	
	X_train = df_train['document']
	
	# process data
	count_vec = CountVectorizer()
	tfidf = TfidfTransformer()
	svd = TruncatedSVD(n_components = input_dim)
	
	X_train_counts = count_vec.fit_transform(X_train)
	X_train_tfidf = tfidf.fit_transform(X_train_counts)
	X_train_svd = svd.fit_transform(X_train_tfidf)
	
	# save transfomer oipeline for applying it on test data later
	transformer_x = TransformerPipeline([count_vec, tfidf, svd])
	
	# vectorize target values
	count_vec_y = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None, token_pattern = r"(?u)\b\w+\b")
	y_train = count_vec_y.fit_transform(df_train['ipcs'])
	y_target_names = count_vec_y.get_feature_names()
	
	# create ParentNode object with processed information
	parent_node = ParentNode(name, level, transformer_x, count_vec_y, y_target_names)
	
	# create a binary ChildModel object for every name in y_target_names
	for index, target in enumerate(y_target_names):
		
		print("Target:", target)
		y_train_child = y_train[:,index].copy()
		
		# define used model
		# for usage of alternative model check the support of the predict_proba method --> if not supported upper code has to be adjusted accordingly
		model = SGDClassifier(loss='log', class_weight='balanced', learning_rate = 'optimal', penalty='l2')
		#model = SGDClassifier(loss='hinge', class_weight='balanced')
		
		# reshape data
		X_train_svd = X_train_svd.reshape(X_train_svd.shape[0], input_dim)
		y_train_child = y_train_child.toarray().reshape(y_train_child.shape[0],)
		
		# fit model and create ChildModel object
		model.fit(X_train_svd, y_train_child)
		child_model = ChildModel(target, (level + 1), model)
		
		# add ChildModel object to parent node
		parent_node.add_child_model(child_model)		
	
	return parent_node