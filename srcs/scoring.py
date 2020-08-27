import numpy as np

from sklearn.metrics import f1_score, precision_score, recall_score


def print_f1_score(true_y, predicted_y, avg_type):
	"""Generate and print F1-Score
	
	Parameters
	----------
	true_y : numpy array
		True y data
	
	predicted_y : numpy array
		Predicted y data
	
	avg_type : String
		Average type used for F1-Score calculcation with multiple classes
	"""
	
	print("F1-Score:", f1_score(true_y, predicted_y, average=avg_type))
	print("Precision:", precision_score(true_y, predicted_y, average=avg_type))
	print("Recall:", recall_score(true_y, predicted_y, average=avg_type))
	
def print_accuracy_scores_top_ipc(true_y, predicted_y):
	"""Generate and print Top Accuracy & Three Guesses Accuracy
	
	Parameters
	----------
	true_y : numpy array
		True y data
	
	predicted_y : numpy array
		Predicted y data
	"""
	
	top_predicted = predicted_y.argmax(axis = 1)
	top_predicted = top_predicted.reshape(top_predicted.shape[0],1)
	top_true = true_y.argmax(axis = 1)
	top_three_predicted = np.argpartition(predicted_y, -3, axis=1)[:, -3:]
	
	three_match_sum = 0
	for index, row in enumerate(true_y):
		three_match = (top_true[index][0] in top_three_predicted[index])
		three_match_sum += three_match
		
	three_acc = float(three_match_sum) / true_y.shape[0]
	
	print("Acc. Top:", (np.mean(top_predicted == top_true)))
	print("Acc. Three:", three_acc)
	
def print_accuracy_score_all_ipcs(true_y, predicted_y):
	"""Generate and print All IPCs Accuracy
	
	Parameters
	----------
	true_y : numpy array
		True y data
	
	predicted_y : numpy array
		Predicted y data
	"""
	
	top_match_sum = 0
	top_predicted = predicted_y.argmax(axis = 1)
	top_predicted = top_predicted.reshape(top_predicted.shape[0],1)
	
	true_y = true_y.toarray()
	
	for index, row in enumerate(true_y):
		ipc_indices = np.where(row == 1)
		top_match = (np.any(ipc_indices == top_predicted[index][0]))
		top_match_sum += top_match
		
	all_acc = float(top_match_sum) / true_y.shape[0]
	
	print("Acc. All:", all_acc)