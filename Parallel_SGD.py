# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

# import the necessary packages
from sklearn.utils import shuffle
import numpy as np
import pandas as pd
import argparse
from mpi4py import MPI
import os
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
os.environ['OPENBLAS_NUM_THREADS'] = '1'

def read_KDD():
	names = ["duration","protocol_type","service","flag","src_bytes",
	    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
	    "logged_in","num_compromised","root_shell","su_attempted","num_root",
	    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
	    "is_host_login","is_guest_login","count","srv_count","serror_rate",
	    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
	    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
	    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
	    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
	    "dst_host_rerror_rate","dst_host_srv_rerror_rate","label"]

	dataframe = pd.read_csv("kddcup.data_10_percent_corrected", names = names, header = None)
	dataframe["label"] = dataframe["label"].str.rstrip(".") # remove "." at the end of all labels in label column

	for x, item in enumerate(dataframe["label"]):
		if item == "normal":
			dataframe.at[x, "label"] = 0
		else:
			dataframe.at[x, "label"] = 1
	dataframe = shuffle(dataframe)
	y = dataframe["label"]

	data1 = pd.get_dummies(dataframe, drop_first = True)
	data1 = data1.iloc[:, :-1]

	X = data1.to_numpy()
	y = y.to_numpy()

	return X, y

def next_batch(batch_iteration, X, y, batchSize):
	return (X[int(batch_iteration):int(batch_iteration + batchSize)], y[int(batch_iteration):int(batch_iteration + batchSize)])

def sigmoid_activation(x):
	# compute and return the sigmoid activation value for a
	# given input value
	np.seterr(over = 'ignore')
	z = np.array(x, dtype = np.float32)
	return 1.0 / (1 + np.exp(-z))

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--epochs", type=float, default = 100,
	help="# of epochs")
ap.add_argument("-a", "--alpha", type=float, default = 0.01,
	help="learning rate")
ap.add_argument("-b", "--batch-size", type=int, default = 256,
	help="size of SGD mini-batches")
ap.add_argument("-f", "--update_frequency", type=int, default = 25,
	help="parameter server update frequency")
args = vars(ap.parse_args())

if(rank == 0):
	(X, y) = read_KDD()
	X = np.c_[np.ones((X.shape[0])), X]

	print("[INFO] starting training...")
	W = np.random.uniform(size=(X.shape[1],))
	for i in range(1, size):
		comm.send(W, dest = i, tag = 1)

	batches_generated = 0
	start = time.time()
	step = 0
	for epoch in np.arange(0, args["epochs"]):
		if(step % args["update_frequency"] == 0):
			for i in range(1, size):
				comm.send(W, dest = i, tag = 2)
		(batchX, batchY) = next_batch(batches_generated, X, y, args["batch_size"])
		batches_generated += 1
		mini_batches_generated = 0
		for i in range(1, size):
			(minibatchX, minibatchY) = next_batch(mini_batches_generated, batchX, batchY, args["batch_size"]/(size - 1))
			mini_batches_generated += 1
			comm.send(minibatchX, dest = i, tag = 3)
			comm.send(minibatchY, dest = i, tag = 4)
		if(step % args["update_frequency"] == 0):
			for i in range(1 , size):
				gradient = comm.recv(source = i, tag = 5)
				# use the gradient computed on the current batch to take
				# a "step" in the correct direction
				W = W + (-args["alpha"] * gradient)
		step += 1

	end = time.time()
	training_time = end - start
	print("Training time: " + str(training_time) + " seconds")
else:
	step = 0															
	accrued_gradients = 0
	W = comm.recv(source = 0, tag = 1)

	for epoch in np.arange(0, args["epochs"]):
		if(step % args["update_frequency"] == 0):
			W = comm.recv(source = 0, tag = 2)

		batchX = comm.recv(source = 0, tag = 3)										# receive mini batch of data from parameter server
		batchY = comm.recv(source = 0, tag = 4)										# receive data lebels from parameter server

		prediction = sigmoid_activation(batchX.dot(W))								# make predictions for mini batch data								
					
		error = prediction - batchY													# calculate error
		gradient = (batchX.T.dot(error)) / batchX.shape[0]							# calculate gradient
		accrued_gradients += gradient
		W = W + (-args["alpha"] * gradient)
		
		if(step % args["update_frequency"] == 0):
			comm.send(accrued_gradients, dest = 0, tag = 5)							# send gradient to parameter server
			accrued_gradients = 0
		step += 1