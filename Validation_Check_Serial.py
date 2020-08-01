# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

# import the necessary packages
from sklearn.linear_model import SGDRegressor
from sklearn.utils import shuffle
import numpy as np
import pandas as pd
import argparse
import time
import os
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

def sigmoid_activation(x):
	# compute and return the sigmoid activation value for a
	# given input value
	np.seterr(over = 'ignore')
	z = np.array(x, dtype = np.float32)
	return 1.0 / (1 + np.exp(-z))

def next_batch(batch_iteration, X, y, batchSize):
	return (X[batch_iteration:batch_iteration + batchSize], y[batch_iteration:batch_iteration + batchSize])

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--epochs", type=float, default=200,
	help="# of epochs")
ap.add_argument("-a", "--alpha", type=float, default=0.01,
	help="learning rate")
ap.add_argument("-b", "--batch-size", type=int, default=64,
	help="size of SGD mini-batches")
args = vars(ap.parse_args())

(X, y) = read_KDD()
# insert a column of 1's as the first entry in the feature
# vector -- this is a little trick that allows us to treat
# the bias as a trainable parameter *within* the weight matrix
# rather than an entirely separate variable
X = np.c_[np.ones((X.shape[0])), X]
# initialize our weight matrix such it has the same number of
# columns as our input features
print("[INFO] starting training...")
W = np.random.uniform(size=(X.shape[1],))

batches_generated = 0
# loop over the desired number of epochs
for epoch in np.arange(0, args["epochs"]):
	# loop over our data in batches
	(batchX, batchY) = next_batch(batches_generated, X, y, args["batch_size"])
	batches_generated += 1
	# take the dot product between our current batch of
	# features and weight matrix `W`, then pass this value
	# through the sigmoid activation function
	prediction = sigmoid_activation(batchX.dot(W))
	# now that we have our predictions, we need to determine
	# our `error`, which is the difference between our predictions
	# and the true values
	error = prediction - batchY
	# the gradient update is therefore the dot product between
	# the transpose of our current batch and the error on the
	# # batch
	gradient = batchX.T.dot(error) / batchX.shape[0]
	# use the gradient computed on the current batch to take
	# a "step" in the correct direction
	W = W + (-args["alpha"] * gradient)

W_Serial = W

y = y.astype('int')

# SkLearn SGD classifier
SGD = SGDRegressor(loss = "huber", alpha = args["alpha"], max_iter = args["epochs"])
SGD.fit(X, y)
W_SGD = (SGD.coef_).transpose()
accuracy_matrix = np.absolute(W_Serial - W_SGD)
print(accuracy_matrix)