import numpy as np

def train_test_split(X, y):
	X = X.tolist()
	y = y.tolist()
	X_train, X_test, y_train, y_test = [], [], [], []
	class_start, class_end = 0, 0
	curr_class = y[0]
	rest_counter = 0
	rest_train, rest_test, rest_y_train, rest_y_test = [], [], [], []
	num_classes = {0: 0, 1:0, 2:0, 3:0, 4:0}
	while (class_start < len(y)) :
		curr_class = y[class_start]
		while (class_end < len(y) and y[class_end] == curr_class):
			class_end += 1
		class_length = class_end - class_start
		cutoff = class_start + int(class_length*0.8)
		if curr_class != 0 or rest_counter % 4 == 0:
			num_classes[curr_class] += class_length
			X_train += X[class_start:cutoff]
			y_train += y[class_start:cutoff]
			X_test += X[cutoff:class_end]
			y_test += y[cutoff:class_end]
		if curr_class == 0:
			rest_counter += 1
		class_start = class_end

	for c, num in num_classes.items():
		print(f'Class {c}: {num}')

	return np.asarray(X_train), np.asarray(X_test), np.asarray(y_train), np.asarray(y_test)