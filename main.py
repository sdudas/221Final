from preprocess import *
from basicClassifier import *

def main():
	data = PreProcess("data/compas-scores-two-years.csv")
	X, Y = data.read_in_data()
	X_train, X_test, Y_train, Y_test = data.split_train_test(X, Y)
	classifier = Classifier()

	sets = {}
	sets['x_train'] = X_train
	sets['y_train'] = Y_train
	sets['x_test'] = X_test
	sets['y_test'] = Y_test
	model = classifier.create_simple_classifier(sets, X_train.shape[1])
	# history = model.fit(X_train.values, Y_train.values, validation_data=(X_test.values, Y_test.values), epochs=20, verbose=1)

main()