import lstm
import os, sys, time
import argparse
import matplotlib.pyplot as plt

def plot_results(predicted_data, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()

def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    #Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label='Prediction')
        plt.legend()
    plt.show()

#Main Run Thread
if __name__=='__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--csv', help='either sp500.csv or sinwave.csv', default='sinwave.csv')
	parser.add_argument('--split', help='ratio of training to testing data', type=float, default=0.9)
	args = parser.parse_args()

	global_start_time = time.time()
	epochs  = 1
	seq_len = 50

	if os.path.exists(args.csv):
		print('> Loading %s' % args.csv)
	else:
		print('> Error, %s does not exist' % args.csv)
		sys.exit(1)

	data_loader_start_time = time.time()

	if args.csv == 'sp500.csv':
		X_train, y_train, X_test, y_test = lstm.load_data(args.csv, seq_len, args.split, True)
	else:
		X_train, y_train, X_test, y_test = lstm.load_data(args.csv, seq_len, args.split, False)

	print("> Data Loading Time : ", time.time() - data_loader_start_time)

	model = lstm.build_model([1, 50, 100, 1])
	model.fit(
	    X_train,
	    y_train,
	    batch_size=512,
	    nb_epoch=epochs,
	    validation_data=(X_test, y_test))
	
	print('Training duration (s) : ', time.time() - global_start_time)

	#predictions = lstm.predict_sequences_multiple(model, X_test, seq_len, 50)
	#plot_results_multiple(predictions, y_test, 50)

	#predicted = lstm.predict_sequence_full(model, X_test, seq_len)
	predicted = lstm.predict_point_by_point(model, X_test)        
	plot_results(predicted, y_test)