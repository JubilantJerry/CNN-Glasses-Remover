import numpy as np
import sys, time
import data_organizer, cnn

def mean_loss(neural_net, has_mask, pairs, decoder):
	num_pairs = len(pairs)
	acc = 0
	for pair in pairs:
		(input_img_temp, output_img) = decoder(pair)
		if (has_mask):
			input_img = input_img_temp
		else:
			input_img = input_img_temp[:-1]
		acc += neural_net.get_loss(input_img, output_img)
	return acc / num_pairs

def store_nn(neural_net, storage_path):
	kwds = neural_net.export_info()
	np.savez(storage_path, **kwds)

def train_nn(neural_net, has_mask, rng_seed, training, validation, decoder,
			 storage_path, max_epochs=0, max_stagnation=0, eta_list=[], wait=0):
	rng = np.random.RandomState(rng_seed + 1)
	stagnation = 0
	epochs = 0
	prev_loss = mean_loss(neural_net, has_mask, validation, decoder)
	min_loss = prev_loss
	
	print("Initial loss = " + str(prev_loss))

	batch_size = neural_net.input_shp[0]
	num_batches_f = int(np.floor(len(training) / batch_size))
	num_batches_c = int(np.ceil(len(training) / batch_size))
	extra = num_batches_c - num_batches_f

	while ((not max_stagnation or stagnation < max_stagnation) and
		   (not max_epochs or epochs < max_epochs)):
		print("Epoch " + str(epochs) + " ", end='')
		dots = 0
		batches = np.empty((num_batches_c, batch_size, 2), dtype=object)
		batches[:num_batches_f] = training[rng.choice(
					range(len(training)), size=(num_batches_f, batch_size),
					replace=True)]
		batches[num_batches_f:] = training[rng.choice(
					range(len(training)), size=(extra, batch_size),
					replace=True)]

		input_batch = np.zeros(neural_net.input_shp, dtype='float32')
		output_batch = np.zeros(neural_net.output_shp, dtype='float32')
		for batch_num in range(num_batches_c):
			batch = batches[batch_num]
			for i in range(batch_size):
				(input_img, output_batch[i]) = decoder(batch[i])
				if (has_mask):
					input_batch[i] = input_img
				else:
					input_batch[i] = input_img[:-1]
			neural_net.train_model(input_batch, output_batch)
			time.sleep(wait / 1000.0)
			num_newdots = ((10 * (batch_num + 1)) // num_batches_c) - dots
			print('.' * num_newdots, end='')
			sys.stdout.flush()
			dots += num_newdots
		print('.' * (10 - dots), end='')
		sys.stdout.flush()

		if (eta_list):
			neural_net.eta.set_value(eta_list[0])
			eta_list=eta_list[1:]

		stored_msg = ""
		loss = mean_loss(neural_net, has_mask, validation, decoder)
		if (loss < min_loss):
			min_loss = loss
			store_nn(neural_net, storage_path)
			stored_msg = "\n\tNeural network stored"
		if (prev_loss < loss):
			stagnation += 1
		else:
			stagnation = 0
		prev_loss = loss

		print("\n\tVerification loss = " + str(loss) +
			  "\n\tStagnation = " + str(stagnation) + stored_msg)
		epochs += 1
