from model import *
from utils import *
from transformers import BertForSequenceClassification
import os

output_dir = './bert_model_save/'

def test(test_loader):
	device = torch.device("cuda")

	print('[II] Start testing')
	
	model = BertForSequenceClassification.from_pretrained(output_dir).cuda()
	model.eval()

	# Tracking variables 
	nb_steps, accuracy = 0, 0
	all_logits = []

	# Predict 
	for batch in test_loader:
		# Add batch to GPU
		b_input_ids = batch[0].to(device)
		b_input_mask = batch[1].to(device)
		b_labels = batch[2].to(device)
		b_lens = batch[3]
		
		# Telling the model not to compute or store gradients, saving memory and 
		# speeding up prediction
		with torch.no_grad():
			# Forward pass, calculate logit predictions
			outputs = model(b_input_ids, token_type_ids=None, 
							attention_mask=b_input_mask)
		
		logits = outputs[0]
		
		# Move logits and labels to CPU
		logits = logits.detach().cpu().numpy()
		label_ids = b_labels.to('cpu').numpy()
		
		# Calculate the accuracy for this batch of test sentences.
		tmp_accuracy = flat_accuracy(logits, label_ids)
		
		# Accumulate the total accuracy.
		accuracy += tmp_accuracy

		# Track the number of batches
		nb_steps += 1

		all_logits.append(logits)
	
	print('[II]  Accuracy: {0:.2f}'.format(accuracy/nb_steps))
	return all_logits
