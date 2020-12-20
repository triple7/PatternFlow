from utilities.dataSet import Dataset, DataCollate
from model.discriminator import define_discriminator
from model.generator import define_generator
from utilities.loss import multi_res_stft_loss
from utilities.audio import HOP_LENGTH
from utilities.optimiser import DecayOptimizer
import os
import time
import argparse
from keras import Model

def create_mode(args):
	generator = Generator(args.condition_dims, args.z_dim)
	discriminator = Discriminator()
	return generator, discriminator

def train(args):
	os.makedirs(args.progress, exist_ok = True)
	
	train_data = DataSet(args.input, upsample_factor = HOP_LENGTH)
	generator, discriminator = create_model(args)
	generator.summary
	discriminator.summary
	model = Model()
	