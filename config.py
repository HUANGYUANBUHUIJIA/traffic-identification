# -*- coding: utf-8 -*-

"""
config file

"""
import os

# if load trained model
load_model = False

# root of train data and test data
train_data_dir = './data/Training'
test_data_dir = './data/Testing'

# reserve root
output_path = './output'
if not os.path.exists(output_path):
    os.makedirs(output_path)

model_file = os.path.join(output_path, 'trained_cnn_model.h5')

# classes number
n_classes = 62

# img size
img_rows, img_cols = 32, 32

# batch size
batch_size = 32

# iterat number
epochs = 100

# if augmentation for sample
data_augmentation = True
