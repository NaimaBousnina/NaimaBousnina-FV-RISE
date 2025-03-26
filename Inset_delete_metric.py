# coding: utf-8

import argparse
import torch
from helper import get_eval_model
from helper import causalMetric_SimHM
from helper import generate_HMs


parser = argparse.ArgumentParser(description='Explain the ArcFace model final decision')

parser.add_argument('--model-prefix', default='', help='path to load model.')
parser.add_argument('--network', default='', type=str, help='')

args = parser.parse_args()

model_path = args.model_prefix
network = args.network

model = get_eval_model(model_path, network)


### Load the pairs of images
df = pd.read_excel('Pairs.xlsx')
probe_samples = df['probe_imgs']
gallery_samples = df['gallery_imgs']


heat_maps_list, probe_imgs_list, gallery_imgs_list = generate_HMs(model, probe_samples, gallery_samples)

# Generate similarity score for the insertion and deletion process 
scores_del, scores_inser, percen_del, percen_inser = causalMetric_SimHM(model, probe_imgs_list, gallery_imgs_list, heat_maps_list, nb_thresholds= 10)
