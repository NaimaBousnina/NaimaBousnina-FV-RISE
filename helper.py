# coding: utf-8

import torch
import cv2

import torch.nn.functional as F
import numpy as np
import matplotlib as matplotlib
import matplotlib.cm as cm

from backbones import get_model
from itertools import groupby
from torchvision import transforms
from config import device
from PIL import Image
import torch.nn as nn


data_transforms = {
	'val': transforms.Compose([
		transforms.ToTensor()
	])
}

def key_func(k):
	return k['Subj_id']

def images_grouping(imag_samples):

	# sort samples data by subjet ID key.
	samples = sorted(imag_samples, key=key_func)

	Img_roups = []
	for key, value in groupby(samples, key_func):
		try:
			Img_roups.append(list(value))
		except KeyboardInterrupt:
			raise
		except Exception as err:
			print(err)

	return Img_roups

def get_eval_model(model_path, network):
	
	# Get the pretrained model.
	weight = torch.load(model_path)
	resnet = get_model(network, dropout=0.5, fp16=False).cuda()
	resnet.load_state_dict(weight)
	model = torch.nn.DataParallel(resnet)
	model.eval()
	#summary(model, (3, 112, 112))
	#print(model)

	return model


def generate_masks(n_masks=5, p1= 0.5, image_size=(112, 112), initial_mask_size=(7, 7)):

	# cell size in the upsampled mask
	Ch = np.ceil(image_size[0] / initial_mask_size[0])
	Cw = np.ceil(image_size[1] / initial_mask_size[1])

	resize_h = int((initial_mask_size[0] + 1) * Ch)
	resize_w = int((initial_mask_size[1] + 1) * Cw)

	masks = []

	for _ in range(n_masks):

		# generate binary mask
		binary_mask = torch.randn(1, 1, initial_mask_size[0], initial_mask_size[1])
		binary_mask = (binary_mask < p1).float()

		# upsampling mask
		mask = F.interpolate(binary_mask, (resize_h, resize_w), mode='bilinear', align_corners=False)

		# random cropping
		i = np.random.randint(0, Ch)
		j = np.random.randint(0, Cw)
		mask = mask[:, :, i:i+image_size[0], j:j+image_size[1]]

		masks.append(mask)

	masks = torch.cat(masks, dim=0)   # (N_masks, 1, H, W)

	return masks

def masked_probe_images(probe_image, masks):

	masked_imgs = []
	for i in range(masks.shape[0]):
		masked = torch.mul(masks[i], probe_image)
		masked_imgs.append(masked)

	return masked_imgs


def get_image(full_path, transformer):
	
	size_ = (112, 112)
	img = cv2.imread(full_path, 1) 

	face_img = cv2.resize(img, (size_[0], size_[1]))

	face_img = face_img[..., ::-1]  # RGB
	face_img = Image.fromarray(face_img, 'RGB')  # RGB
	face_img = transformer(face_img)
	face_img = face_img.to(device)
	return face_img 

def read_image(imag_samples):

    transformer = data_transforms['val']
    with torch.no_grad():
        img = torch.zeros([1, 3, 112, 112], dtype=torch.float, device=device)
        img_ = get_image(imag_samples, transformer)
        img[0] = img_
    return img[0]

def cosine_similarity(probe_descriptions, gallery_descriptions):

	x0 = probe_descriptions / np.linalg.norm(probe_descriptions)
	x1 = gallery_descriptions / np.linalg.norm(gallery_descriptions)
	similarity_score = np.sum(x0 * x1, -1)

	return similarity_score


def score_difference(reference_score, masked_score):

	dif = abs(reference_score - masked_score)

	return dif		

def dist_difference(reference_score, masked_score):

	dif = abs(reference_score - masked_score)

	return dif


def causalMetric_SimHM(model, probe_imgs_list, gallery_imgs_list, sim_HM, nb_thresholds= 10):

	list_scores_del = []
	list_scores_inser = []

	percentage_del = []
	percentage_inser = []
	for l in range(nb_thresholds + 1):
		print('nb_thresholds:', nb_thresholds)

		p_delition = l/nb_thresholds*100
		p_insertion = (100 - p_delition)

		percentage_del.append(p_delition)
		percentage_inser.append(p_insertion)
		
		scores_del = []
		scores_inser = []

		for i in range(len(probe_imgs_list)):
			for j in range(len(probe_imgs_list[i])):

				print('Subject:', i)

				# Visualize the heat map
				#import_heat_map = np.uint8(255 * sim_HM[i][j])
				#import_heat_map = cv2.applyColorMap(import_heat_map, cv2.COLORMAP_JET)
				#plt.imshow(import_heat_map)
				#plt.show()

				# Extract the face descriptions of the gallery
				gallery_desc = model(gallery_imgs_list[i][j]).cpu().detach().numpy()[0]

								### Delition Process ###
				
				k_delition = p_delition*112*112/100 # the percentage of pixels to be hidden

				# Generate the indices of top k_delition pixels
				reshaped_sim_HM_del = sim_HM[i][j].reshape(sim_HM[i][j].shape[0]*sim_HM[i][j].shape[1])
				_, indice_delition = torch.topk(torch.from_numpy(reshaped_sim_HM_del), int(k_delition), largest=False)
				indice_delition = indice_delition.detach().cpu().numpy()

				# Mask the top k_delition imprtoant face regions
				probe_img = probe_imgs_list[i][j].detach().cpu().numpy()
				reshaped_probe_img_del = probe_img.reshape(3, probe_img.shape[1]*probe_img.shape[2])
				reshaped_probe_img_del[:, indice_delition] = 0

				# Reshape the face image back to the original dimensions (Delition)
				masked_img_del = reshaped_probe_img_del.reshape(3, probe_img.shape[1],probe_img.shape[2])

				# Visualize the masked face image
				#masked_img_del_show = masked_img_del.transpose((1,2,0))
				#masked_img_del_show = np.uint8(255 * masked_img_del_show)
				#plt.imshow(masked_img_del_show)
				#plt.show()

				# Extract the face descriptions of masked probe image (deletion)
				masked_probe_img_del = torch.from_numpy(masked_img_del).unsqueeze(0)
				maskes_probe_desc_del = model(masked_probe_img_del).cpu().detach().numpy()[0]

				# Similarity score and cosine distance (delition)
				simil_score_del = cosine_similarity(maskes_probe_desc_del, gallery_desc)
				#cosine_dist_del = 1 - (simil_score_del + 1)/2

				scores_del.append(simil_score_del)


									### Insertion Process ###
				k_insertion = p_insertion*112*112/100 # the percentage of pixels to be shown

				# Generate the indices of top k_insertion pixels
				reshaped_sim_HM_inser = sim_HM[i][j].reshape(sim_HM[i][j].shape[0]*sim_HM[i][j].shape[1])
				_, indice_insertion = torch.topk(torch.from_numpy(reshaped_sim_HM_inser), int(k_insertion), largest=True)
				indice_insertion = indice_insertion.detach().cpu().numpy()

				# Mask the top less k_insertion imprtoant face regions
				probe_img = probe_imgs_list[i][j].detach().cpu().numpy()
				reshaped_probe_img_inser = probe_img.reshape(3, probe_img.shape[1]*probe_img.shape[2])
				reshaped_probe_img_inser[:, indice_insertion] = 0
				
				# Reshape the face image back to the original dimensions (Insertion)
				masked_img_inser = reshaped_probe_img_inser.reshape(3, probe_img.shape[1],probe_img.shape[2])
				
				# Visualize the masked face image
				#masked_img_inser_show = masked_img_inser.transpose((1,2,0))
				#masked_img_inser_show = np.uint8(255 * masked_img_inser_show)
				#plt.imshow(masked_img_inser_show)
				#plt.show()

				# Extract the face descriptions of masked probe image (insertion)
				masked_probe_img_inser = torch.from_numpy(masked_img_inser).unsqueeze(0)
				maskes_probe_desc_inser = model(masked_probe_img_inser).cpu().detach().numpy()[0]

				# Similarity score and cosine distance (insertion)
				simil_score_inser = cosine_similarity(maskes_probe_desc_inser, gallery_desc)
				#cosine_dist_inser = 1 - (simil_score_inser + 1)/2

				scores_inser.append(simil_score_inser)

		list_scores_del.append(scores_del)
		list_scores_inser.append(scores_inser)



	return list_scores_del, list_scores_inser, percentage_del, percentage_inser


def calculate_recall(scores):

	TP = 0
	FN = 0
	threshold = 0.5

	for i in range(len(scores)):

		cosine_dist = 1 - (scores[i] + 1)/2
		if cosine_dist > threshold:
			FN += 1
		else:
			TP += 1

	Recall = float(TP) / float(TP + FN)
	return Recall

def generate_HMs(model, probe_samples, gallery_samples):

	## Generate the masks 
	n_masks = 10000
	p1= 0.1

	heat_maps_list = []
	probe_imgs_list = []
	gallery_imgs_list = []


	for i in range(len(probe_samples)): #loop over the individuals
		print('subject', i)

		## name of the probe image
		probe_name = os.path.basename(probe_samples[i]).rsplit('.', 1)[0]


		# name of the gallery image
		gallery_name = os.path.basename(gallery_samples[i]).rsplit('.', 1)[0]


		## Read the current probe image
		probe_img = read_image(probe_samples[i])
		#print(probe_img.shape)

		## Generate the masks 
		masks = generate_masks(n_masks, p1, image_size=(112, 112), initial_mask_size=(5, 5)).to(device)

		## Generate the masked probe images for the current probe image
		masked_imgs = masked_probe_images(probe_img, masks)
		#print(len(masked_imgs))

		# Extract the face descriptions of the gallery image
		galerry_img = read_image(gallery_samples[i])
		galerry_img = galerry_img.unsqueeze(0)
		gallery_descriptions = model(galerry_img).cpu().detach().numpy()[0]

		# Extract the face descriptions of the reference probe image
		refe_probe_img = probe_img.unsqueeze(0)
		refe_probe_descriptions = model(refe_probe_img).cpu().detach().numpy()[0]

		ref_similarity = cosine_similarity(refe_probe_descriptions, gallery_descriptions)
		#print(ref_similarity)


		import_weighted_sum = []


		for n in range(len(masked_imgs)): # loop over the masked probe images

            # Extract the face descriptions of the masked probe image
			masked_image = masked_imgs[n].unsqueeze(0)
			probe_descriptions = model(masked_image).cpu().detach().numpy()[0]


			# Compute the similarity score between the descriptions of the gallery and masked probe images
			masked_similarity = cosine_similarity(probe_descriptions, gallery_descriptions)

			# Compute the difference between the masked distance and the reference distance.
			mask_weight = dist_difference(ref_similarity, masked_similarity)

			if masked_similarity < ref_similarity: # This means that the occluded face region important
				t1 = torch.Tensor([mask_weight])
				masks[n] = torch.mul(masks[n], t1.to(device))
				import_weighted_sum.append(masks[n])


		if len(import_weighted_sum)==0:
			import_sum = torch.zeros([1, 112, 112], dtype=torch.float, device=device)
		else:
			import_sum = torch.cat(import_weighted_sum, dim=0)

		import_sum = torch.mean(import_sum, axis=0).unsqueeze(0)



        ####                                                         ####
        #### Generate face heat maps for the important face regions  ####
        ####                                                         ####

        numer = import_sum - torch.min(import_sum)
        denom = (import_sum.max() - import_sum.min())
        import_sum = numer / denom
        import_sum /= torch.max(import_sum)

       # Genrate the heat map
       import_heat_map = import_sum.cpu().numpy().transpose((1,2,0))

       heat_maps_list.append(import_heat_map)
       probe_imgs_list.append(probe_img)
       gallery_imgs_list.append(galerry_img)


    return heat_maps_list, probe_imgs_list, gallery_imgs_list
