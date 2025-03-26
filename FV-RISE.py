# coding: utf-8

import os
import argparse
import torch
import cv2

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from config import device
from PIL import Image
from helper import get_eval_model
from helper import generate_masks
from helper import masked_probe_images
from helper import cosine_similarity
from helper import dist_difference
from helper import read_image


parser = argparse.ArgumentParser(description='Explain the ArcFace model final decision')

parser.add_argument('--model-prefix', default='', help='path to load model.')
parser.add_argument('--network', default='', type=str, help='')

args = parser.parse_args()

model_path = args.model_prefix
network = args.network

### Load and evaluate the Arface model
model = get_eval_model(model_path, network)


### Load the pairs of images
df = pd.read_excel('Pairs.xlsx')
probe_samples = df['probe_imgs']
gallery_samples = df['gallery_imgs']


##############################################
### The start of the FV-RISE-EX process... ###
##############################################

## Number of masks to be generated
n_masks = 10000
p1= 0.1

for i in range(len(probe_samples)): 

    print('Pair number:', i)

    ## name of the probe image
    probe_name = os.path.basename(probe_samples[i]).rsplit('.', 1)[0]

    # name of the gallery image
    gallery_name = os.path.basename(gallery_samples[i]).rsplit('.', 1)[0]

    ## Read the current probe image
    probe_img = read_image(probe_samples[i])
    #print(probe_img.shape)

    ## Generate the masks 
    masks = generate_masks(n_masks, p1, image_size=(112, 112), initial_mask_size=(5, 5)).to(device)
    #print(masks.shape)
    #plt.imshow(masks[0][0].cpu().detach().numpy(), cmap='gray')
    #plt.show()

    ## Generate the masked probe images for the current probe image
    masked_imgs = masked_probe_images(probe_img, masks)
    #print(len(masked_imgs))

    ## Show the masked probe images
    #for k in range(len(masked_imgs)):
    #    masked_ = masked_imgs[k].cpu().numpy().transpose((1,2,0))
    #    plt.imshow(masked_)
    #    plt.show()

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
    Not_import_weighted_sum = []

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

        if masked_similarity > ref_similarity: # This means that the occluded face region is not important

            t2 = torch.Tensor([mask_weight])
            masks[n] = torch.mul(masks[n], t2.to(device))
            Not_import_weighted_sum.append(masks[n])

    if len(import_weighted_sum)==0:

        import_sum = torch.zeros([1, 112, 112], dtype=torch.float, device=device)
    else:

        import_sum = torch.cat(import_weighted_sum, dim=0)

    import_sum = torch.mean(import_sum, axis=0).unsqueeze(0)

    if len(Not_import_weighted_sum)==0:

        Not_import_sum = torch.zeros([1, 112, 112], dtype=torch.float, device=device)
    else:

        Not_import_sum = torch.cat(Not_import_weighted_sum, dim=0)

    Not_import_sum = torch.mean(Not_import_sum, axis=0).unsqueeze(0)


    ####                                                         ####
    #### Generate face heat maps for the important face regions  ####
    ####                                                         ####

    # Reference probe image
    ref_probe_img_1 = probe_img.cpu().numpy().transpose((1,2,0))
    ref_probe_img_1 = np.uint8(255 * ref_probe_img_1)
    #ref_probe_img_1 = cv2.cvtColor(ref_probe_img_1, cv2.COLOR_BGR2RGB)

    numer = import_sum - torch.min(import_sum)
    denom = (import_sum.max() - import_sum.min())
    import_sum = numer / denom
    import_sum /= torch.max(import_sum)

    # Genrate the heat map
    import_heat_map = import_sum.cpu().numpy().transpose((1,2,0))
    import_heat_map = np.uint8(255 * import_heat_map)
    import_heat_map = cv2.applyColorMap(import_heat_map, cv2.COLORMAP_JET)

    # Superimpose the heat map and the reference probe image to generate the face heat map.
    ref_probe_img_1 = Image.fromarray(ref_probe_img_1)
    import_heat_map = Image.fromarray(import_heat_map)

    background_1 = ref_probe_img_1.convert("RGBA")
    import_overlay = import_heat_map.convert("RGBA")
    import_face_heat_map = Image.blend(background_1, import_overlay, 0.5)

    ### Save the face heat maps ###
    path = 'S_HMs' + '/' + probe_name + '_' + gallery_name + '.png'
    import_face_heat_map.save(path)

    # Display of the face heat map ###
    #plt.imshow(import_face_heat_map)
    #plt.show()


    ####                                                             ####
    #### Generate face heat maps for the non-important face regions  ####
    ####                                                             ####

    # Reference probe image
    ref_probe_img_1 = probe_img.cpu().numpy().transpose((1,2,0))
    ref_probe_img_1 = np.uint8(255 * ref_probe_img_1)
    #ref_probe_img_1 = cv2.cvtColor(ref_probe_img_1, cv2.COLOR_BGR2RGB)

    numer = Not_import_sum - torch.min(Not_import_sum)
    denom = (Not_import_sum.max() - Not_import_sum.min())
    Not_import_sum = numer / denom

    Not_import_sum /= torch.max(Not_import_sum)

    # Genrate the heat map
    Not_import_heat_map = Not_import_sum.cpu().numpy().transpose((1,2,0))
    Not_import_heat_map = np.uint8(255 * Not_import_heat_map)
    Not_import_heat_map = cv2.applyColorMap(Not_import_heat_map, cv2.COLORMAP_JET)

    # Superimpose the heat map and the reference probe image to generate the face heat map.
    ref_probe_img_1 = Image.fromarray(ref_probe_img_1)
    Not_import_heat_map = Image.fromarray(Not_import_heat_map)

    background_1 = ref_probe_img_1.convert("RGBA")
    Not_import_overlay = Not_import_heat_map.convert("RGBA")
    Not_import_face_heat_map = Image.blend(background_1, Not_import_overlay, 0.5)

    ### Save the face heat maps ###
    path = 'D_HMs' + '/' + probe_name + '_' + gallery_name + '.png'
    Not_import_face_heat_map.save(path)

    # Display of the face heat map ###
    #plt.imshow(import_face_heat_map)
    #plt.show()
