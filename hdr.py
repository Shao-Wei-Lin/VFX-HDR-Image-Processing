import sys, os
import rawpy, cv2, argparse
import numpy as np 
import matplotlib.pyplot as plt 
import pyexr
from alignment import *

def process_command(): # settin parsers
    parsers = argparse.ArgumentParser()
    parsers.add_argument('--alignment', '-a', help = 'Using alignment or not', action = 'store_true')
    parsers.add_argument('--dodge_type', '-d', help = 'set type of using dodging and burning', default = 1)
    return parsers.parse_args()

def output_to_exr(img, file_name):
    img = img.astype(np.float16)
    pyexr.write(file_name, img)

def get_energy(img_list, delta_t, radiance_file = None):
    img_list = np.asarray(img_list)
    img_E = np.sum(img_list*delta_t, axis = 0) / np.sum(delta_t ** 2)
    
    E_norm = ((img_E - np.min(img_E)) / (np.max(img_E) - np.min(img_E))) # normalize

    if radiance_file != None:
        '''
        rad_E_gray = np.sum(img_E * (np.asarray([0.2990, 0.5870, 0.1140]).reshape(1, 1, 3)), axis = -1) # RGB -> Gray
        rad_fig = plt.figure()
        im = plt.imshow(rad_E_gray, cmap = 'jet')
        rad_fig.colorbar(im)
        plt.savefig(radiance_file + '.png')
        '''
        output_to_exr(img_E, radiance_file + '.exr')        

    return E_norm

def tone_mapping_global(img, a = 0.18, using_dab = True, to_gray = True): # this img is an normalized HDR image.
    if to_gray:
        print('Transfer to gray before normalize')
        # transfer to gray 
        img_gray = np.sum(img * (np.asarray([0.2990, 0.5870, 0.1140]).reshape(1, 1, 3)), axis = -1) # RGB -> Gray
    
        # performing dodging and burning
        L_w_avg = np.exp(np.sum(np.log(1e-9 + img_gray)) / np.prod(img_gray.shape)) 
        L_m = (a/L_w_avg)*img_gray 
        if using_dab:
            V = dodging_and_burning(L_m) 
            L_d = (L_m * (1 + L_m / np.max(L_m)**2)) / (1 + V)
        else:
            L_d = (L_m * (1 + L_m / np.max(L_m)**2)) / (1 + L_m)
        # recovery to rgb
        L_rgb = np.clip((L_d / img_gray).reshape(L_d.shape[0], L_d.shape[1], 1) * img, 0, 1)
        return L_rgb 
    else: 
        print('Not transfer to gray before normalize')
        # method without using gray 
        L_w_avg = np.exp(np.sum(np.log(1e-9 + img)) / np.prod(img.shape)) 
        L_m = (a/L_w_avg)*img 
        if using_dab == True:
            V = dodging_and_burning(L_m, a = a) 
            L_d = (L_m * (1 + L_m / np.max(L_m)**2)) / (1 + V)
        else:
            L_d = (L_m * (1 + L_m / np.max(L_m)**2)) / (1 + L_m)
        return L_d
    

def dodging_and_burning(img, a = 0.18):
    # settings according to paper
    phi = 15
    epsilon = 0.05

    # processing
    for i in range(1, 50, 2):
        V_s = cv2.GaussianBlur(img, (i, i), 0) # represent V_{s}
        V_s1 = cv2.GaussianBlur(img, (i+2, i+2), 0) # represent V_{s+1}
        V = (V_s - V_s1) / ((2 ** phi)*a/(i ** 2) + V_s)
        dist = np.linalg.norm(V)
        
        if dist > epsilon: 
            return V_s  

def dodging_and_burning_type2(img, a = 0.18):
    # settings according to paper
    phi = 10 
    epsilon = 0.05

    L_w_avg = np.exp(np.sum(np.log(1e-9 + img)) / np.prod(img.shape)) 
    L_m = (a/L_w_avg)*img 
    # processing
    for i in range(1, 100, 2):
        V_s = cv2.GaussianBlur(img, (i, i), 0) # represent V_{s}
        V_s1 = cv2.GaussianBlur(img, (i+2, i+2), 0) # represent V_{s+1}
        V = (V_s - V_s1) / ((2 ** phi)*a/(i ** 2) + V_s)
        dist = np.linalg.norm(V)
        if dist > epsilon:  
            print(i, dist)
            break
    L_d = L_m / (1 + V_s)

    return L_d
    


def save_image(img, filename):
    if(np.max(img) <= 1):
        img = np.round(img * 255)
    img_cv2 = img[:, :, ::-1]
    cv2.imwrite(filename, img_cv2)

if __name__ == '__main__':
    # get parsers
    args = process_command()
    dodge_type = args.dodge_type 
    using_alignment = args.alignment 

    # loading image and classify them 
    raw_0 = []
    delta_0 = np.asarray([2 ** ((i+5)/3) for i in range(9)]).reshape(-1, 1, 1, 1)

    filelist = sorted(os.listdir('hdr'))
    for f_name in filelist:
        raw_img = rawpy.imread(os.path.join('hdr', f_name))
        rgb16 = raw_img.postprocess(output_bps = 16, no_auto_bright = True, use_camera_wb = True).astype('float')
        if(f_name.split('_')[0] == '0'):
            raw_0.append(rgb16)
    
    if using_alignment == True:
        print('Implement image alignment')
        raw_0 = align_image(raw_0)

    # get HDR image and store in radiance map 
    energy_0 = get_energy(raw_0, delta_0)
    
    # tone_mapping 
    if dodge_type == 1:
        a = 0.18
        print('Apply dodging and burning by type 1')
        img_0 = tone_mapping_global(energy_0, a = a, using_dab = True, to_gray = True)
        save_image(img_0, filename = f'0_HDR-{a}_wt_dab.png')

    elif dodge_type == 2:
        print('Apply dodging and burning by type 2')
        img_0 = tone_mapping_global(energy_0, a = 0.18, using_dab = False)
        img_0 = dodging_and_burning_type2(img_0)
        save_image(img_0, filename = f'0_HDR-0.18_rgb_wt_dab_v2.png')


    


