import sys, os
import rawpy 
import numpy as np 
import cv2

def shift(img, m, n, rgb):
    if rgb == 0:
        if m > 0:
            a = np.pad(img, ((m, 0), (0,0)), mode='edge')
            img = np.roll(img, m, axis = 0)        
            img = np.concatenate((a[:m, :], img[m:, :]), axis = 0)
        elif m < 0:
            a = img[-1, :].reshape(1, img.shape[1])
            img = np.roll(img, m, axis = 0)
            img = np.concatenate((img[:m, :], np.repeat(a, abs(m), axis = 0)), axis = 0)

        if n > 0:
            a = np.pad(img, ((0, 0), (n, 0)), mode='edge')
            img = np.roll(img, n, axis = 1)        
            img = np.concatenate((a[:, :n], img[:, n:]), axis = 1)
        elif n < 0:
            a = img[:, -1].reshape(img.shape[0], 1)
            img = np.roll(img, n, axis = 1)
            img = np.concatenate((img[:, :n], np.repeat(a, abs(n), axis = 1)), axis = 1)
        return img

    else:
        if m > 0:
            a = img[0, :, :].reshape(1, img.shape[1], 3)
            img = np.roll(img, m, axis = 0)        
            img = np.concatenate((np.repeat(a, m, axis = 0), img[m:, :, :]), axis = 0)
        elif m < 0:
            a = img[-1, :, :].reshape(1, img.shape[1], 3)
            img = np.roll(img, m, axis = 0)
            img = np.concatenate((img[:m, :, :], np.repeat(a, abs(m), axis = 0)), axis = 0)

        if n > 0:
            a = img[:, 0, :].reshape(img.shape[0], 1, 3)
            img = np.roll(img, n, axis = 1)        
            img = np.concatenate((np.repeat(a, n, axis = 1), img[:, n:]), axis = 1)
        elif n < 0:
            a = img[:, -1, :].reshape(img.shape[0], 1, 3)
            img = np.roll(img, n, axis = 1)
            img = np.concatenate((img[:, :n, :], np.repeat(a, abs(n), axis = 1)), axis = 1)

        return img        

def shift_image(bin1, bin2):
    zoom_size = [2**i for i in range(9, -1, -1)]
    shift_dir = [-1, 0, 1]
    x = 0
    y = 0
    for n in zoom_size:
        min = 100000000
        x *= 2
        y *= 2
        img1 = bin1[::n, ::n]
        img2 = bin2[::n, ::n]
        img2 = shift(img2, x, y, rgb = 0)
        current = 0
        for i in shift_dir:
            for j in shift_dir:
                current = np.sum(np.bitwise_xor(img1, shift(img2, i, j, rgb = 0)))
                # print('zoom size =',n, 'bitwise xor =', current)
                if current < min:
                    min = current
                    i_x = i 
                    i_y = j
        x += i_x 
        y += i_y

    return x,y

def align_image(img_list):
    img_list = np.asarray(img_list) 
    align_list = (img_list/257).astype('uint8')
    align_list = np.sum( align_list * np.asarray([0.299, 0.587, 0.114]).reshape(1, 1, 1, 3), axis = 3) # RGB -> gray
    
    # to binary
    threshold = np.median(align_list, axis = (1, 2))
    bin_value = []
    bin_img = []
    for i in range(len(threshold)):
        im_bool = align_list[i] > threshold[i]
        bin_value.append(im_bool)
        bin_img.append( im_bool*255 )
    bin_value = np.asarray(bin_value)
    bin_img = np.asarray(bin_img)

    shift_size = []
    for i in range(1, len(bin_value)):
        shift_size.append( shift_image(bin_value[0], bin_value[i]) )
    print(shift_size)

    for i in range(1, len(img_list)):
        img_list[i] = shift(img_list[i], shift_size[i-1][0], shift_size[i-1][1], rgb = 1)

    return img_list


