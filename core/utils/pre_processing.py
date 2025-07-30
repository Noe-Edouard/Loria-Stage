
import os
import re
import numpy as np
from skimage.io import imsave, imread

def dias_pre_processing(input_dir, output_dir):
    """Fusionne les images par minimum pixel à pixel."""
    pattern = re.compile(r'image_s(\d+)_i\d+\.png')
    images_dict = {}

    for fname in os.listdir(input_dir):
        match = pattern.match(fname)
        if match:
            idx = int(match.group(1))
            images_dict.setdefault(idx, []).append(os.path.join(input_dir, fname))

    os.makedirs(output_dir, exist_ok=True)

    for idx, file_list in images_dict.items():
        img_path = os.path.join(output_dir, f'image_{idx}.png')
        if os.path.exists(img_path):
            print(f"Image déjà traitée : {img_path}, on saute.")
            continue
        imgs = []
        shapes = []
        for f in file_list:
            arr = imread(f)
            imgs.append(arr)
            shapes.append((f, arr.shape))
        # On garde seulement les images de la même taille que la première
        ref_shape = imgs[0].shape
        imgs_same_shape = [img for img in imgs if img.shape == ref_shape]
        if len(imgs_same_shape) < len(imgs):
            for (fname, shape) in shapes:
                if shape != ref_shape:
                    print(f"Taille différente ignorée pour {fname} : {shape} au lieu de {ref_shape}")
        if len(imgs_same_shape) == 0:
            print(f"Aucune image valide pour le groupe {idx}, on saute.")
            continue
        img = np.minimum.reduce(imgs_same_shape)
        imsave(img_path, img)
        print(f'Saved {img_path}')



if __name__ == "__main__":
    input_dir = 'data/datasets/DIAS/raw/full/images' 
    output_dir = 'data/datasets/DIAS/clean/images'  
    dias_pre_processing(input_dir, output_dir)
