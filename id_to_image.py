import numpy as np
import os
import pandas as pd
from PIL import Image

save_path = '../dataset/CHD/id_image_29/'

if not os.path.exists(save_path):
    os.makedirs(save_path)

data_root = '../dataset/CHD/preprocessed/'

attacks = ['DoS', 'Fuzzy', 'gear', 'RPM']


def convert_canid_bits(cid):
    try:
        s = bin(int(str(cid), 16))[2:].zfill(29)
        bits = list(map(int, list(s)))
        return bits
    except:
        return None


for attack in attacks:

    print('Start converting', attack, 'data...')

    attack_path = data_root + '{}_dataset.csv'.format(attack)

    data = pd.read_csv(attack_path, low_memory=False)
    data['bit_ID'] = data['ID'].apply(convert_canid_bits)

    img_num = 0

    for i in range(0, len(data), 29):

        data_range = data.iloc[i: i + 29]
        id_range = data_range['bit_ID']
        flag_range = data_range['Flag']

        if 'T' in flag_range.values:
            label = 1  # attack
        else:
            label = 0  # normal

        img = np.zeros((1, 29))

        for j in id_range:
            img = np.vstack([img, j])

        img = np.delete(img, 0, axis=0)
        img = Image.fromarray(img.astype(np.uint8))

        full_save_path = save_path + attack

        if not os.path.exists(full_save_path):
            os.mkdir(full_save_path)

        img.save(full_save_path + '/{}_{}.png'.format(label, img_num))

        img_num += 1
        if img_num % 100000 == 0:
            print('{}th data in process...'.format(img_num))

    print(attack + ', Done converting to {} images'.format(img_num))

for attack in attacks:

    under = os.listdir(os.path.join(save_path, attack))
    print(len(under), attack, 'images')

    for i in under:
        full = os.path.join(save_path, attack, i)
        img = Image.open(full)
        size = img.size

        if size[0] != 29 or size[1] != 29:
            print(full)

        img.close()