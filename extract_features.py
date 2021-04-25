from scipy.spatial.distance import hamming, cosine
import pandas as pd
import numpy as np
import pickle
import math

with open('gates_data_medium.pickle', 'rb') as f:
    data = pickle.load(f)

metadata = pd.read_csv('METADATA.csv')

gates0 = data['c0_hgates']
gates1 = data['c1_hgates']

# ax_0_clean = np.zeros((512,), dtype=np.uint64)
# ax_1_clean = np.zeros((512,), dtype=np.uint64)
# ax_0_backd = np.zeros((512,), dtype=np.uint64)
# ax_1_backd = np.zeros((512,), dtype=np.uint64)

c0_clean, c1_clean, or_clean = [], [], []
c0_backd, c1_backd, or_backd = [], [], []
hamming_clean, hamming_backd = [], []
cos_clean, cos_backd = [], []

for i in range(len(metadata)):
    # if 100 <= i <= 120:
    if True:
        poisoned = metadata['poisoned'].iloc[i]
        trigger_0_src_class = metadata['triggers_0_source_class'].iloc[i]
        trigger_0_tgt_class = metadata['triggers_0_target_class'].iloc[i]
        trigger_1_src_class = metadata['triggers_1_source_class'].iloc[i]
        trigger_1_tgt_class = metadata['triggers_1_target_class'].iloc[i]

        # mask = 0 if the gate is ZERO otherwise 1
        mask0 = (gates0[i] > np.finfo(np.float).eps).astype(np.uint64)
        mask1 = (gates1[i] > np.finfo(np.float).eps).astype(np.uint64)
        # mask_and = np.bitwise_and(mask0, mask1)
        mask_or = np.bitwise_or(mask0, mask1)
        # mask_xor = np.bitwise_xor(mask0, mask1)

        sum0 = mask0.sum()
        sum1 = mask1.sum()
        # sum_and = mask_and.sum()
        sum_or = mask_or.sum()
        # sum_xor = mask_xor.sum()

        dist_hamming = hamming(mask0, mask1)
        if sum0 == 0 or sum1 == 0:
            continue
        dist_cos = cosine(mask0, mask1)

        if math.isnan(dist_cos):
            print(f'i={i}')
            print(mask0)
            print(mask1)

        if poisoned:
            c0_backd.append(sum0)
            c1_backd.append(sum1)
            or_backd.append(sum_or)
            hamming_backd.append(dist_hamming)
            cos_backd.append(dist_cos)
        #     ax_0_backd += mask0
        #     ax_1_backd += mask1
        else:
            c0_clean.append(sum0)
            c1_clean.append(sum1)
            or_clean.append(sum_or)
            hamming_clean.append(dist_hamming)
            cos_clean.append(dist_cos)
        #     ax_0_clean += mask0
        #     ax_1_clean += mask1

        mask0 = ''.join([str(x) for x in mask0])
        mask1 = ''.join([str(x) for x in mask1])
        # mask_and = ''.join([str(x) for x in mask_and])
        mask_or = ''.join([str(x) for x in mask_or])
        # mask_xor = ''.join([str(x) for x in mask_xor])

        # print(f'Poisoned: {poisoned}')
        # # print(f'Size: {len(mask0)}')
        # print(f'Trigger 0 source class: {trigger_0_src_class}')
        # print(f'Trigger 0 target class: {trigger_0_tgt_class}')
        # print(f'Trigger 1 source class: {trigger_1_src_class}')
        # print(f'Trigger 1 target class: {trigger_1_tgt_class}')
        # print(f'Hamming: {dist_hamming}')
        # print(f'Sum mask 0: {sum0}')
        # print(f'Sum mask 1: {sum1}')
        # # # print(f'Sum mask and: {sum_and}')
        # # print(f'Sum mask or : {sum_or}')
        # # print(f'Sum mask xor: {sum_xor}')
        # # print(f'Diff sum abs: {abs(sum0 - sum1)}')
        # print(f'mask0:    {mask0}')
        # print(f'mask1:    {mask1}')
        # # print(f'mask_and: {mask_and}')
        # # print(f'mask_or:  {mask_or}')
        # # print(f'mask_xor: {mask_xor}')
        # print()
# print(f'0 clean: m={ax_0_clean.mean()}, std={ax_0_clean.std()}')
# print(f'1 clean: m={ax_1_clean.mean()}, std={ax_1_clean.std()}')
# print()
# print(f'0 backd: m={ax_0_backd.mean()}, std={ax_0_backd.std()}')
# print(f'1 backd: m={ax_1_backd.mean()}, std={ax_1_backd.std()}')
# print()

# c0_clean, c1_clean, or_clean = np.array(c0_clean), np.array(c1_clean), np.array(or_clean)
# c0_backd, c1_backd, or_backd = np.array(c0_backd), np.array(c1_backd), np.array(or_backd)
# print(f'c0_clean: mean={c0_clean.mean()}, std={c0_clean.std()}')
# print(f'c1_clean: mean={c1_clean.mean()}, std={c1_clean.std()}')
# print(f'or_clean: mean={or_clean.mean()}, std={or_clean.std()}')
# print()
# print(f'c0_backd: mean={c0_backd.mean()}, std={c0_backd.std()}')
# print(f'c1_backd: mean={c1_backd.mean()}, std={c1_backd.std()}')
# print(f'or_backd: mean={or_backd.mean()}, std={or_backd.std()}')

# hamming_clean, hamming_backd = np.array(hamming_clean), np.array(hamming_backd)
# print(f'hamming_clean: mean={hamming_clean.mean()}, std={hamming_clean.std()}')
# print(f'hamming_backd: mean={hamming_backd.mean()}, std={hamming_backd.std()}')

cos_clean, cos_backd = np.array(cos_clean), np.array(cos_backd)
# print(cos_clean)
# print(cos_backd)
print(f'cos_clean: mean={cos_clean.mean()}, std={cos_clean.std()}')
print(f'cos_backd: mean={cos_backd.mean()}, std={cos_backd.std()}')

# TODO: check whether we have two zeros on the same position
