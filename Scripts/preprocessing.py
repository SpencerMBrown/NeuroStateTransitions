# import os
import numpy as np
import pandas as pd
import nibabel as nb
from sklearn.preprocessing import StandardScaler


def get_bold(sub, run, atlas, parcel):
    patha = [f'rfMRI_REST1_7T_PA',
             f'rfMRI_REST2_7T_AP',
             f'rfMRI_REST3_7T_PA',
             f'rfMRI_REST4_7T_AP']

    pathb = [f'{patha[0]}_Atlas_MSMAll_hp2000_clean.dtseries.nii',
             f'{patha[1]}_Atlas_MSMAll_hp2000_clean.dtseries.nii',
             f'{patha[2]}_Atlas_MSMAll_hp2000_clean.dtseries.nii',
             f'{patha[3]}_Atlas_MSMAll_hp2000_clean.dtseries.nii']

    pa = patha[run]
    pb = pathb[run]

    bold_path = f'./Data/{sub}/{sub}_7T_rfMRI/MNINonLinear/Results/{pa}/{pb}'
    bold_img = nb.load(bold_path)
    bold_header = bold_img.header
    bold_dat = bold_img.get_fdata()

    label_path = f'./Atlases/{atlas}/Schaefer2018_{parcel}Parcels_7Networks_order.dlabel.nii'
    label = nb.load(label_path)

    map1L = [i for i in bold_img.header.get_index_map(1)][1]
    map1R = [i for i in bold_img.header.get_index_map(1)][2]

    map1L = [i for i in map1L.vertex_indices]
    map1R = [i for i in map1R.vertex_indices]

    labelLR = [i for i in label.header.get_index_map(1)]

    map2L = label.get_fdata()[0][labelLR[0].index_offset:(labelLR[0].index_offset + labelLR[0].index_count)]
    map2L = map2L.astype('int')
    map2R = label.get_fdata()[0][labelLR[1].index_offset:(labelLR[1].index_offset + labelLR[1].index_count)]
    map2R = map2R.astype('int')

    atlas_l = map2L[map1L]
    atlas_r = map2R[map1R]
    atlas_lr = np.concatenate([atlas_l, atlas_r])

    cortex_l = [i for i in bold_header.get_index_map(1)][1]
    cortex_r = [i for i in bold_header.get_index_map(1)][2]

    bold_l = bold_dat[:, cortex_l.index_offset:(cortex_l.index_offset + cortex_l.index_count)]
    bold_r = bold_dat[:, cortex_r.index_offset:(cortex_r.index_offset + cortex_r.index_count)]

    bold_raw = np.hstack([bold_l, bold_r])

    # bold = [np.mean(bold[:, atlas_lr == i], axis=1) for i in np.arange(1, parcel + 1)]
    bold = []
    for i in np.arange(1, parcel + 1):
        bold_subset = bold_raw[:, atlas_lr == i]
        bold_subset = bold_subset[~np.any(bold_subset == 0, axis=1), :]

        if bold_subset.shape[1] == 0:
            continue

        bold_subset = StandardScaler().fit_transform(bold_subset) # Z-score normalization prior to averaging
        # print(bold_subset.shape, np.mean(bold_subset[:, 0]), np.var(bold_subset[:, 0]))
        bold_subset = np.mean(bold_subset, axis=1)

        bold.append(bold_subset)

    bold = np.stack(bold, axis=0)
    bold = StandardScaler().fit_transform(bold.T).T
    # print(bold[0, :].shape, np.mean(bold[0, :]), np.var(bold[0, :]))

    return bold

def get_adj(bold):
    adj = np.corrcoef(bold)
    adj = 1 - adj

    # Removing rows/columns with nans: <- not needed anymore, fixed above.
    # adj = adj[~np.all(np.isnan(adj), axis=1), :]
    # adj = adj[:, ~np.all(np.isnan(adj), axis=0)]

    return adj

##############################
# Task loading
##############################

def get_bold_task(sub, run, atlas, parcel, task):
    patha = [f'tfMRI_{task}_RL',
             f'tfMRI_{task}_LR']

    pathb = [f'{patha[0]}_Atlas_MSMAll.dtseries.nii',
             f'{patha[1]}_Atlas_MSMAll.dtseries.nii']

    pa = patha[run]
    pb = pathb[run]

    bold_path = f'./Data/{sub}/{sub}_3T_{task}/MNINonLinear/Results/{pa}/{pb}'
    bold_img = nb.load(bold_path)
    bold_header = bold_img.header
    bold_dat = bold_img.get_fdata()

    label_path = f'./Atlases/{atlas}/Schaefer2018_{parcel}Parcels_7Networks_order.dlabel.nii'
    label = nb.load(label_path)

    map1L = [i for i in bold_img.header.get_index_map(1)][1]
    map1R = [i for i in bold_img.header.get_index_map(1)][2]

    map1L = [i for i in map1L.vertex_indices]
    map1R = [i for i in map1R.vertex_indices]

    labelLR = [i for i in label.header.get_index_map(1)]

    map2L = label.get_fdata()[0][labelLR[0].index_offset:(labelLR[0].index_offset + labelLR[0].index_count)]
    map2L = map2L.astype('int')
    map2R = label.get_fdata()[0][labelLR[1].index_offset:(labelLR[1].index_offset + labelLR[1].index_count)]
    map2R = map2R.astype('int')

    atlas_l = map2L[map1L]
    atlas_r = map2R[map1R]
    atlas_lr = np.concatenate([atlas_l, atlas_r])

    cortex_l = [i for i in bold_header.get_index_map(1)][1]
    cortex_r = [i for i in bold_header.get_index_map(1)][2]

    bold_l = bold_dat[:, cortex_l.index_offset:(cortex_l.index_offset + cortex_l.index_count)]
    bold_r = bold_dat[:, cortex_r.index_offset:(cortex_r.index_offset + cortex_r.index_count)]

    bold_raw = np.hstack([bold_l, bold_r])

    # bold = [np.mean(bold[:, atlas_lr == i], axis=1) for i in np.arange(1, parcel + 1)]
    bold = []
    for i in np.arange(1, parcel + 1):
        bold_subset = bold_raw[:, atlas_lr == i]
        bold_subset = bold_subset[~np.any(bold_subset == 0, axis=1), :]

        if bold_subset.shape[1] == 0:
            continue

        bold_subset = StandardScaler().fit_transform(bold_subset)  # Z-score normalization prior to averaging
        # print(bold_subset.shape, np.mean(bold_subset[:, 0]), np.var(bold_subset[:, 0]))
        bold_subset = np.mean(bold_subset, axis=1)

        bold.append(bold_subset)

    bold = np.stack(bold, axis=0)
    bold = StandardScaler().fit_transform(bold.T).T
    # print(bold[0, :].shape, np.mean(bold[0, :]), np.var(bold[0, :]))

    return bold


def task_events(sub, run, task):
    patha = [f'tfMRI_{task}_RL',
             f'tfMRI_{task}_LR']
    # path = f'./Data/{sub}/{sub}_3T_{task}/MNINonLinear/Results/{patha[run]}/{task}_run{run+1}_TAB.txt'
    path = f'./Data/{sub}/{sub}_3T_{task}/MNINonLinear/Results/{patha[run]}/EVs/cue.txt'
    return pd.read_csv(path, sep='\t')

##############################
# Debug tests
##############################

# import os
# os.chdir('/Users/spencerbrown/PycharmProjects/Neuro/HCP')
# bold = get_bold('100610', 0, 'Schaefer2018_fslr32k', 1000)


# TODO: Add labels to get_bold.
