import numpy as np
import pandas as pd
import h5py
import os
import math
import pickle
from datetime import timedelta
from modules.data_downloader import download_data


def remove_outlier_and_nan(numpy_array, upper_bound=1000):
    numpy_array = np.nan_to_num(numpy_array)
    numpy_array[numpy_array > upper_bound] = 0
    VIS = numpy_array[:, :, :, 2]
    VIS[VIS > 1] = 1  # VIS channel ranged from 0 to 1
    return numpy_array


def flip_SH_image_matrix(info_df, image_matrix):
    SH_idx = info_df.index[info_df.data_set == 'SH']
    image_matrix[SH_idx] = np.flip(image_matrix[SH_idx], 1)
    return image_matrix


def mark_good_quality_VIS(label_df, image_matrix):
    tmp_df = pd.DataFrame(columns=['vis_mean', 'vis_std'])
    for i in range(image_matrix.shape[0]):
        VIS_matrix = image_matrix[i, :, :, 2]
        tmp_df.loc[i] = [VIS_matrix.mean(), VIS_matrix.std()]

    tmp_df['hour'] = label_df.apply(lambda x: x.local_time.hour, axis=1)
    return tmp_df.apply(
        lambda x: (0.1 <= x.vis_mean <= 0.7) and (0.1 <= x.vis_std <= 0.31) and (7 <= x.hour <= 16),
        axis=1
    )


def scale_to_0_1(matrix):
    out = matrix - matrix.min()
    tmp_max = out.max()
    if tmp_max != 0:
        out /= tmp_max
    return out


def fix_reversed_VIS(image_matrix):
    for i in range(image_matrix.shape[0]):
        IR1_matrix = image_matrix[i, :, :, 0]
        VIS_matrix = image_matrix[i, :, :, 2]
        reversed_VIS_matrix = 1 - VIS_matrix
        VIS_IR1_distance = abs(scale_to_0_1(IR1_matrix) - scale_to_0_1(VIS_matrix)).mean()
        reversed_VIS_IR1_distance = abs(scale_to_0_1(IR1_matrix) - scale_to_0_1(reversed_VIS_matrix)).mean()
        if reversed_VIS_IR1_distance > VIS_IR1_distance:
            VIS_matrix *= -1
            VIS_matrix += 1


def crop_center(matrix, crop_width):
    total_width = matrix.shape[1]
    start = total_width // 2 - crop_width // 2
    end = start + crop_width
    return matrix[:, start:end, start:end, :]


def get_minutes_to_noon(local_time):
    minutes_in_day = 60 * local_time.hour + local_time.minute
    noon = 60 * 12
    return abs(noon - minutes_in_day)


def extract_label_and_feature_from_info(info_df):
    # --- region feature ---
    info_df['region_code'] = pd.Categorical(info_df.data_set).codes
    info_df['lon'] = (info_df.lon+180) % 360 - 180  # calibrate longitude, ex: 190 -> -170
    # --- time feature ---
    info_df['GMT_time'] = pd.to_datetime(info_df.time, format='%Y%m%d%H')
    info_df['local_time'] = info_df.GMT_time \
        + info_df.apply(lambda x: timedelta(hours=x.lon/15), axis=1)
    # --- year_day ---
    SH_idx = info_df.index[info_df.data_set == 'SH']
    info_df['yday'] = info_df.local_time.apply(lambda x: x.timetuple().tm_yday)
    info_df.loc[SH_idx, 'yday'] += 365 / 2  # TC from SH
    info_df['yday_transform'] = info_df.yday.apply(lambda x: x / 365 * 2 * math.pi)
    info_df['yday_sin'] = info_df.yday_transform.apply(lambda x: math.sin(x))
    info_df['yday_cos'] = info_df.yday_transform.apply(lambda x: math.cos(x))
    # --- hour ---
    info_df['hour_transform'] = info_df.apply(lambda x: x.local_time.hour / 24 * 2 * math.pi, axis=1)
    info_df['hour_sin'] = info_df.hour_transform.apply(lambda x: math.sin(x))
    info_df['hour_cos'] = info_df.hour_transform.apply(lambda x: math.cos(x))
    # split into 2 dataframe
    label_df = info_df[['data_set', 'ID', 'local_time', 'Vmax', 'R35_4qAVG', 'MSLP']]
    feature_df = info_df[['lon', 'lat', 'region_code', 'yday_cos', 'yday_sin', 'hour_cos', 'hour_sin']]
    return label_df, feature_df


def data_cleaning_and_organizing(image_matrix, info_df):
    image_matrix = remove_outlier_and_nan(image_matrix)
    image_matrix = flip_SH_image_matrix(info_df, image_matrix)
    fix_reversed_VIS(image_matrix)

    label_df, feature_df = extract_label_and_feature_from_info(info_df)
    feature_df['minutes_to_noon'] = label_df['local_time'].apply(get_minutes_to_noon)
    feature_df['is_good_quality_VIS'] = mark_good_quality_VIS(label_df, image_matrix)
    return image_matrix, label_df, feature_df


def train_valid_split(label_df, feature_df, image_matrix, phase):
    if phase == 'train':
        target_index = label_df.index[label_df.ID < '2015000']
    elif phase == 'valid':
        target_index = label_df.index[label_df.ID > '2015000']
    return {
        'label': label_df.loc[target_index].reset_index(drop=True),
        'feature': feature_df.loc[target_index].reset_index(drop=True),
        'image': image_matrix[target_index]
    }


def extract_features_from_raw_file(data_folder):
    data_files = {
        'train': ['TCIR-ATLN_EPAC_WPAC.h5', 'TCIR-CPAC_IO_SH.h5'],
        'test': ['TCIR-ALL_2017.h5']
    }

    datasets = {}
    for phase, file_list in data_files.items():
        # collect matrix from every file in the list
        matrix = []
        for file_name in file_list:
            file_path = data_folder+file_name
            if not os.path.isfile(file_path):
                print('file %s not found! try to download it!' % file_path)
                download_data(data_folder)
            with h5py.File(file_path, 'r') as hf:
                center_cropped_matrix = crop_center(hf['matrix'][:], 128)
                matrix.append(center_cropped_matrix)
        image_matrix = np.concatenate(matrix, axis=0)
        # collect info from every file in the list
        info_df = pd.concat([
            pd.read_hdf(data_folder+file_name, key='info', mode='r')
            for file_name in file_list
        ]).reset_index(drop=True)

        image_matrix, label_df, feature_df = data_cleaning_and_organizing(image_matrix, info_df)

        if phase == 'train':
            datasets['train'] = train_valid_split(label_df, feature_df, image_matrix, 'train')
            datasets['valid'] = train_valid_split(label_df, feature_df, image_matrix, 'valid')
        elif phase == 'test':
            datasets['test'] = {
                'label': label_df,
                'feature': feature_df,
                'image': image_matrix
            }

    return datasets


def remove_bad_quality_VIS_data(label_df, feature_df, image_matrix):
    good_VIS_index = feature_df.index[
        feature_df['is_good_quality_VIS']
    ]
    label_df = label_df.loc[good_VIS_index].reset_index(drop=True)
    feature_df = feature_df.loc[good_VIS_index].reset_index(drop=True)
    image_matrix = image_matrix[good_VIS_index]
    return label_df, feature_df, image_matrix


def remove_0_R35_data(label_df, feature_df, image_matrix):
    positive_R35_index = label_df.index[
        label_df['R35_4qAVG'] > 0
    ]
    label_df = label_df.loc[positive_R35_index].reset_index(drop=True)
    feature_df = feature_df.loc[positive_R35_index].reset_index(drop=True)
    image_matrix = image_matrix[positive_R35_index]
    return label_df, feature_df, image_matrix


def load_dataset(data_folder, phase, good_VIS_only, positive_R35_only):
    if phase not in ['train', 'valid', 'test']:
        print('phase should be one of train/valid/test.')
        return

    pickle_path_format = '%sTCIR.%s.pickle'
    pickle_path = pickle_path_format % (data_folder, phase)

    if not os.path.isfile(pickle_path):
        print('pickle %s not found! try to extract it from raw data!' % pickle_path)
        datasets = extract_features_from_raw_file(data_folder)
        for phase in datasets:
            print('saving %s phase data pickle!' % phase)
            save_path = pickle_path_format % (data_folder, phase)
            with open(save_path, 'wb') as save_file:
                pickle.dump(datasets[phase], save_file, protocol=4)

    with open(pickle_path, 'rb') as load_file:
        dataset = pickle.load(load_file)

    if good_VIS_only or positive_R35_only:
        label = dataset['label']
        feature = dataset['feature']
        image = dataset['image']
        if good_VIS_only:
            label, feature, image = remove_bad_quality_VIS_data(label, feature, image)
        if positive_R35_only:
            label, feature, image = remove_0_R35_data(label, feature, image)
        dataset = {
            'label': label,
            'feature': feature,
            'image': image
        }

    return dataset
