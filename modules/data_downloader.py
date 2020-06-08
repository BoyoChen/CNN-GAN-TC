import wget
import tarfile
import os

file_dict = {
    'TCIR-ATLN_EPAC_WPAC.h5': 'https://learner.csie.ntu.edu.tw/~boyochen/TCIR/TCIR-ATLN_EPAC_WPAC.h5.tar.gz',
    'TCIR-CPAC_IO_SH.h5': 'https://learner.csie.ntu.edu.tw/~boyochen/TCIR/TCIR-CPAC_IO_SH.h5.tar.gz',
    'TCIR-ALL_2017.h5': 'https://learner.csie.ntu.edu.tw/~boyochen/TCIR/TCIR-ALL_2017.h5.tar.gz'
}

compressed_postfix = '.tar.gz'


def download_compressed_file(data_folder, file_name):
    file_url = file_dict[file_name]
    file_path = data_folder + file_name + compressed_postfix
    wget.download(file_url, out=file_path)


def uncompress_file(data_folder, file_name):
    compressed_file_path = data_folder + file_name + compressed_postfix
    if not os.path.isfile(compressed_file_path):
        download_compressed_file(data_folder, file_name)

    with tarfile.open(compressed_file_path) as tar:
        tar.extractall(path=data_folder)


def verify_data(data_folder):
    for file_name in file_dict:
        file_path = data_folder + file_name
        if not os.path.isfile(file_path):
            print('data download failed!')
            return False
    return True


def download_data(data_folder):
    if data_folder[-1] != '/':
        data_folder += '/'

    if not os.path.isdir(data_folder):
        os.mkdir(data_folder)

    for file_name in file_dict:
        file_path = data_folder + file_name
        if not os.path.isfile(file_path):
            uncompress_file(data_folder, file_name)

    return verify_data(data_folder)
