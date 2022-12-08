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
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(tar, path=data_folder)


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
