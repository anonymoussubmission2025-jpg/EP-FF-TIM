import os
from scripts.downloads.utils import download_file_from_google_drive


ids = {'mini_imagenet': '1roXWm-COzPhyel_OZu7vseIpYV_-EH3G',
       'tiered_imagenet': '13CdfY6kOk9_G4omuA7ZJ29AsZXrS5sG-',
       'cub': '1Ay-91MwLmNWYv7nNIOoI45XZHiz2JQRb',
       'cifar': '1UDuASW-mrMicN_F0ymQK-NUQ9u5tQ9Ub'
       }


def download_data(dataset):
    id = ids[dataset]
    name = f"{dataset}.zip"
    print('Start Download (may take a few minutes)')
    download_file_from_google_drive(id, name)
    print('Finish Download')
    os.system(f'unzip {name}')
    os.system(f'rm {name}')


if __name__ == '__main__':
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    os.chdir(data_dir)
    download_data('cub')