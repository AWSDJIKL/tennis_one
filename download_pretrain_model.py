import gdown
import tarfile
import os

download_url = [
    ["checkpoint.tar", "https://drive.google.com/file/d/1WgrIytz73ixF8T92-aidNwFOVjCP7yrU/view?usp=sharing",
     ["./rtdetrv2/references/deploy/lib/checkpoint"]],
    ["dataset.tar", "https://drive.google.com/file/d/1Gr5MY-UoVthexoB95VbBYs2H4nzuYCxI/view?usp=sharing",
     ["./rtdetrv2/dataset", "./hot/dataset"]],
    ["pretrained.tar", "https://drive.google.com/file/d/18CMFCVaHpOGuHLuYt8MLIvejGNNzwfWG/view?usp=sharing",
     ["./hot/checkpoint/pretrained", "./rtdetrv2/references/deploy/checkpoint/pretrained"]],
    ["yolo11.tar", "https://drive.google.com/file/d/1Gj4k9u-17cfTzKBLIG1K1Rei5I7Qmtos/view?usp=sharing", ["./rtdetrv2"]]
]

for tar_file_name, url, unzip_dir in download_url:
    gdown.download(url, tar_file_name, quiet=False, fuzzy=True)
    for path in unzip_dir:
        with tarfile.open(tar_file_name, mode='r:*') as tar:
            tar.extractall(path=path)
            print(f"已解压 {tar_file_name} 到 {path}")
    os.remove(tar_file_name)
