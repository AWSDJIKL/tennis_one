# Only in linux

# 1. Set up

### 1.1 conda env

~~~
conda create -n tennis_one python=3.11
# install pytorch
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
# install yolo11
pip3 install ultralytics
# install utils tool
pip3 install -r requirements.txt
~~~

### 1.2 orbbecsdk setup
~~~
cd pyorbbecsdk
pip3 install -r requirements.txt
mkdir build
cd build
cmake -Dpybind11_DIR=`pybind11-config --cmakedir` ..
make -j4
make install

cd ..
# 仅在当前terminal生效
export PYTHONPATH=$PYTHONPATH:$(pwd)/install/lib/
# 可以在~/.bashrc中添加，替换$(pwd)为 tennis_one/ptorbbecsdk   注意要写绝对路径
export PYTHONPATH=$PYTHONPATH:/home/tennis_one/tennis_one/ptorbbecsdk/install/lib/
sudo bash ./scripts/install_udev_rules.sh
sudo udevadm control --reload-rules && sudo udevadm trigger
python3 examples/net_device.py # Requires ffmpeg installation for network device
~~~

### 1.3 rt-detrv2 setup

~~~
cd rtdetrv2
pip3 install -r requirements.txt
~~~

### 1.4 HoT setup

~~~
cd hot
pip3 install -r requirements.txt
~~~


### 1.5 pretrain model set up

~~~
python download_pretrain_model.py
~~~

# 2. Run

### 2.1 Only record by orbbecsdk

### 2.2 Crop video by rt-detr-v2

### 2.3 Create 2D and 3D bone data by HoT

