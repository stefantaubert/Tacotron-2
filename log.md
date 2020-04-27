## install cuda 10.0 + cudnn 7.6.5 ([ref](https://stackoverflow.com/questions/55224016/importerror-libcublas-so-10-0-cannot-open-shared-object-file-no-such-file-or))
```
sudo dpkg -i cuda-repo-ubuntu1804_10.0.130-1_amd64.deb
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
sudo apt-get update
sudo apt-get install cuda-10-0
# download: https://developer.nvidia.com/compute/machine-learning/cudnn/secure/7.6.5.32/Production/10.0_20191031/cudnn-10.0-linux-x64-v7.6.5.32.tgz
tar -xvfz cudnn-10.0-linux-x64-v7.6.5.32.tgz
sudo cp -P cuda/include/cudnn.h /usr/local/cuda-10.0/include/
sudo cp -P cuda/lib64/libcudnn* /usr/local/cuda-10.0/lib64/
sudo chmod a+r /usr/local/cuda-10.0/lib64/libcudnn*
sudo nano /home/mi/.bashrc
# add the following lines at the end
export PATH=/usr/local/cuda-10.0/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64:${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
# save
# now reload bashrc
source /home/mi/.bashrc
```

## init repo with gpu support
```
ln -s /datasets/LJSpeech-1.1-lite/ /datasets/code/Tacotron-2/LJSpeech-1.1
ln -s /datasets/models/tacotron2/output/ /datasets/code/Tacotron-2/tacotron_output
sudo apt-get install -y libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0 ffmpeg
conda create -n tacotron2 python=3.6
conda activate tacotron2
pip install -r req_tf_1.13.1_all.txt
```

add: to launch.json
`"env": {"PYTHONPATH":"${workspaceFolder}"}`

## misc
To free up space:
```
conda clean --all
rm -rf ~./local/share/Trash/files
trash-empty
```

get nvidia driver version:
```
cat /proc/driver/nvidia/version
```