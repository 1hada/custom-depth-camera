sudo apt-get install -y  tmuxp
sudo apt-get install -y ros-humble-usb-cam
# ideall we could install like this but currently the available debian package does not allow some of the nodes to launch properly # sudo apt-get install -y ros-humble-image-pipeline

pip install -U colcon-common-extensions # i found the jetson nano needed this o be installed seperately
