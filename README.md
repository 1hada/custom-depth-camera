
https://answers.ros.org/question/396439/setuptoolsdeprecationwarning-setuppy-install-is-deprecated-use-build-and-pip-and-other-standards-based-tools/
requirements to avoid the pip install warning
```
pip install setuptools==58.2.0
```

and setup.cfg should be 
```
[develop]
script_dir=$base/lib/camera_simulator
[install]
install_scripts=$base/lib/camera_simulator
```

```
colcon build --symlink-install
source install/setup.bash
sudo apt  install ffmpeg -y
ffmpeg -fflags +genpts -i 2023-08-20-194610.webm -r 24 calibration-1.mp4
```

# run the camera and the calibnration logic

https://github.com/klintan/ros2_video_streamer
camera
```
ros2 run camera_simulator camera_simulator --type images --path /home/biobe/Pictures/Webcam/calibration-images --loop # --calibration_file src/ros2_video_streamer/data/camera.yaml
```

Calibration logic
```
ros2 run camera_calibration cameracalibrator --size 7x9 --square 0.02 --ros-args -r image:=/my_camera/image_raw -p camera:=/my_camera
```



# get the image pipeline to better get the calibration for a monocular camera
`sudo apt install ros-humble-image-pipeline `
OR

https://navigation.ros.org/tutorials/docs/camera_calibration.html
```
source /opt/ros/humble/setup.bash 
colcon build --symlink-install
sudo apt install ros-humble-camera-calibration-parsers
sudo apt install ros-humble-camera-info-manager
sudo apt install ros-humble-launch-testing-ament-cmake

cd src
git clone -b humble git@github.com:ros-perception/image_pipeline.git

```