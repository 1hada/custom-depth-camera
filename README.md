
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
ros2 run camera_simulator camera_simulator --type video --path /home/biobe/Videos/Webcam/calibration-video-1.mp4 --loop # --calibration_file src/ros2_video_streamer/data/camera.yaml
```

Calibration logic
```
ros2 run camera_calibration cameracalibrator --size 7x9 --square 0.02 --ros-args -r image:=/image/image_raw -p camera:=/camera_simulator
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


# Stereo camera tuning
http://wiki.ros.org/camera_calibration/Tutorials/StereoCalibration

```
rosrun camera_calibration cameracalibrator.py --approximate 0.1 --size 8x6 --square 0.108 right:=/my_stereo/right/image_raw left:=/my_stereo/left/image_raw right_camera:=/my_stereo/right left_camera:=/my_stereo/left
```




# Results 
Results from testing the errors at lengths which are more appropriate for human sized robots increases exponentially near the edges, indicating that the main error is associated with the quantization error rather than the angles of the cameras. This again is assuming a zero centerd model.

If we continued the 0,0 point in the image place to be the center. The next QUESTION IN REDUCING THE DEPTH ERROR OF STEREOSCOPIC RESOLUTION IN HUMAN SIZED ROBOTS  is , can we reduce the errors associated with image reconstruction by using weighted averages of quantized images. Thus if the quatized image has a largely different depth value for its distaprity image, then the hypotheses to test will be if the accuracy of depth improves in conditions with repeating patterns, TODO any other errors that plaque stereo cameras and can be solved with weighted averaged distarities.

