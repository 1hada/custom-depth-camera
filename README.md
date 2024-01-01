# Background
The purpose of this repo is 2-fold.
On the one hand I am providing rough guidance for someone to make a stereo camera using ros2.
- I am not providing a fully detailed set of instructions because there are blogs which have dedicated pages to such instructions.
On the other hand this repo represents my "scratching an itch" to empirically compare the effect of rotating cameras towards the focal point of interest. In this example I rotate the cameras inwards nby 0.6 degrees in order to have them focus at a theoretical center point which is 6-8 meters away. The cameras will have a base line distance of 82 mm. 

In order to fully compare the effect of rotating the cameras I simultaneously set up another stereo camera with cameras that also have an 82mm baseline distance. These cameras are facing forward with no rotation.

The cameras were attained via amazon and can be found at :
- https://www.amazon.com/ELP-Megapixel-Distortion-Webcamera-Industrial/dp/B0BGH2LDMP

The housing for the cameras is 3d printed and made on fusion 360.
- [Housing](media/USB16MP01-parts.stl)

# Set Up

### Get the repo
```
git clone git@github.com:1hada/custom-depth-camera.git --recurse-submodules
```

### Environment Configurations
```
mkdir -p ~/.tmuxp 
mkdir -p ~/.ros/camera_info
sudo mkdir -p /etc/udev/rules.d/ 

# I use a USB hub dedicated to these cameras and define these rules to make it as easy as pie to know exactly which camera corresponds to left/right and upper/lower
sudo cp .udev/99-usb-camera.rules /etc/udev/rules.d/99-usb-camera.rules

cp .tmuxp/*  ~/.tmuxp/
cp .camera_info/*  ~/.ros/camera_info/

source /opt/ros/humble/setup.bash
```


### Get Camera Calibrations
In order to get the camera calibration parameters you'll need to print a calibration pattern. 
For my example I use a  checkerboard pattern with the following specifications `--size 12x8 --square 0.028`
```
tmuxp load ~/.tmuxp/get-camera-config.yaml
```
If everything is connected correctly you will be able to press "calibrate" and then "save" the calibration file.
- Be sure to also rename the calibration file if you are like be and have mutlple stereo cameras.
http://wiki.ros.org/camera_calibration/Tutorials/StereoCalibration


### Run 2 Stereo Cameras
```
tmuxp load ~/.tmuxp/double-camera-to-depth-suite.yaml
```


# Results 
During this experiment I printed slots to hold 4 cameras. 
- There were two "top" cameras used as a pair for the "top" stereoscopic camera. 
  - These cameras were rotated towards eachother by 0.6 degrees (0.010472 radians) 
- There were two "bottom" cameras used as a pair for the "bottom" stereoscopic camera. 
  - These cameras are facing straight.

The results of this experiment are emperical since they are meant to determine further research into the practical effect of camera angles for stereo cameras which have a wide field of view. The results show that the any variance in the camera angle is negligible and may in fact result in a worse point cloud (due to angle accuracy as per any manufacturing tolerances) than the simpler set up of simply have the cameras face forward.

Here is a video showing the resulting point clouds. Yellow is the Lower stereo camera and White is the top stereo camera where the cameras are angled inwards.

<figure class="video_container">
  <video controls="true" allowfullscreen="true">
    <source src="media/demo-pointcoud.webm" type="video/webm">
  </video>
</figure>

# Misc

get the image pipeline to better get the calibration for a monocular camera
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

