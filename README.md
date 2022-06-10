# Vehicle-direction-identification
Vehicle direction identification consists of three module detection , tracking and direction recognization.


### Algorithm used : Yolo algorithm for detection + SORT algorithm to track vehicles + vector based direction detection
### Backend : opencv and python
### Library required:

- opencv = '4.5.4-dev'
- scipy = '1.4.1'
- filterpy 
- lap
- scikit-image

### FLOW CHART
<img src="vehicledirection.png" alt="flowchart" />



# Quick Overview about structure

#### 1) main.py

- Loading model and user configurations
- perform io interfacing tasks


#### 2) yolo.py

- use opencv modules to detect objects from user given media(photo/video)
- detection take place inside this file


#### 3) config.json

- user configuration are mentioned inside this file
- for examples : input shapes and model parameters(weights file path , config file path etc) are added in config.json

#### 5) sort.py

- SORT algorithm implementations
- Kalman filter operations



### Limitations:

There are few primary drawbacks of this appoach

1) direction recogization totally depends on detection and tracking.
2) if camera properly arranged then it gives accurate results (Suppose any object is in front of camera and come forward towards camera then it gives bad results)
    but if you try to use this approach in cctv suviellence then it gives satisfactory results.
    
3) in few cases , it performs bad, because right now it works on only single keypoint (center of object) we can improve its performace by detecting multiple keypoints and use majority votes result.
 

