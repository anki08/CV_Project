We will train a CNN to do vision-based driving in SuperTuxKart.

We will design a simple low-level controller that acts as an auto-pilot to drive in SuperTuxKart. We then use this auto-pilot to train a vision based driving system. To get started, first download and install SuperTuxKart on your machine. If you are working on colab, be sure that you have the GPU hardware accelerator enabled. To enable the GPU hardware accelerator go to Runtime > Change Runtime Type > GPU > Save your colab notebook will then restart with GPU enabled.

runtime

Once you have GPU enabled use the following to install SuperTuxKart:
 > %pip install -U PySuperTuxKart


### Controller
In the first part of this homework, you will write a low-level controller in controller.py. The controller function takes as input an aim point and the current velocity of the car. The aim point is a point on the center of the track 15 meters away from the kart, as shown below.

Example: 
![Example](https://github.com/anki08/CV_Project/blob/main/SuperTuxKart/controller.png )

We use screen coordinates for the aim point: [-1..1] in both x and y directions. The point (-1,-1) is the top left of the screen and (1, 1) the bottom right.

In the first part, we will use a ground truth aim point from the simulator itself. In the second part, we remove this restriction and predict the aim point directly from the image.

The goal of the low-level controller is to steer towards this point. The output of the low-level controller is a pystk.Action. You can specify:

1. pystk.Action.steer the steering angle of the kart normalized to -1 … 1
2. pystk.Action.acceleration the acceleration of the kart normalized to 0 … 1
3. pystk.Action.brake boolean indicator for braking
4. pystk.Action.drift a special action that makes the kart drift, useful for tight turns
5. pystk.Action.nitro burns nitro for fast acceleration
Implement your controller in the control function in controller.py. You do not need any deep learning to design this low-level controller. You may use numpy instead of pytorch if you wish.

Once you finish, you could test your controller using
> python3 -m SuperTuxKart.controller [TRACK_NAME] -v

Hint: Skid if the steering angle is too large.

Hint: Target a constant velocity.

Hint: Steering and relative aim point use different units. Use the aim point and a tuned scaling factor to select the amount of normalized steering.

Hint: Make sure that your controller is able to complete all levels before proceeding to the next part of the homework because you will use your controller to build the training set for your planner.


### Planner

In the second part, you will train a planner to predict the aim point. The planner takes as input an image and outputs the aim point in the image coordinate. Your controller then maps those aim points to actions.

####Data
Use your low-level controller to collect a training set for the planner.

> python3 -m SuperTuxKart.utils zengarden lighthouse hacienda snowtuxpeak cornfield_crossing scotland

We highly recommend you limit yourself to the above training levels, adding additional training levels may create an unbalanced training set and lead to issues with the final test_grader.

This function creates a dataset of images and corresponding aim points in drive_data. You can visualize the data using

>python3 -m SuperTuxKart.visualize_data drive_data
> 
Below are a few examples from the master-solution controller.

![Example](https://github.com/anki08/CV_Project/blob/main/SuperTuxKart/data.png )


Model
Implement your planner model in Planner class of planner.py. Your planner model is a torch.nn.Module that takes as input an image tensor and outputs the aiming point in image coordinates (x:0..127, y:0..95). We recommend using an encoder-decoder structure to predict a heatmap and extract the peak using a spatial argmax layer in utils.py. Complete the training code in train.py and train your model using python3 -m homework.train.

Vision-Based Driving
Once you completed everything, use

>python3 -m SuperTuxKart.planner [TRACK_NAME] -v

to drive with your CNN planner and controller.

The red circle in the image below is being predicted using the trained master-solution planner network as a substitute for the ground truth aim point used previously.

![Example](https://github.com/anki08/CV_Project/blob/main/SuperTuxKart/planner.png )
