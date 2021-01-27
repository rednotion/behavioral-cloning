# **Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

In this writeup, I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  


[//]: # (Image References)
[loss_graph]: ./loss_graphs/model.png "Loss graph"

---
## Code Structure
The project includes the following files:
* `model.py` containing the script to create and train the model
* `drive.py` for driving the car in autonomous mode
* `model.h5` containing a trained convolution neural network 
* `writeup_report.md` summarizing the results

### Training the model
The model can be trained with additional flags setting the number of epochs, batch size, and file name.
```sh
python model.py --epochs=50 --name=model --batchsize=32
```
The `model.py` file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

At the end of the python script, it will save the
- model to the to the `/models` directory
- loss graph (train vs validation set) to the `/loss_graphs` directory.

### Using the model
Using the Udacity provided simulator and the drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py models/model.h5 run_name
```

## Model Architecture and Training Strategy

### Final Model Architecture

The final model has the following features:
- **Preprocessing layers**: cropping of the image, normalization of pixels
-  **Convolutional neural network**: 5 Conv2D layers with filter sizes of 3x3 and 5x5, and depths from 24 to 64 (see lines 164-176 in `model.py`)
- **Non-linearity**: Relu activation function
- **Regularization**: Dropout layers are included after each convolutional layer to prevent overfitting
- **Optimization of parameters**: Using ADAM optimizer so that learning rate does not have to be manually selected
- **Train/val/test split**: Model was trained and validated on different datasets to prevent overfitting. It was then tested on the simulator
- **Appropriate training data**: Training data was chosen to keep the vehicle on the road, including center lane driving, and images from the left and right cameras. 
- Model was trained for **30 epochs** with a **batch size of 32**

```python
model = Sequential()
# <-- preprocessing goes here --> #
model.add(Conv2D(filters=24, kernel_size=5, strides=(2, 2), activation='relu'))
model.add(Dropout(0.5))
model.add(Conv2D(filters=36, kernel_size=5, strides=(2, 2), activation='relu'))
model.add(Dropout(0.5))
model.add(Conv2D(filters=48, kernel_size=5, strides=(2, 2), activation='relu'))
model.add(Dropout(0.5))
model.add(Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=3, strides=(1, 1), activation='relu'))
model.add(Flatten())  # 512
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=56, activation='relu'))
model.add(Dense(units=10, activation='relu'))
model.add(Dense(units=1))
```

And this is the loss graph for the train and validation datasets for the 50 epochs of training:
![][loss_graph]


### Solution Design Process

#### 1. Getting the data
First, I used the provided dataset and augmented it to increase the dataset size and variety. In particular I:
- Kept the main image and steering
- Used the left camera image and reduced the steering by `UNDER_STEER_DELTA`
- Used the right camera image and increased the steering by `OVER_STEER_DELTA`
- Used the flipped version of the main image and multiplied the steering by `-1`

For the L and R images (and corresponding deltas), I could set a fixed directional change for L/R **because all the turns in the main track were left turns**. It is important to note that this can't be applied for data collected from the second/advanced track, as there are both left and right turns.

`UNDER_STEER_DELTA` and `OVER_STEER_DELTA` were manually tuned by training the network for a couple of epochs, and then adjusting based on the performance in the simulator.

Finally, I also collected some of my own data. I discuss the use of this data in section 4 *Attempts to improve driving behavior* below.
- Main track: One drive through the track
- Main track: "swerving"/"recovery" data for some portions of the track
- Advanced track: One drive through the track

#### 2. Processing and data split
After the collection process, I had 32,148 of data points. I shuffled and then kept 33% of the data for validation, which would help determine whether I was overfitting during the training process. 

I also preprocessed this data by cropping out some pixels from the top and bottom of the camera image, and standardized the pixel values
```python
model.add(Cropping2D(cropping=((crop_top, crop_bottom), (0, 0)), input_shape=(160, 320, 3)))
model.add(Lambda(lambda x: (x / 255.0) - 0.5,
                 input_shape=(n_rows, n_cols, n_channels),
                 output_shape=(n_rows, n_cols, n_channels)))
```
#### 3. Deriving the model
The overall strategy for deriving a model architecture was to create a deep convolutional neural network, since we were dealing with images. I first took reference from the sample architecture on the [Nvidia blog on self-driving cars](https://developer.nvidia.com/blog/deep-learning-self-driving-cars/), and then iterated from there. In particular, I noticed
- That the conv features get deeper as we go further into the model (start with 3 output layers, and end with 64)
- Dense, FC layers after the convolutional layers

Because of the number of parameters, I also added **dropout layers** to help with regularization and overfitting. I also used **relu** to introduce non-linearity. I also used an ADAM optimizer so that manually training the learning rate wasn't necessary. 

For the number of epochs, I monitored the trend in the **training vs validation** loss, as evidenced in the graph earlier. I noticed that at 10 epochs the validation loss was still decreasing, but at 30 epochs it starts to plateau, so I stopped the training there.

#### 4. Attempts to improve driving behavior
 After training the model on the above data, I noticed it performed pretty well, and only struggled along big curves, where it would run onto the lane lines a bit (but not go off road).
- I collected additional "swerving"/"recovery" data (where the car is too far off and attempts to navigate back to center), but the addition of the data actually made the car performance worse
- I also collected some data from the 2nd track to try and improve the generalizability of the model, but it also made the performance worse on the main track

Thus, the final model still relied on the initial (but augmented dataset).

