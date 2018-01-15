# nvidia-cnn
End-to-End Deep Learning for Self-Driving Cars


|Lake Track Test Run Video|
|:--------:|
|[![Lake Track](http://img.youtube.com/vi/MEBihons2IU/0.jpg)](https://youtu.be/MEBihons2IU)|

# Getting Started
The featured convolutional neural network architecture is based on the [NVIDIA end-to-end model](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/). \
First, you will need [anaconda](https://www.continuum.io/downloads) or [miniconda](https://conda.io/miniconda.html) to install dependencies.

```python
# Use TensorFlow with CPU
conda env create -f environment.yml 

# Use TensorFlow with GPU
conda env create -f environment-gpu.yml
```

### Run the pretrained model

Start up [the Udacity self-driving simulator](https://github.com/udacity/self-driving-car-sim), choose the Lake track, and press the Autonomous Mode button. \
Then, run the model server with the following script:

```python
python drive.py saved_models/track1/model.pb
```

### Train your own model

Collect training images and steering angles using the simulator's Training Mode. Then run the following:

```python
python train.py -d path-to-your-training-data -s path-to-save-tensorflow-checkpoints
```

This will generate checkpoints in the specified save folder. A new epoch will be saved if it has a better validation score than the previous.

## Model Architecture

The design of the network is based on [the NVIDIA model](https://arxiv.org/pdf/1604.07316.pdf). 

- Image normalization
- Convolution: 5x5, filter: 24, strides: 2x2, activation: ELU
- Convolution: 5x5, filter: 36, strides: 2x2, activation: ELU
- Convolution: 5x5, filter: 48, strides: 2x2, activation: ELU
- Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
- Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
- Drop out (0.5)
- Fully connected: neurons: 100, activation: ELU
- Fully connected: neurons:  50, activation: ELU
- Fully connected: neurons:  10, activation: ELU
- Fully connected: neurons:   1 (output)

## Notes

The pretrained model was trained on two laps around the lake track. \
The training data consists of images from the center, left, and right cameras, as well as the corresponding steering angle at each time step.
See [naokishibuya's work](https://github.com/naokishibuya/car-behavioral-cloning) for in-depth data collection.

## Acknowledgements
I'd like to thank [naokishibuya](https://github.com/naokishibuya) for his work, which serves as the basis for drive.py and utils.py, and helped me a lot in understanding the NVIDIA architecture.\
The code for freezing a Tensorflow and then loading it for testing with Autonomous Mode is from [morgangiraud](https://blog.metaflow.fr/tensorflow-how-to-freeze-a-model-and-serve-it-with-a-python-api-d4f3596b3adc).\
The CNN tensorflow code is based on what I learned from [LazyProgrammer's deep learning course](https://lazyprogrammer.me/). 
I highly recommend his content, which covers many theoretical and practical deep learning fundamentals.\
I was inspired to work on this project after watching [Siraj's Youtube video](https://www.youtube.com/watch?v=EaY5QiZwSP4).\
\
I hope this project can help others begin their machine learning journey.
