# Humerus-Bone-Fracture-Detection
Transfer learning is a very powerful and bleeding-edge tool to achieve high accuracies on classification tasks on images when the data and the computational power is in limited supply.

Here I attempted to create an Custom Deep Learning model using Deep Neural Networks with intermediate convolutional layers to classify X-ray images of humerus bone fracture from the ones that are not fractured.

As I pass an input image, an output is given as a “Positive” or “Negative” label. The input data is in the form of X Ray images of the Humerus bones. 
Hence, the effort is to train a Supervised Learning model with the data to give correct label to the input image inorder to predict a fracture. 
Some preprocessing of the data like converting RGB images to Grayscale, switching between channels_first & channels_last depending upon the backend Deep Learning engine that will be employed for computations will have to be done.
This repository contains a Keras implementation of a 121 layer Densenet Model on [MURA dataset](https://stanfordmlgroup.github.io/competitions/mura/) with additional 3 Densely Connected Layers at top. 

I achieved an accuracy of about 79% after disconnecting the top 60 layers of the pre-trained model and let it train on the new dataset from a ground truth of no more than being a mere chance (i.e. 50%).

The accuracy improved to ~85% after playing aroung with different hyperparameters and got an understanding of what the model is learning by implementing Class Activation Maps on the gradients of intermediate convolution and activation layers. 

Here, I trained the Densenet on XR_HUMERUS of the dataset for 40 epochs with a batch size of 16.

You can train the model by running train__humerus_fracture_detection_keras_model.ipynb Jupyter Notebook with necessary directory changes. 
You can load the trained model by running load_humerus_fracture_detection_keras_model.ipynb Jupyter Notebook.
You can visualize model's learning at different intermediate layers by running visualize__humerus_fracture_detection_keras_model_cmap.ipynb Jupyter notebook.

Few visualizations are available in layer outputs directory. 

The notebook can be used to implement other available pre-trained models from Tensorflow Keras for Transfer Learning.

