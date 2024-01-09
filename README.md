# Project Overview
The objective of this project is to build a machine learning pipeline which takes an audio segment as an input to predict:
1. whether the audio segment has been recorded indoors or outdoors (the basic task); and
2. the exact recording location in a specific region (the advanced task).  

_**major library used: pandas, numpy, matplotlib, librosa, scikit-learn, tensorflow and keras**_

## The Dataset
This project is based on the MLEnd London Sounds dataset, a dataset of unstructured raw audio files. 
The audio files are recorded in 6 areas in London. For each area, there are 6 specific spots covering both indoor and outdoor areas. 
The label csv file contains the name of the audio file, its recording area and spot and the label whether it belongs to indoor or outdoor.

## Feature Extraction and Normalisation
As the data inputs are unstructured raw audio files, the Python library librosa is used to process the audio files to get numeric features. 
The features extracted are:
- Mel-frequency cepstral coefficients (MFCC) in which each MFCC captures certain details of the spectral envelope of the audio signal;
- root mean square (RMS) value which measures the average loudness;
- zero-crossing rate which is an indicator to measure the fluctation of the audio wave; and
- spectral flatness which quantify how much noise-like a sound is.

## The Model and Results
### The Basic Task
The features are passed into a model using Support Vector Machine (SVM) algorithm. The optimal hypermeters are searched through GridSearch with cross validation. 
The final model has a training accuracy of 88.3% and a validation accuracy of 77.7%.

### The Advanced Task
This task focuses on the audios recorded in the campus area and aims to predict the exact recording spot.
The advanced model use neural network which consists of fully-connected layers.
Dropout layers are added to prevent overfitting.
The final model has a training accuracy of 95.83% and a validation accuracy of 68.42%.
The results imply that the architecture can be further improved.
