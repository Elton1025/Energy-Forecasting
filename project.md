Accurate energy consumption forecasting is important for optimizing power grid operations, reducing costs, and
ensuring sustainable energy distribution. This project tackles short-term load forecasting, predicting future power
usage based on historical data by framing it as a multivariate time-series forecasting task. We will utilize historical
energy usage data and temporal features to predict household power consumption 24 hours into the future.
To evaluate the most effective approach, our team will implement and compare three deep learning architectures
spanning different model families. We will establish a classical Multi-Layer Perceptron as our baseline to capture
basic non-linear relationships. Against this, we will compare a 1D-Convolutional Neural Network to extract local
temporal features, and a Long Short-Term Memory network to capture long-term sequential dependencies. The
models will be evaluated using a common protocol featuring the same train/validation/test splits and standard
regression metrics to ensure a meaningful comparison of how different neural network families process sequential
data.
Dataset: UCI Individual Household Electric Power Consumption Dataset.
Source: https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption
Details: This dataset contains 20,75,259 multivariate time series measurements gathered in a house in Sceaux,
France, over 47 months. It includes minute averaged active power (in kilowatts), reactive power, voltage, and global
intensity, alongside date and time features. The dataset will be preprocessed and aggregated into hourly intervals
to train our forecasting models.
Proposed Models for Comparison:
1. Classical Baseline: Multi Layer Perceptron (MLP)
2. CNN Variant: 1D-Convolutional Neural Network (1D-CNN)
3. RNN/LSTM Variant: Long Short Term Memory (LSTM) network
Model Assignment:
 Data Preprocessing Pipeline, Multi Layer Perceptron (MLP) baseline, and 1D-Convolutional Neural Network (1D-CNN)
 Long Short Term Memory (LSTM) network and final common metrics evaluation
Existing State of the-Art Methods Related to the Proposed Project:
Current state-of-the-art methods for multivariate time-series forecasting utilize hybrid architectures (such as CNNLSTMs) and pure attention-based mechanisms (Transformers). Transformers, initially designed for Natural
Language Processing, are currently achieving state-of-the-art results in short-term load forecasting by effectively
modeling long-range dependencies without the vanishing gradient problems common in traditional Recurrent
Neural Networks.
Objectives:
1. To accurately predict household active electrical power consumption up to 24 hours into the future.
2. To comparatively analyze the predictive performance of classical, convolutional, and recurrent deep
learning architectures on time-series load data. 

