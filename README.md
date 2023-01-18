# Meter Detection

Detecting malfunctional smart meters based on electricity usage and targeting them for replacement can save significant resources. For this purpose, we developed a novel deep-learning method for malfunctional smart meter detection based on long short-term memory (LSTM) and a modified convolutional neural network (CNN). Our method uses LSTM to predict the reading of a master meter based on data collected from submeters. If the predicted value is significantly different from master meter reading data over a period of time, the diagnosis part will be activated, classifying every submeter to identify the malfunctional submeter based on CNN. We propose a time series-recurrence plot (TS-RP) CNN, by combining the sequential raw data of electricity and its recurrence plots in the phase space as dual input branches of CNN.

**For more details, please refer to the [paper](https://ieeexplore.ieee.org/abstract/document/9300285).**

If you are using our work in your research, please cite us as
```
@ARTICLE{9300285,

  author={Liu, Ming and Liu, Dongpeng and Sun, Guangyu and Zhao, Yi and Wang, Duolin and Liu, Fangxing and Fang, Xiang and He, Qing and Xu, Dong},

  journal={IEEE Industrial Electronics Magazine}, 

  title={Deep Learning Detection of Inaccurate Smart Electricity Meters: A Case Study}, 

  year={2020},

  volume={14},

  number={4},

  pages={79-90},

  doi={10.1109/MIE.2020.3026197}}

```

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

```
Keras=2.2
tensorflow=1.9
```

## Explanations for each file

### Data

Our raw data is in fodler sitaiqu including the usage([kilowatt_everyday_2year.xlsx](https://github.com/minoriwww/MeterDetection/blob/master/sitaiqu/kilowatt_everyday_2year.xlsx)), the current([electriccurrent_hours_2year.xlsx](https://github.com/minoriwww/MeterDetection/blob/master/sitaiqu/electriccurrent_hours_2year.xlsx)) and the voltage([voltage_hours_2year.xlsx](https://github.com/minoriwww/MeterDetection/blob/master/sitaiqu/voltage_hours_2year.xlsx)).

Data processing is accomplished in [data_processing0.py](https://github.com/minoriwww/MeterDetection/blob/master/data_processing0.py)

### Residential Area’s Error Prediction Task

[input.py](https://github.com/minoriwww/MeterDetection/blob/master/input.py) will generate the input for lstm.

[more_lstm.py](https://github.com/minoriwww/MeterDetection/blob/master/more_lstm.py) is used to compare the result in different sequence length. Hence, in order to exlude the contingency, we choose to predict 10 times for each sequence length in [k_lstm.py](https://github.com/minoriwww/MeterDetection/blob/master/k_lstm.py) and [draw_k_lstm.py](https://github.com/minoriwww/MeterDetection/blob/master/draw_k_lstm.py).

The comparision of classical methods is accomplished in [svr.py](https://github.com/minoriwww/MeterDetection/blob/master/svr.py). 

### Malfunction-injected Residential Area Detection Task
We generated our data of residential area with malfunctional meters in [bomb.py](https://github.com/minoriwww/MeterDetection/blob/master/bomb.py).

The detection task is finished in [check.py](https://github.com/minoriwww/MeterDetection/blob/master/check.py).

### Malfunctional Submeter Classification Task

We generated our data in [samples.py](https://github.com/minoriwww/MeterDetection/blob/master/samples.py), which imported [single_bomb_wave.py](https://github.com/minoriwww/MeterDetection/blob/master/single_bomb_wave.py) and [single_input_wave.py](https://github.com/minoriwww/MeterDetection/blob/master/single_input_wave.py).

The classification task is accomplished in [combine_model.py](https://github.com/minoriwww/MeterDetection/blob/master/combine_model.py).

To test the performance of different proportions of malfunctional meters, we did some comparision in [change_bome_rate.py](https://github.com/minoriwww/MeterDetection/blob/master/change_bomb_rate.py).


