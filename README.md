# Meter Detection

A system designed to find malfunctional meters in a residential area.
## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

```
Give examples
```

## Explanations for each file

### Data Preprocessing and Analysis Part

Our raw data is in fodler sitaiqu including the usage[kilowatt_everyday_2year.xlsx](https://github.com/minoriwww/MeterDetection/blob/master/sitaiqu/kilowatt_everyday_2year.xlsx), the current[electriccurrent_hours_2year.xlsx](https://github.com/minoriwww/MeterDetection/blob/master/sitaiqu/electriccurrent_hours_2year.xlsx) and the voltage[voltage_hours_2year.xlsx](https://github.com/minoriwww/MeterDetection/blob/master/sitaiqu/voltage_hours_2year.xlsx).

Data processing is accomplished in [data_processing0.py](https://github.com/minoriwww/MeterDetection/blob/master/data_processing0.py)

### Residential Area’s Error Prediction Task

[input.py](https://github.com/minoriwww/MeterDetection/blob/master/input.py) will generate the input for lstm.

[more_lstm.py](https://github.com/minoriwww/MeterDetection/blob/master/more_lstm.py) is used to compare the result in different sequence length. Hence, in order to exlude the contingency, we choose to predict 10 times for each sequence length in [k_lstm.py](https://github.com/minoriwww/MeterDetection/blob/master/k_lstm.py) and [draw_k_lstm.py](https://github.com/minoriwww/MeterDetection/blob/master/draw_k_lstm.py).

The comparision of classical methods is accomplished in [svr.py](https://github.com/minoriwww/MeterDetection/blob/master/svr.py). 

### Malfunction-injected Residential Area Detection Task
We generated our data of residential area with malfunctional meters in [bomb.py](https://github.com/minoriwww/MeterDetection/blob/master/bomb.py).

The detection task is finished in [check.py](https://github.com/minoriwww/MeterDetection/blob/master/check.py).

### Malfunctional Submeter Classification Task

We generated our data in [samples.py](https://github.com/minoriwww/MeterDetection/blob/master/samples.py), which imported [single_bomb_wave.py](https://github.com/minoriwww/MeterDetection/blob/master/single_bomb_wave.py) and (single_input_wave.py)[https://github.com/minoriwww/MeterDetection/blob/master/single_input_wave.py].

The classification task is accomplished in [combine_model.py](https://github.com/minoriwww/MeterDetection/blob/master/combine_model.py).

To test the performance of different proportions of malfunctional meters, we did some comparision in [change_bome_rate.py](https://github.com/minoriwww/MeterDetection/blob/master/change_bomb_rate.py).


## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors


See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments
