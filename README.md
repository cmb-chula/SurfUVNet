# Surface-Ultraviolet-Radiation-Forecasting-for-Clinical-Applications-with-Deep-Neural-Network


#### Note: model weight provided in this repo is for model that has encoder input length = 1190 (14 day * 85 timestep) and decoder input length = 85

Our model performance for weighted ultraviolet forecasting with different day interval length of input between 8 AM to 4 PM

| Model   | MAPE (2018) |  MAPE (2019) |
| ------------- |:-------------:|:-------------:|
| Seq2Seq-7  | 10.18±0.53 | 10.60±0.34 |
| Seq2Seq-14  | 10.41±0.43 | 10.51±0.41 |
| Seq2Seq-21  |  11.35±1.64 | 11.19±0.33 |

## Dependencies

This project use Python 3.5.2 with the following lib dependencies:

* [Tensorflow 1.14.0 (up to 1.15.2)](https://www.tensorflow.org/)
* [Keras 2.2.5](https://keras.io/)

A list of all python package can be found in `requirement.txt`

To install dependencies, run

`pip3 install -r requirements.txt`

## How to train model?

For training your own model, we recommend you to use jupyter notebook to handle the input and output before feeding to the model. 

To running jupyter notebook server after install requirements , run the command below in repo folder

`jupyter notebook` or `jupyter notebook --port=7000` if you want to open jupyter server in port 7000 

This command will open your server and give an jupyter url ex. localhost:7000. Copy and paste the url in your web browser. You will see list of file inside repo folder.

The instruction for training model is in `Training_example.ipynb`.


## How to forecast?

We provide forecasting instruction in `Prediction_example.ipynb` with generated example data.

 However, we provide weight for Seq2Seq-14 model that has encoder input length = 1190 (14 day * 85 timestep) and decoder input length = 85 with output length = 85. If you want to forecast with different input and output length, you have to train your own model.

## Citation
