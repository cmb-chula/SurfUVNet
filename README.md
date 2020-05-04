# SurfUVNet: Surface ultraviolet radiation forecasting with deep neural network

Our model performance for next-day anti-psoriasis-weighted ultraviolet forecasting with different input sizes (7, 14, or 21 days) on 2018 or 2019 ultraviolet radiation data during the summer and winter periods in Thailand (October-March).

| Model   | MAPE (2018) |  MAPE (2019) |
| ------------- |:-------------:|:-------------:|
| Seq2Seq-7  | 10.18±0.53 | 10.60±0.34 |
| Seq2Seq-14  | 10.41±0.43 | 10.51±0.41 |
| Seq2Seq-21  |  11.35±1.64 | 11.19±0.33 |

## Dependencies

This project use Python 3.5.2 with the following dependencies:

* [Tensorflow 1.14.0 (up to 1.15.2)](https://www.tensorflow.org/)
* [Keras 2.2.5](https://keras.io/)

A list of all required python packages can be found in `requirement.txt`

To install dependencies, run

`pip3 install -r requirements.txt`

## How to train model?

For training your own model, we have provided a jupyter notebook, `Training_example.ipynb`, for handling the input and output data before feeding them to the model. 

To launch a jupyter notebook server after installing all required python packages, run the command below in repo folder

`jupyter notebook` or `jupyter notebook --port=7000` if you want to open jupyter server in port 7000 

This will start your server at the URL `localhost:7000`. Copy and paste this URL into your web browser.

## How to make a UV forecast?

We provide the instruction for making UV forecast in `Prediction_example.ipynb` with example data.

#### Note: The model provided in this repository has an encoder input length of 1190 = 14 days * 85 time steps (10-minute intervals from 5am to 7pm) and a decoder input length of 85 = 1 day x 85 time steps.

If you want to make a forecast 

* using input data from a different time period (e.g., other than 14 days prior to the forecast date), or
* for a different time period (e.g., other than the next day), or
* for a different weighted spectrum (e.g., other than anti-psoriasis),

then you have to train your own new model.
