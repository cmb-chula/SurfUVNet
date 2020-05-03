
from keras import layers, models, optimizers, regularizers, callbacks
from keras import backend as K

def tilted_loss(q,y,y_pred):
    e = (y_pred-y)
    return K.mean(K.maximum(q*e, (q-1)*e))

def create_model(
        encoder_input_length, #length of encoder input
        decoder_input_length, #length of decoder input
        do = 0.2, #dropout ratio
        rdo = 0.2, #dropout ratio for rnn
        size = 85, #size of weighted ultraviolet input in each day
        quantile = 0.33, #quantile value for quantile loss
        channel = 3, #number of variable in each timestep of input
        latent_dim = 256, #number of variable in each timestep of input
    ):

    encoder_inputs = layers.Input(shape=(encoder_input_length, channel)) 
    biencoder = layers.Bidirectional(layers.LSTM(latent_dim, dropout=rdo, return_sequences=True))
    biencoder_output = biencoder(encoder_inputs)
    encoder,state_h, state_c = layers.LSTM(latent_dim, dropout=rdo, return_state=True)(biencoder_output)
    
    encoder_states = [state_h, state_c]

    decoder_inputs = layers.Input(shape=(decoder_input_length, channel)) 

    decoder_lstm = layers.LSTM(latent_dim, dropout=rdo, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                        initial_state=encoder_states)

    decoder_dense_1 = layers.Dense(latent_dim//2, activation="sigmoid")
    decoder_outputs = decoder_dense_1(decoder_outputs)
    decoder_dropout_1 = layers.Dropout(do)
    decoder_outputs = decoder_dropout_1(decoder_outputs)

    decoder_dense_2 = layers.Dense(latent_dim//4, activation="sigmoid")
    decoder_outputs = decoder_dense_2(decoder_outputs)
    decoder_dropout_2 = layers.Dropout(do)
    decoder_outputs = decoder_dropout_2(decoder_outputs)

    decoder_dense_3 = layers.Dense(1, activation="sigmoid")
    decoder_outputs = decoder_dense_3(decoder_outputs)

    model = models.Model([encoder_inputs, decoder_inputs], decoder_outputs)

    return model