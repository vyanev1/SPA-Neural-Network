import os
from typing import List

import numpy as np
import pandas as pd
from keras import backend as k
from keras.models import load_model
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM

from generate_data import get_combined_data, split_two_halves, split_input_output, get_column_names
from image_processing import three_markers
from lstm_train_postprocessing import plot_force_predictions, plot_positional_predictions, plot_curvature_predictions, \
    plot_positional_errors


def split_train_test(X_in: np.ndarray, y_in: np.ndarray, train_p: float):
    train_size = int(X_in.shape[0] * train_p)

    train_X, test_X = X_in[:train_size], X_in[train_size:]
    train_Y, test_Y = y_in[:train_size], y_in[train_size:]

    return train_X, test_X, train_Y, test_Y


def generate_predictions_df(model, inputs, column_names: List[str]):
    predictions = pd.DataFrame()
    for input_val in inputs:
        input_row = np.asarray([input_val])
        input_row = input_row.reshape(input_row.shape[0], 1, input_row.shape[-1])
        what: np.ndarray = model.predict(input_row)
        if df_num == 1:
            X_coords_predicted, Y_coords_predicted = split_two_halves(what)
            all_coords = [(int(x), int(y)) for x, y in list(zip(X_coords_predicted, Y_coords_predicted))]
            prediction = np.asarray(list(input_val) + all_coords, dtype=object).reshape(1, len(all_coords) + len(input_val))
            predictions = predictions.append(pd.DataFrame(prediction, columns=column_names))
        else:
            prediction = np.asarray(list(input_val) + list(what.flatten())).reshape(1, len(input_val) + what.shape[-1])
            predictions = predictions.append(pd.DataFrame(prediction, columns=column_names))
    return predictions


def train_model(X: np.ndarray, y: np.ndarray):
    # split into train/test
    x_train, x_test, y_train, y_test = split_train_test(X, y, 0.8)

    # reshape into (n_samples, time_steps, features)
    features = max(1, len(x_train.shape) > 1 and x_train.shape[-1])
    x_train = x_train.reshape(x_train.shape[0], 1, features)
    x_test = x_test.reshape(x_test.shape[0], 1, features)
    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

    # define model
    model = Sequential()
    model.add(LSTM(250, activation='relu', kernel_initializer='he_normal', input_shape=(1, features)))
    model.add(Dense(200, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(150, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(100, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(50, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(y_train.shape[1]))

    # compile the model
    model.compile(optimizer='adam', loss='mse', metrics='mae')

    # fit the model
    model.fit(x_train, y_train, epochs=500, batch_size=64, verbose=2, validation_data=(x_test, y_test))

    # evaluate the model
    mse, mae = model.evaluate(x_test, y_test, verbose=0)
    print('MSE: %.3f, RMSE: %.3f, MAE: %.3f' % (mse, np.sqrt(mse), mae))

    return model


if __name__ == "__main__":
    # load the datasets
    dfs = get_combined_data(three_markers)

    for df_num in range(len(dfs)):
        # Get the dataset
        df = dfs[df_num]

        model_file_name = f"saved_models/model_{df_num}_{'three_markers' if three_markers else 'all_markers'}"
        if os.path.exists(model_file_name):
            k.clear_session()
            # Load the model from the saved file
            model = load_model(model_file_name)
        else:
            # Shuffle the data
            df_shuffled = df.sample(frac=1).reset_index(drop=True)

            # split into input/ouput samples
            X, y = split_input_output(df_shuffled)
            print(X.shape, y.shape)

            # Format the data and train the neural network
            model = train_model(X, y)

            # Save the model
            # model.save(model_file_name)

        # Generate a dataframe of predictions for all inputs
        input_pressures = [0, 1, 2, 3, 4, 5, 6]
        input_distances = [10, 20, 30]
        inputs = [(pressure, distance) for distance in input_distances for pressure in input_pressures]

        predictions = generate_predictions_df(model, inputs, get_column_names(df, df_num, three_markers))

        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.expand_frame_repr',
                               False):
            print('Predictions: ')
            print(predictions)

        if df_num == 0:
            if three_markers:
                df, _, _ = get_combined_data(three_markers=False)
            plot_curvature_predictions(df, predictions)
        elif df_num == 1:
            plot_positional_predictions(df, predictions)
            plot_positional_errors(df, predictions)
        elif df_num == 2:
            plot_force_predictions(df, predictions)
