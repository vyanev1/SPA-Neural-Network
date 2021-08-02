from typing import List

import cv2
import numpy as np
import pandas as pd
from tensorflow.keras import Sequential, regularizers
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from generate_data import get_combined_data, split_two_halves, split_input_output, get_column_names


def split_sequence(sequence, n_steps):
    x = list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence):
            break
        # gather input and output parts of the pattern
        seq_x = sequence[i:]
        x.append(seq_x)
    return np.asarray(x)


def split_train_test(X_in: np.ndarray, y_in: np.ndarray, train_p: float):
    train_size = int(X_in.shape[0] * train_p)

    train_X, test_X = X_in[:train_size], X_in[train_size:]
    train_Y, test_Y = y_in[:train_size], y_in[train_size:]

    return train_X, test_X, train_Y, test_Y


def draw_predictions_vs_actual_points(X_coords_predicted, Y_coords_predicted, X_coords_actual, Y_coords_actual):
    width, height = 1280, 720
    init_X, init_Y = 0.6 * width, 0.2 * height
    blank_img = np.zeros((height, width, 3), np.uint8)
    for i in range(len(X_coords_predicted)):
        cX_p = int(X_coords_predicted[i] + init_X)
        cY_p = int(Y_coords_predicted[i] + init_Y)
        cX_a = int(X_coords_actual[i] + init_X)
        cY_a = int(Y_coords_actual[i] + init_Y)
        cv2.circle(blank_img, (cX_a, cY_a), 1, (125, 125, 125), -1)  # draw actual points in grey color
        cv2.circle(blank_img, (cX_p, cY_p), 1, (255, 255, 255), -1)  # draw predicted points in white color
    cv2.imshow(f"predicted position", blank_img)
    cv2.waitKey(0)


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
    model.add(Dense(200, activation='relu', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.0001)))
    model.add(Dense(150, activation='relu', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.0001)))
    model.add(Dense(100, activation='relu', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.0001)))
    model.add(Dense(50, activation='relu', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.0001)))
    model.add(Dense(y_train.shape[1]))

    # compile the model
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # fit the model
    model.fit(x_train, y_train, epochs=500, batch_size=64, verbose=2, validation_data=(x_test, y_test))

    # evaluate the model
    mse, mae = model.evaluate(x_test, y_test, verbose=0)
    print('MSE: %.3f, RMSE: %.3f, MAE: %.3f' % (mse, np.sqrt(mse), mae))

    return model


if __name__ == "__main__":
    # load the datasets
    dfs = get_combined_data()

    for df_num in reversed(range(len(dfs))):
        # Get the dataset
        df = dfs[df_num]

        # Shuffle the data
        df_shuffled = df.sample(frac=1).reset_index(drop=True)

        # split into input/ouput samples
        X, y = split_input_output(df_shuffled)
        print(X.shape, y.shape)

        # Format the data and train the neural network
        model = train_model(X, y)

        # Generate a dataframe of predictions for all inputs
        input_pressures = [0, 1, 2, 3, 4, 5, 6]
        input_distances = [10, 20]
        inputs = [(pressure, distance) for distance in input_distances for pressure in input_pressures]

        predictions = generate_predictions_df(model, inputs, get_column_names(df, df_num))

        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.expand_frame_repr',
                               False):
            print('Predictions: ')
            print(predictions)

        # Draw predictions vs actual coordinates
        if df_num == 1:
            for pressure_val, distance in inputs:
                X_coords_predicted, Y_coords_predicted = list(zip(
                    *predictions.loc[(predictions["distance_mm"] == distance) & (predictions["pressure"] == pressure_val)]
                    .drop(["pressure", "distance_mm"], axis=1).values.flatten()
                ))

                _, y = split_input_output(df.loc[(df["distance_mm"] == distance) & (df["pressure"] == pressure_val)])
                X_coords_actual, Y_coords_actual = split_two_halves(np.mean(y, axis=0))

                draw_predictions_vs_actual_points(
                    X_coords_predicted, Y_coords_predicted,
                    X_coords_actual, Y_coords_actual
                )
