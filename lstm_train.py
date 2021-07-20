import cv2
import numpy as np
import pandas as pd
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from generate_data import get_combined_data, split_two_halves, split_input_output


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


def split_train_test(X_in: np.ndarray, Y_in: np.ndarray, train_p: float):
    train_size = int(X_in.size * train_p)

    train_X, test_X = X_in[:train_size], X_in[train_size:]
    train_Y, test_Y = Y_in[:train_size], Y_in[train_size:]

    return train_X, test_X, train_Y, test_Y


# load the dataset
curv_and_pos_dfs = get_combined_data()

df_num = 0
for df_num in (2, 1):
    df = curv_and_pos_dfs[df_num - 1]

    # split into samples
    X, y = split_input_output(df)
    print(X.shape, y.shape)

    # split into train/test
    X_train, X_test, y_train, y_test = split_train_test(X, y, 0.8)

    # reshape into (n_samples, time_steps, features)
    X_train = X_train.reshape(X_train.shape[0], 1, 1)
    X_test = X_test.reshape(X_test.shape[0], 1, 1)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    # define model
    model = Sequential()
    model.add(LSTM(250, activation='relu', kernel_initializer='he_normal', input_shape=(X_train.shape[1], 1)))
    model.add(Dense(200, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(150, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(100, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(50, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(y_train.shape[1]))

    # compile the model
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # fit the model
    model.fit(X_train, y_train, epochs=500, batch_size=32, verbose=2, validation_data=(X_test, y_test))

    # evaluate the model
    mse, mae = model.evaluate(X_test, y_test, verbose=0)
    print('MSE: %.3f, RMSE: %.3f, MAE: %.3f' % (mse, np.sqrt(mse), mae))

    if df_num == 1:
        columns = [f"curvature {i+1}" for i in range(4)]
    else:
        columns = [f"chamber {j+1}" for j in range(2, 12)]

    pressures = [0, 1, 2, 3, 4]
    predictions = pd.DataFrame()
    for pressure_val in pressures:
        pressure_row = np.asarray([[pressure_val]])
        pressure_row = pressure_row[0].reshape(pressure_row.shape[0], 1, 1)
        what: np.ndarray = model.predict(pressure_row)
        if df_num == 1:
            prediction = np.concatenate(([pressure_val], what.flatten())).reshape(what.shape[0], what.shape[1] + 1)
            predictions = predictions.append(pd.DataFrame(prediction, columns=["pressure_val"] + columns))
        elif df_num == 2:
            X_coords_predicted, Y_coords_predicted = split_two_halves(what)
            all_coords = [(int(x), int(y)) for x, y in list(zip(X_coords_predicted, Y_coords_predicted))]

            prediction = np.asarray([pressure_val] + all_coords, dtype=object).reshape(1, len(all_coords) + 1)
            predictions = predictions.append(pd.DataFrame(prediction, columns=["pressure_val"] + columns))

            X_coords_actual, Y_coords_actual = split_two_halves(y[pressure_val])

            init_X = 800
            init_Y = 100
            blank_img = np.zeros((720, 1280, 3), np.uint8)
            for i in range(len(X_coords_predicted)):
                cX_p = int(X_coords_predicted[i] + init_X)
                cY_p = int(Y_coords_predicted[i] + init_Y)
                cX_a = int(X_coords_actual[i] + init_X)
                cY_a = int(Y_coords_actual[i] + init_Y)
                cv2.circle(blank_img, (cX_p, cY_p), 1, (255, 255, 255), -1)
                cv2.circle(blank_img, (cX_a, cY_a), 1, (125, 125, 125), -1)
            cv2.imshow(f"predicted position", blank_img)
            cv2.waitKey(0)

    print('Predictions: ')
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.expand_frame_repr', False):
        print(predictions)
