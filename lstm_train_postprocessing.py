from matplotlib import pyplot as plt

from generate_data import DISTANCE, PRESSURE, FORCE, INPUT_COLUMNS, split_input_output, split_two_halves
from pandas import DataFrame

import numpy as np

from image_processing import three_markers

input_distances = [10, 20, 30]
input_pressures = [0, 1, 2, 3, 4, 5, 6]


def plot_force_predictions(df: DataFrame, predictions: DataFrame):
    fig, axs = plt.subplots(1, len(input_distances))
    for i in range(len(input_distances)):
        pressure_force_measured = df[df[DISTANCE] == input_distances[i]] \
            .drop(DISTANCE, axis=1) \
            .groupby(PRESSURE, as_index=False).mean()

        pressure_force_predicted = predictions[predictions[DISTANCE] == input_distances[i]] \
            .drop(DISTANCE, axis=1) \
            .groupby(PRESSURE, as_index=False).mean()

        prediction_error = abs(pressure_force_predicted[FORCE].values - pressure_force_measured[FORCE].values)

        ax2 = axs[i].twinx()
        ax2.fill_between(
            pressure_force_measured[PRESSURE].values,
            prediction_error,
            color="0.8",
            alpha=0.3,
            label="Prediction Error"
        )
        ax2.set_ylabel('Prediction error (N)')
        ax2.set_xlim(0, 6)
        ax2.set_ylim(0, 0.1)
        ax2.legend(loc='upper right')

        axs[i].plot(
            pressure_force_measured[PRESSURE].values,
            pressure_force_measured[FORCE].values,
            "r-",
            label="Measured force",
        )
        axs[i].plot(
            pressure_force_predicted[PRESSURE].values,
            pressure_force_predicted[FORCE].values,
            "b-",
            label="Predicted force"
        )
        axs[i].grid(True)
        axs[i].set_title(f'Distance: {input_distances[i]} mm')
        axs[i].set_xlabel('Pressure (pA)')
        axs[i].set_ylabel('Force (N)')
        axs[i].set_xlim(0, 6)
        axs[i].set_ylim(0, 1)
        axs[i].legend(loc='upper left')
    fig.tight_layout()
    plt.show()


def plot_positional_predictions(df: DataFrame, predictions: DataFrame):
    for distance in input_distances:
        fig, axs = plt.subplots(1, 2)
        fig.suptitle(f'Distance: {distance} mm')
        for pressure_val in input_pressures:
            X_coords_predicted, Y_coords_predicted = np.asarray(list(zip(
                *predictions.loc[(predictions[DISTANCE] == distance) & (predictions[PRESSURE] == pressure_val)]
                .drop(INPUT_COLUMNS, axis=1).values.flatten()
            )))

            _, y = split_input_output(df.loc[(df[DISTANCE] == distance) & (df[PRESSURE] == pressure_val)])
            X_coords_actual, Y_coords_actual = split_two_halves(np.mean(y, axis=0))

            color = (1.0, (6.0 - pressure_val) / 6.0, 0.0)

            axs[0].plot(
                X_coords_predicted,
                -1 * Y_coords_predicted,
                color=color,
                label=f"{pressure_val} pA"
            )
            axs[0].set_title('Predicted positions')
            axs[0].set_xlabel('X coordinate')
            axs[0].set_ylabel('Y coordinate')
            axs[0].set_xlim(-250, 150)
            axs[0].set_ylim(-450, -50)
            axs[0].legend(loc='upper left')

            axs[1].plot(
                X_coords_actual,
                -1 * Y_coords_actual,
                color=color,
                label=f"{pressure_val} pA"
            )
            axs[1].set_title('Measured positions')
            axs[1].set_xlabel('X coordinate')
            axs[1].set_ylabel('Y coordinate')
            axs[1].set_xlim(-250, 150)
            axs[1].set_ylim(-450, -50)
            axs[1].legend(loc='upper left')
        fig.tight_layout()
        plt.show()


def plot_curvature_predictions(df: DataFrame, predictions: DataFrame):
    for distance in input_distances:
        predicted_curvatures = predictions.loc[(predictions[DISTANCE] == distance)] \
            .drop(INPUT_COLUMNS, axis=1)
        measured_curvatures = df.loc[(df[DISTANCE] == distance)] \
            .groupby(PRESSURE, as_index=False).mean() \
            .drop(INPUT_COLUMNS, axis=1)
        fig, axs = plt.subplots(predicted_curvatures.shape[1], 1)
        fig.suptitle(f'Distance: {distance} mm')
        if three_markers:
            axs = [axs]
        for i in range(predicted_curvatures.shape[1]):
            prediction_error = abs(predicted_curvatures[f'curvature {i+1}'].values
                                   - measured_curvatures[f'curvature {i+1}'].values)
            ax2 = axs[i].twinx()
            ax2.fill_between(
                input_pressures,
                prediction_error,
                color="0.8",
                alpha=0.3,
                label="Prediction Error"
            )
            ax2.set_ylabel('Prediction error')
            ax2.set_xlim(0, 6)
            ax2.set_ylim(0, 0.015)
            ax2.legend(loc='upper right')

            axs[i].plot(
                input_pressures,
                predicted_curvatures[f'curvature {i+1}'].values,
                label=f'Predicted curvature' + (f'for chambers [{i * 3 + 1} - {i * 3 + 3}]' if not three_markers else '')
            )
            axs[i].plot(
                input_pressures,
                measured_curvatures[f'curvature {i+1}'].values,
                label=f'Measured curvature' + (f'for chambers [{i * 3 + 1} - {i * 3 + 3}]' if not three_markers else '')
            )
            axs[i].grid(True)
            axs[i].set_xlabel('Pressure (pA)')
            axs[i].set_ylabel('Curvature')
            axs[i].set_xlim(0, 6)
            axs[i].set_ylim(0, 0.015)
            axs[i].legend(loc='upper left')
        fig.tight_layout()
        plt.show()
