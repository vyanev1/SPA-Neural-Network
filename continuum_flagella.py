import os
from collections import defaultdict

import numpy as np
from elastica import BaseSystemCollection, Constraints, Forcing, CallBacks, SlenderBodyTheory, \
    CallBackBaseClass, PositionVerlet, integrate, CosseratRod, OneEndFixedRod, EndpointForces, GravityForces

from continuum_flagella_postprocessing import plot_velocity, plot_video, compute_projected_velocity


class SoftPneumaticActuatorSimulator(BaseSystemCollection, Constraints, Forcing, CallBacks):
    pass


def run_actuator_sim(b_coeff, PLOT_FIGURE=False, SAVE_FIGURE=False, SAVE_VIDEO=False, SAVE_RESULTS=False):

    actuator_sim = SoftPneumaticActuatorSimulator()

    # setting up test params
    n_elem = 12
    start = np.zeros((3,))
    direction = np.array([-1.0, 0.0, 0.0])
    normal = np.array([0.0, -1.0, 0.0])
    base_length = 1.12
    base_radius = 0.075
    density = 1107
    nu = 5.0                # TODO: Viscous damping coefficient =?
    E = 0.66677674782e6     # MPa (OLD_VALUE: 1e7)
    poisson_ratio = 0.5

    # Create and append the cosserat rod to the simulator
    shearable_rod = CosseratRod.straight_rod(
        n_elem,
        start,
        direction,
        normal,
        base_length,
        base_radius,
        density,
        nu,
        E,
        poisson_ratio,
    )

    actuator_sim.append(shearable_rod)

    # Constrain one end of the rod to be fixed in place
    actuator_sim.constrain(shearable_rod).using(
        OneEndFixedRod,
        constrained_position_idx=(0,),
        constrained_director_idx=(0,)
    )

    # Add endpoint forcing
    actuator_sim.add_forcing_to(shearable_rod).using(
        EndpointForces,
        start_force=np.array([0.0, 0.0, 0.0]),
        end_force=np.array([0.0, 0.0, 10.0]),
        ramp_up_time=0.1
    )

    actuator_sim.add_forcing_to(shearable_rod).using(
        EndpointForces,
        start_force=np.array([0.0, 0.0, 0.0]),
        end_force=np.array([5000.0, 0.0, 0.0]),
        ramp_up_time=5
    )

    # Add gravity forces
    actuator_sim.add_forcing_to(shearable_rod).using(
        GravityForces,
        acc_gravity=np.array([-9.80665, 0.0, 0.0])
    )

    period = 1.0

    # Add muscle torque forcing
    # wave_length = b_coeff[-1]
    # actuator_sim.add_forcing_to(shearable_rod).using(
    #     MuscleTorques,
    #     base_length=base_length,
    #     b_coeff=b_coeff[:-1],
    #     period=period,
    #     wave_number=2.0 * np.pi / (wave_length),
    #     phase_shift=0.0,
    #     rest_lengths=shearable_rod.rest_lengths,
    #     ramp_up_time=period,
    #     direction=normal,
    #     with_spline=True,
    # )

    # TODO: Determine slender body forces
    fluid_density = 1.0
    reynolds_number = 1e-4
    dynamic_viscosity = (fluid_density * base_length * base_length) / (period * reynolds_number)
    actuator_sim.add_forcing_to(shearable_rod).using(
        SlenderBodyTheory, dynamic_viscosity=dynamic_viscosity
    )

    # Add call backs
    class SoftPneumaticActuatorCallBack(CallBackBaseClass):
        """
        Call back function for continuum snake
        """

        def __init__(self, step_skip: int, callback_params: dict):
            CallBackBaseClass.__init__(self)
            self.every = step_skip
            self.callback_params = callback_params

        def make_callback(self, system, time, current_step: int):
            if current_step % self.every == 0:
                self.callback_params["time"].append(time)
                self.callback_params["step"].append(current_step)
                self.callback_params["position"].append(system.position_collection.copy())
                self.callback_params["velocity"].append(system.velocity_collection.copy())
                self.callback_params["avg_velocity"].append(system.compute_velocity_center_of_mass())
                self.callback_params["center_of_mass"].append(system.compute_position_center_of_mass())
                return

    pp_list = defaultdict(list)
    actuator_sim.collect_diagnostics(shearable_rod).using(
        SoftPneumaticActuatorCallBack, step_skip=200, callback_params=pp_list
    )

    actuator_sim.finalize()
    timestepper = PositionVerlet()
    # timestepper = PEFRL()

    final_time = 5.0 * period
    dt = 2.5e-5 * period
    total_steps = int(final_time / dt)
    print("Total steps", total_steps)
    integrate(timestepper, actuator_sim, final_time, total_steps)

    if PLOT_FIGURE:
        filename_plot = "soft_pneumatic_actuator_velocity.png"
        plot_velocity(pp_list, period, filename_plot, SAVE_FIGURE)

        if SAVE_VIDEO:
            filename_video = "soft_pneumatic_actuator.mp4"
            plot_video(pp_list, video_name=filename_video, margin=0.2, fps=60)

    if SAVE_RESULTS:
        import pickle

        filename = "soft_pneumatic_actuator.dat"
        file = open(filename, "wb")
        pickle.dump(pp_list, file)
        file.close()

    # Compute the average forward velocity. These will be used for optimization.
    [_, _, avg_forward, avg_lateral] = compute_projected_velocity(pp_list, period)

    return avg_forward, avg_lateral, pp_list


if __name__ == "__main__":
    # Options
    PLOT_FIGURE = True
    SAVE_FIGURE = False
    SAVE_VIDEO = True
    SAVE_RESULTS = False
    CMA_OPTION = False

    if CMA_OPTION:
        import cma

        SAVE_OPTIMIZED_COEFFICIENTS = False

        def optimize_snake(spline_coefficient):
            [avg_forward, _, _] = run_actuator_sim(
                spline_coefficient,
                PLOT_FIGURE=False,
                SAVE_FIGURE=False,
                SAVE_VIDEO=False,
                SAVE_RESULTS=False,
            )
            return -avg_forward

        # Optimize snake for forward velocity. In cma.fmin first input is function
        # to be optimized, second input is initial guess for coefficients you are optimizing
        # for and third input is standard deviation you initially set.
        optimized_spline_coefficients = cma.fmin(optimize_snake, 5 * [0], 0.5)

        # Save the optimized coefficients to a file
        filename_data = "optimized_coefficients.txt"
        if SAVE_OPTIMIZED_COEFFICIENTS:
            assert filename_data != "", "provide a file name for coefficients"
            np.savetxt(filename_data, optimized_spline_coefficients, delimiter=",")

    else:
        if os.path.exists("optimized_coefficients.txt"):
            t_coeff_optimized = np.genfromtxt("optimized_coefficients.txt", delimiter=",")
            wave_length = (0.3866575573648976 * 1.0)  # 1.0 is base length, wave number is 16.25
            t_coeff_optimized = np.hstack((t_coeff_optimized, wave_length))
        else:
            t_coeff_optimized = np.array([17.4, 48.5, 5.4, 14.7, 0.38])

        # run the simulation
        [avg_forward, avg_lateral, pp_list] = run_actuator_sim(
            t_coeff_optimized, PLOT_FIGURE, SAVE_FIGURE, SAVE_VIDEO, SAVE_RESULTS
        )

        print("average forward velocity:", avg_forward)
        print("average forward lateral:", avg_lateral)
