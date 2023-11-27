import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import m1.main as ms1
import m2.main as ms2
import m3.main as ms3

if __name__ == '__main__':
    # Get data from csv
    df = pd.read_csv('/Users/jinwoongshin/PycharmProjects/pythonProject/lab8-dataset/WALKING_AND_TURNING.csv',
                     usecols=['timestamp', 'accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z', 'mag_x',
                              'mag_y', 'mag_z'])

    # Change timestamps to be seconds since start
    df['timestamp'] -= df['timestamp'][0]
    df['timestamp'] /= 10 ** 9
    time = df['timestamp'].to_numpy()

    # -------------------------------------------------------------
    # Turn Detection
    # -------------------------------------------------------------

    # Smooth angular rate of change using an EWMA with specified alpha
    smooth_gyro_z = ms2.smooth_ewma(df['gyro_z'].values, 0.07)
    # Integrate angular rate of change from smoothed gyro_z to get angular displacement
    theta_z = np.concatenate(([0], ms1.integrate(smooth_gyro_z, time)))
    # Detect turns of specified increments from angular rate of change and angular displacement
    cw_turns, ccw_turns = ms3.find_turns(smooth_gyro_z, theta_z, np.pi / 4, 0.125)
    cw_turns = np.array(cw_turns)
    ccw_turns = np.array(ccw_turns)
    print("CW:", cw_turns)
    print("CCW:", ccw_turns)

    # -------------------------------------------------------------
    # Step Detection
    # -------------------------------------------------------------

    threshold = 9.75

    # Smooth data
    # Exponential weighted moving average with alpha=0.02
    smooth_accel_z = ms2.smooth_ewma(df['accel_z'].values, 0.02)

    # Count steps by counting intersections of the smoothed data with a threshold value
    num_steps = 0
    intersection_indices = []
    intersection_times = []
    for i in range(len(time) - 1):
        if smooth_accel_z[i] <= threshold and smooth_accel_z[i + 1] > threshold:
            num_steps += 1
            intersection_indices += [i]
            intersection_times += [time[i]]

    print(num_steps)
    print(intersection_times)

    # -------------------------------------------------------------
    # Assemble Position Data
    # -------------------------------------------------------------

    # Current heading angle (start going north)
    heading = np.pi / 2
    # Location info for plotting
    x_loc = [0]
    y_loc = [0]

    # Variables for tracking position in turn arrays
    cw_idx = 0
    ccw_idx = 0
    for i in range(len(time) - 1):
        step = 0
        # Check if we're stepping forward
        if i in intersection_indices:
            step = 1
        # Adjust heading according to turns
        if len(cw_turns) > 0 and i in cw_turns[:, 0]:
            heading += cw_turns[cw_idx, 1]
            cw_idx += 1
        elif len(ccw_turns) > 0 and i in ccw_turns[:, 0]:
            heading += ccw_turns[ccw_idx, 1]
            ccw_idx += 1
        # Move according to heading and step
        x_loc += [x_loc[-1] + step * np.cos(heading)]
        y_loc += [y_loc[-1] + step * np.sin(heading)]

    # Plot results
    fig, axs = plt.subplots(1)
    fig.canvas.manager.set_window_title('Milestone 4')
    fig.canvas.manager.resize(750, 750)
    plt.ylim(-5, 35)
    plt.xlim(-5, 35)
    axs.set_title('Position')
    axs.set_ylabel('Y (m)')
    axs.set_xlabel('X (m)')

    axs.plot(x_loc, y_loc, label='Position', color='#0000ff', marker='o', markersize=2, alpha=0.5)
    axs.legend()

    plt.show()
