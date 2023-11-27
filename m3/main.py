import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import m1.main as m1
import m2.main as m2

def find_turns(gyro_data, data, turn_increment, threshold=0.25, tolerance=np.pi / 32):
    '''
    Returns two 2D arrays containing the midpoint index and angle of each CW and CCW turn

    data: array containing angular rate of change data
    data: array containing angular displacement data
    turn_increment: the expecte increments of which valid turns will be
        - e.g. turn_increment=pi/2 means valid turns are ..., -pi, -pi/2, 0, pi/2, pi, ...
    threshold: the minimum angular rate of change (rad/s) to be considered the start/end of a turn
    tolerance: the maximum amount by which a recorded turn can be closer to 0 than the final multiple of turn_increment
    '''
    # Limit the tolerance to at most half the turn increment
    tolerance = (turn_increment / 2) if (turn_increment <= tolerance * 2) else tolerance
    cw_bounds, ccw_bounds = get_turn_bounds(gyro_data, threshold)
    cw_turns = []
    ccw_turns = []
    idx = 0

    # Record CW turns
    for i in range(len(cw_bounds)):
        # Bounds and initial angle of the current turn
        idx = cw_bounds[i][0]
        turn_end = cw_bounds[i][1]
        init_angle = data[idx]
        increments = 0
        # Find the next index of data where the angle changes by turn_increment
        idx_tmp = np.argmax(data[idx:turn_end] <= init_angle - turn_increment + tolerance)
        while idx_tmp != 0:
            idx += idx_tmp
            increments += 1
            idx_tmp = np.argmax(data[idx:turn_end] <= init_angle - (increments + 1) * turn_increment + tolerance)
        if increments > 0:
            # Record middle index and turn angle
            cw_turns.append([np.floor((cw_bounds[i][0] + turn_end) / 2).astype(int), -increments * turn_increment])
    # Record CCW turns
    for i in range(len(ccw_bounds)):
        # Bounds and initial angle of the current turn
        idx = ccw_bounds[i][0]
        turn_end = ccw_bounds[i][1]
        init_angle = data[idx]
        increments = 0
        # Find the next index of data where the angle changes by turn_increment
        idx_tmp = np.argmax(data[idx:turn_end] >= init_angle + turn_increment - tolerance)
        while idx_tmp != 0:
            idx += idx_tmp
            increments += 1
            idx_tmp = np.argmax(data[idx:turn_end] >= init_angle + (increments + 1) * turn_increment - tolerance)
        if increments > 0:
            # Record middle index and turn angle
            ccw_turns.append([np.floor((ccw_bounds[i][0] + turn_end) / 2).astype(int), increments * turn_increment])
    return cw_turns, ccw_turns


def get_turn_bounds(data, threshold=0.25):
    '''
    Returns two 2D arrays containing the start and end indices of each CW and CCW turn

    data: array containing angular rate of change data from which turn bounds will be determined
    threshold: the minimum angular rate of change (rad/s) to be considered the start/end of a turn
    '''
    cw_turn_bounds = []
    ccw_turn_bounds = []
    i = 0
    while i < len(data):
        # Step through the data until the start of a turn is reached
        if data[i] <= -threshold:
            # Record the start and (temporary) end bounds of the CW turn
            cw_turn_bounds.append([i, i])
            # Step through the data until the end of the CW turn is reached
            while (i < len(data)) and (data[i] <= -threshold):
                i += 1
            # Record the actual end index of the CW turn
            cw_turn_bounds[-1][1] = i
        if data[i] >= threshold:
            # Record the start and (temporary) end bounds of the CCW turn
            ccw_turn_bounds.append([i, i])
            # Step through the data until the end of the CCW turn is reached
            while (i < len(data)) and (data[i] >= threshold):
                i += 1
            # Record the actual end index of the CCW turn
            ccw_turn_bounds[-1][1] = i
        i += 1
    return cw_turn_bounds, ccw_turn_bounds


if __name__ == '__main__':
    # Get data from csv
    df = pd.read_csv('/Users/jinwoongshin/PycharmProjects/pythonProject/lab8-dataset/TURNING.csv',
                     usecols=['timestamp', 'accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z', 'mag_x',
                              'mag_y', 'mag_z'])

    # Change timestamps to be seconds since start
    df['timestamp'] -= df['timestamp'][0]
    df['timestamp'] /= 10 ** 9
    time = df['timestamp'].to_numpy()

    # Smooth angular rate of change using an EWMA with specified alpha (0.12)
    smooth_gyro_z = m2.smooth_ewma(df['gyro_z'].values, 0.12)
    # Integrate angular rate of change from smoothed gyro_z data to get angular displacement
    theta_z = np.concatenate(([0], m1.integrate(smooth_gyro_z, time)))
    # Detect turns of specified increments from angular rate of change and angular displacement
    cw_turns, ccw_turns = find_turns(smooth_gyro_z, theta_z, np.pi / 2)
    print("CW:", cw_turns)  # DEBUG
    print("CCW:", ccw_turns)  # DEBUG

    # Mask timestamps for plotting turn points
    cw_tstamp = time[np.ravel(cw_turns)[::2].astype(int)]
    ccw_tstamp = time[np.ravel(ccw_turns)[::2].astype(int)]

    # Plot results
    fig, axs = plt.subplots(2, sharex=False, sharey=False)
    fig.canvas.manager.set_window_title('Milestone 3')
    fig.tight_layout()

    axs[0].set_title('Angular Rate of Change')
    axs[0].set_ylabel('Angular Rate of Change (rad / s)')
    axs[0].set_xlabel('Time (s)')
    threshold_line = np.full(df['timestamp'].to_numpy().shape, 0.25)
    axs[0].plot(df['timestamp'].values, threshold_line, label='Threshold (+)', color='#0080ff')
    axs[0].plot(df['timestamp'].values, -threshold_line, label='Threshold (-)', color='#00ff80')
    axs[0].plot(time, df['gyro_z'].values, label='Raw', color='#808000')
    axs[0].plot(time, smooth_gyro_z, label='Smoothed', color='#f00000')
    axs[0].legend()

    axs[1].set_title('Angular Displacement and Turns')
    axs[1].set_ylabel('Angular Displacement (rad)')
    axs[1].set_xlabel('Time (s)')
    axs[1].axhline(color="#000000")
    axs[1].plot(time, np.concatenate(([0], m1.integrate(df['gyro_z'].values, time))), label='Raw', color='#00b000')
    axs[1].plot(time, np.concatenate(([0], m1.integrate(smooth_gyro_z, time))), label='Smoothed', color='#b00000')
    axs[1].vlines(np.concatenate((cw_tstamp, ccw_tstamp)), 0,
                  np.concatenate((np.array(cw_turns)[:, 1], np.array(ccw_turns)[:, 1])), label='Turns', color='#0000ff')
    axs[1].legend()

    plt.show()
