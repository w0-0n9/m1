import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def integrate(data, time):
    '''
    Integrate the provided dataset, data, over the time series, time

    data: dataset to integrate. Its Nth dimension must match the length of the time series
    time: time series to integrate over
    '''
    if data.shape != time.shape:
        time = np.repeat(time, data.shape[1]).reshape(data.shape)
        return (data[:-1,...] + ((data[1:,...] - data[:-1,...]) / 2)).cumsum(axis=0) * (time[1:,...] - time[:-1,...])
    else:
        return (data[:-1] + ((data[1:] - data[:-1]) / 2)).cumsum(axis=0) * (time[1:] - time[:-1])

if __name__ == '__main__':
    # Get data and calculate desired values
    df = pd.read_csv('/Users/jinwoongshin/PycharmProjects/pythonProject/lab8-dataset/ACCELERATION.csv')
    time = df['timestamp'].values
    acc = df[['acceleration', 'noisyacceleration']].values
    vel = np.vstack(([0,0], integrate(acc, time)))
    dist = np.vstack(([0,0], integrate(vel, time)))

    # Print result summaries
    print("Final distances:\n\tReal: {}\tNoisy: {}\tError: {}".format(dist[-1,0], dist[-1,1], abs(dist[-1,1] - dist[-1,0])))

    # Plot results
    fig, axs = plt.subplots(3, sharex=False, sharey=False)
    fig.canvas.manager.set_window_title('Milestone 1')
    fig.tight_layout()

    axs[0].set_title('Acceleration')
    axs[0].set_ylabel('Acceleration (m / s²)')
    axs[0].set_xlabel('Time (s)')
    axs[0].plot(time, acc[:,0], label='Real', color='#00f000')
    axs[0].plot(time, acc[:,1], label='Noisy', color='#f00000')
    axs[0].legend()

    axs[1].set_title('Speed')
    axs[1].set_ylabel('Speed (m / s)')
    axs[1].set_xlabel('Time (s)')
    axs[1].plot(time, vel[:,0], label='Real', color='#00b000')
    axs[1].plot(time, vel[:,1], label='Noisy', color='#b00000')
    axs[1].legend()

    axs[2].set_title('Distance')
    axs[2].set_ylabel('Distance (m)')
    axs[2].set_xlabel('Time (s)')
    axs[2].plot(time, dist[:,0], label='Real', color='#007000')
    axs[2].plot(time, dist[:,1], label='Noisy', color='#700000')
    axs[2].legend()

    plt.show()
