import matplotlib.pyplot as plt
import numpy as np


def plot_y(y_true,y):
    plt.figure(1)
    x = np.linspace(0,len(y_true),len(y))
    plt.scatter(x,y_true,label="y_true",linewidth=6)
    plt.scatter(x,y,label="y_pred",linewidth=1.5)
    plt.xlabel("time for simulation")
    plt.ylabel("y")


def plot_compare_y(y_true,y):
    plt.figure(2)
    plt.plot(y_true,y)
    plt.xlabel("y_true")
    plt.ylabel("y_estimated")


def plot_metric(score_log,y_ture):
    score_log = np.array(score_log)

    plt.figure(3)
    plt.subplot(2, 2, 1)
    plt.scatter(y_ture,score_log[:, 0], c='#d28ad4')
    plt.ylabel('RMSE')

    plt.subplot(2, 2, 2)
    plt.scatter(y_ture,score_log[:, 1], c='#e765eb')
    plt.ylabel('MAE')

    plt.subplot(2, 2, 3)
    plt.scatter(y_ture,score_log[:, 2], c='#6b016d')
    plt.ylabel('MAPE(%)')

    plt.show()