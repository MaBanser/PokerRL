import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

def create_brier_plot(epochs, train_losses, test_losses, brier_scores, threshold) -> Figure:
    """
    Creates a plot for the training results.

    Args:
        epochs (int): Total number of epochs.
        losses (list): List of loss values for each epoch.
        accuracies (list): List of accuracy values for each epoch.
        threshold (float): Threshold value for accuracy.

    Returns:
        Figure: A matplotlib Figure object containing the plot.
    """
    fig, ax = plt.subplots(2,1,sharex=True,figsize=(8,8))
    ax[0].set_xlim((0,epochs))
    train_loss_plot, = ax[0].plot(train_losses, label='Training Loss')
    test_loss_plot, = ax[0].plot(test_losses, label='Test Loss')
    ax[0].set_ylabel('Loss', fontsize=20)
    ax[0].set_title(f'Epoch: 0', fontweight='bold', fontsize=22) 
    ax[0].set_xticks(np.arange(0, epochs, max(epochs//20, 1)))
    ax[0].legend()
    brier_plot, = ax[1].plot(brier_scores, label='Brier Score')
    ax[1].axhline(threshold, xmax=epochs, c='gray', ls='--', label='Random Guessing')
    ax[1].set_ylim((0, threshold+0.2))
    ax[1].set_ylabel('Brier Score', fontsize=20)
    ax[1].set_xlabel('Epoch', fontsize=20)
    ax[1].set_xticks(np.arange(0, epochs, max(epochs//20, 1)))
    ax[1].legend()
    return fig

def update_brier_plot(fig: Figure, epoch: int, train_losses, test_losses, brier_scores):
    """
    Updates the plots with new data.

    Args:
        fig (Figure): The matplotlib Figure object to update.
        epoch (int): Current epoch number.
        losses (list): List of loss values for each epoch.
        accuracies (list): List of accuracy values for each epoch.
    """
    ax1 = fig.axes[0]
    ax2 = fig.axes[1]
    train_loss_plot = ax1.lines[0]
    test_loss_plot = ax1.lines[1]
    brier_plot = ax2.lines[0]
    train_loss_plot.set_xdata(np.arange(len(train_losses)))
    train_loss_plot.set_ydata(train_losses)
    ax1.set_ylim((0, max(max(train_losses),max(test_losses)) if train_losses else 1))
    test_loss_plot.set_xdata(np.arange(len(test_losses)))
    test_loss_plot.set_ydata(test_losses)
    ax1.set_title(f'Epoch: {epoch+1}', fontweight='bold', fontsize=22)
    brier_plot.set_xdata(np.arange(len(brier_scores)))
    brier_plot.set_ydata(brier_scores)
    fig.canvas.draw()
    fig.canvas.flush_events()


def create_rl_plot(episodes, losses, rewards, avg_reward_over_hundred) -> Figure:
    """
    Creates a plot for the training results.

    Args:
        episode (int): Total number of episodes.
        losses (list): List of loss values for each episode.
        rewards (list): List of reward values for each episode.

    Returns:
        Figure: A matplotlib Figure object containing the plot.
    """
    fig, ax = plt.subplots(2,1,sharex=True,figsize=(8,8))
    ax[0].set_xlim((0,episodes))
    loss_plot, = ax[0].plot(losses, label='Loss')
    ax[0].set_ylabel('Loss', fontsize=20)
    ax[0].set_title(f'Episode: 0', fontweight='bold', fontsize=22) 
    ax[0].set_xticks(np.arange(0, episodes, max(episodes//100, 1)))
    ax[0].legend()
    reward_plot, = ax[1].plot(rewards, label='Reward')
    avg_reward_plot, = ax[1].plot(avg_reward_over_hundred, label='Average reward over last 100')
    ax[1].set_ylabel('Reward', fontsize=20)
    ax[1].set_xlabel('Episode', fontsize=20)
    ax[1].set_xticks(np.arange(0, episodes, max(episodes//100, 1)))
    ax[1].legend()
    return fig

def update_rl_plot(fig: Figure, episode, losses, rewards, avg_reward_over_hundred):
    """
    Updates the plots with new data.

    Args:
        fig (Figure): The matplotlib Figure object to update.
        episode (int): Current episode number.
        losses (list): List of loss values for each episode.
        rewards (list): List of reward values for each episode.
        avg_reward_over_hundred (list): List of average reward values over the last 100 episodes.
    """
    ax1 = fig.axes[0]
    ax2 = fig.axes[1]
    loss_plot = ax1.lines[0]
    reward_plot = ax2.lines[0]
    avg_reward_plot = ax2.lines[1]
    loss_plot.set_xdata(np.arange(len(losses)))
    loss_plot.set_ydata(losses)
    ax1.set_ylim((0, max(losses)) if losses else 1)
    ax1.set_title(f'Epoch: {episode+1}', fontweight='bold', fontsize=22)
    reward_plot.set_xdata(np.arange(len(rewards)))
    reward_plot.set_ydata(rewards)
    avg_reward_plot.set_xdata(np.arange(len(avg_reward_over_hundred)))
    avg_reward_plot.set_ydata(avg_reward_over_hundred)
    fig.canvas.draw()
    fig.canvas.flush_events()
    