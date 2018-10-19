import numpy as np
import matplotlib.pyplot as plt



def prepare_standardplot(title, xlabel):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle(title)
    ax1.set_ylabel('loss')
    ax1.set_xlabel(xlabel)
    ax1.set_yscale('log')
    ax2.set_ylabel('accuracy [% correct]')
    ax2.set_xlabel(xlabel)
    return fig, ax1, ax2

def finalize_standardplot(fig, ax1, ax2):
    ax1handles, ax1labels = ax1.get_legend_handles_labels()
    if len(ax1labels) > 0:
        ax1.legend(ax1handles, ax1labels)
    ax2handles, ax2labels = ax2.get_legend_handles_labels()
    if len(ax2labels) > 0:
        ax2.legend(ax2handles, ax2labels)
    fig.tight_layout()
    plt.subplots_adjust(top=0.9)

def plotCurves(tr_loss, tr_acc, ts_loss, ts_acc, title):
    fig, ax1, ax2 = prepare_standardplot(title, 'epoch')
    ax1.plot(tr_loss, label = "training")
    ax1.plot(ts_loss, label = "validation")
    ax2.plot(tr_acc, label = "training")
    ax2.plot(ts_acc, label = "validation")
    finalize_standardplot(fig, ax1, ax2)
    return fig


def plot_loss(loss_tr, loss_ts):
    
    plt.style.use('seaborn')
    plt.plot(loss_tr, label='Training set', c='blue')
    plt.plot(loss_ts, label='Test set', c='red')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss')
    plt.legend()
    plt.show()
    
    
def plot_acc(acc_tr, acc_ts):
    # Make plot with label prediction accuracy
    plt.plot(acc_tr, label='Training set', c='blue')
    plt.plot(acc_ts, label='Test set', c='red')
    plt.legend()
    plt.xlabel('Iteration')
    plt.ylabel('Prediction Accuracy')
    plt.title('Prediction Accuracy')
    plt.show()
    
    