import matplotlib as mpl
mpl.use('Agg') # don't display plots while plotting ("batch mode")
import matplotlib.pyplot as plt

import sys
import os
import pickle


def plot_CustomLoss(dnnTag):

    print('Plotting loss history. - Custom version')

    model_history = None
    pickleFile = './outputs/'+dnnTag+'/model_customHistory.pkl'
    with open(pickleFile, 'rb') as f:
        model_history = pickle.load(f)

    plt.clf()
    fig = plt.figure()
    plt.grid()
    x = (range(len(model_history['train_loss'])+1))[1:]
    plt.plot(x, model_history['train_loss'], label='Training set')
    #plt.plot(x, model_history['valid_loss'], label='Validation set')
    plt.legend(loc='upper right')
    plt.ylabel('Loss (categorical crossentropy)')
    plt.xlabel('Number of training epochs')
    saveFile = './outputs/'+dnnTag+'/plots/loss_custom.pdf'
    fig.savefig(saveFile)
    plt.close()


def plot_Loss(dnnTag):

    print('Plotting loss history. - Buggy Keras version')

    model_history = None
    pickleFile = './outputs/'+dnnTag+'/model_history.pkl'
    with open(pickleFile, 'rb') as f:
        model_history = pickle.load(f)

    plt.clf()
    fig = plt.figure()
    plt.grid()
    x = (range(len(model_history['loss'])+1))[1:]
    plt.plot(x, model_history['loss'], label='Training set')
    plt.plot(x, model_history['val_loss'], label='Validation set')
    plt.legend(loc='upper right')
    plt.ylabel('Loss (categorical crossentropy)')
    plt.xlabel('Number of training epochs')
    saveFile = './outputs/'+dnnTag+'/plots/loss.pdf'
    fig.savefig(saveFile)
    plt.close()


def plot_LossBoth(dnnTag):

    print('Plotting loss history. - Both versions: Keras and custom')

    model_history = None
    pickleFile = './outputs/'+dnnTag+'/model_history.pkl'
    with open(pickleFile, 'rb') as f:
        model_history = pickle.load(f)

    model_customHistory = None
    pickleFileCustom = './outputs/'+dnnTag+'/model_customHistory.pkl'
    with open(pickleFileCustom, 'rb') as f:
        model_customHistory = pickle.load(f)

    plt.clf()
    fig = plt.figure()
    plt.grid()
    x = (range(len(model_history['loss'])+1))[1:]
    plt.plot(x, model_history['loss'], label='Training set (vanilla Keras history)')
    plt.plot(x, model_customHistory['train_loss'], label='Training set (custom history callback)')
    plt.plot(x, model_history['val_loss'], label='Validation set (vanilla Keras history)')
    #plt.plot(x, model_customHistory['valid_loss'], label='Validation set (custom history callback)', linestyle='--')
    plt.legend(loc='upper right')
    plt.ylim([0.00, 0.35])
    plt.ylabel('Loss (categorical crossentropy)')
    plt.xlabel('Number of training epochs')
    saveFile = './outputs/'+dnnTag+'/plots/loss_both.pdf'
    fig.savefig(saveFile)
    plt.close()


def plot_Metrics(dnnTag):

    print('Plotting metrics history.')
    
    model_history = None
    with open('./outputs/'+dnnTag+'/model_history.pkl', 'rb') as f:
        model_history = pickle.load(f)

    plt.clf()
    fig = plt.figure()
    plt.grid()
    x = (range(len(model_history['categorical_accuracy'])+1))[1:]
    plt.plot(x, model_history['categorical_accuracy'], label='Training set')
    plt.plot(x, model_history['val_categorical_accuracy'], label='Validation set')
    plt.legend(loc='lower right')
    plt.ylim([0.00, 1.00])
    plt.ylabel('Categorical accuracy')
    plt.xlabel('Number of training epochs')
    fig.savefig('./outputs/'+dnnTag+'/plots/accuracy.pdf')
    plt.close()


if __name__ == '__main__':

    dnnTag = sys.argv[1] # dnn_yy-mm-dd-hh-mm-ss
    os.makedirs('outputs/'+dnnTag+'/plots', exist_ok=True)

    plot_CustomLoss(dnnTag)
    plot_Loss(dnnTag)
    plot_LossBoth(dnnTag)
    plot_Metrics(dnnTag)
