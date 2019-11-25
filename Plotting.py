import matplotlib as mpl
mpl.use('Agg') # don't display plots while plotting ("batch mode")
import matplotlib.pyplot as plt

import sys
import os
import pickle
import numpy as np


def insert_CMS(axisObject):

    axisObject.text(0.05,0.88,"CMS", transform=axisObject.transAxes, fontweight='bold', fontsize=16)
    axisObject.text(0.17,0.88,"Simulation, Work in progress", transform=axisObject.transAxes, style='italic', fontsize=12)


def insert_InfoBox(axisObject, customHist, kerasHist, type='loss', n_last_epochs=20):

    train_x = np.array(customHist['train_'+str(type)][-n_last_epochs:])
    valid_x = np.array(kerasHist['val_'+str(type)][-n_last_epochs:])
    textstr = '\n'.join((
        r'After epoch '+str(len(customHist['train_'+str(type)]))+':',
        r'Training set: %.3f %%' % (100*customHist['train_'+str(type)][-1], ),
        r'Validation set: %.3f %%' % (100*kerasHist['val_'+str(type)][-1], ),
        r'Last '+str(n_last_epochs)+' epochs:',
        r'Training set: $%.3f \pm %.3f$ %%' % (100*train_x.mean(), 100*train_x.std(), ),
        r'Validation set: $%.3f \pm %.3f$ %%' % (100*valid_x.mean(), 100*valid_x.std(), )))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    axisObject.text(0.5, 0.5, textstr, transform=axisObject.transAxes, bbox=props)


def plot_Loss(dnnTag):

    print('Plotting loss history.')

    pickleFile = './outputs/'+dnnTag+'/model_history.pkl'
    with open(pickleFile, 'rb') as f:
        model_history = pickle.load(f)

    pickleFileCustom = './outputs/'+dnnTag+'/model_customHistory.pkl'
    with open(pickleFileCustom, 'rb') as f:
        model_customHistory = pickle.load(f)

    plt.clf()
    fig, ax = plt.subplots()
    plt.grid()
    x = (range(len(model_customHistory['train_loss'])+1))[1:]
    plt.plot(x, model_customHistory['train_loss'], label='Training set')
    plt.plot(x, model_history['val_loss'], label='Validation set', linestyle='--')
    plt.legend(loc='upper right')
    plt.ylim([0.00, 0.25])
    plt.ylabel('Loss (categorical crossentropy)')
    plt.xlabel('Number of training epochs')
    plt.title('Tag: '+dnnTag)

    insert_InfoBox(ax, model_customHistory, model_history, 'loss')
    insert_CMS(ax)

    saveFile = './outputs/'+dnnTag+'/plots/loss.pdf'
    fig.savefig(saveFile)
    plt.close()


def plot_Metrics(dnnTag):

    print('Plotting metrics history.')

    pickleFile = './outputs/'+dnnTag+'/model_history.pkl'
    with open(pickleFile, 'rb') as f:
        model_history = pickle.load(f)

    pickleFileCustom = './outputs/'+dnnTag+'/model_customHistory.pkl'
    with open(pickleFileCustom, 'rb') as f:
        model_customHistory = pickle.load(f)
    
    plt.clf()
    fig, ax = plt.subplots()
    plt.grid()
    x = (range(len(model_customHistory['train_categorical_accuracy'])+1))[1:]
    plt.plot(x, model_customHistory['train_categorical_accuracy'], label='Training set')
    plt.plot(x, model_history['val_categorical_accuracy'], label='Validation set', linestyle='--')
    plt.legend(loc='lower right')
    plt.ylim([0.00, 1.00])
    plt.ylabel('Categorical accuracy')
    plt.xlabel('Number of training epochs')
    plt.title('Tag: '+dnnTag)
    
    insert_InfoBox(ax, model_customHistory, model_history, 'categorical_accuracy')
    insert_CMS(ax)

    saveFile = './outputs/'+dnnTag+'/plots/accuracy.pdf'
    fig.savefig(saveFile)
    plt.close()


if __name__ == '__main__':

    dnnTag = sys.argv[1] # dnn_yy-mm-dd-hh-mm-ss
    os.makedirs('outputs/'+dnnTag+'/plots', exist_ok=True)

    plot_Loss(dnnTag)
    plot_Metrics(dnnTag)
