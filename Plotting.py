import matplotlib as mpl
mpl.use('Agg') # don't display plots while plotting ("batch mode")
import matplotlib.pyplot as plt

from config.SampleClasses import dict_Classes

import sys
import os
import pickle
import numpy as np
import json


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

    parameters = get_Parameters(dnnTag)

    plt.clf()
    fig, ax = plt.subplots()
    plt.grid()
    x = (range(len(model_customHistory['train_loss'])+1))[1:]
    plt.plot(x, model_customHistory['train_loss'], label='Training set')
    plt.plot(x, model_history['val_loss'], label='Validation set', linestyle='--')
    plt.legend(loc='upper right')
    #plt.ylim([0.00, 0.25])
    ylabel = None
    if parameters['focal_loss']:
        ylabel = 'Categorical focal loss'
    else:
        ylabel = 'Loss (categorical crossentropy)'
    plt.ylabel(ylabel)
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


def get_Parameters(dnnTag):

    with open('./outputs/'+dnnTag+'/parameters.json') as f:
        parameters = json.loads(f.read())

    return parameters


def load_Predictions(dnnTag):  # type=train/test/validation

    predDir = './outputs/'+dnnTag+'/predictions/'
    parameters = get_Parameters(dnnTag)
    
    result = dict()
    
    for dataset_type in ['train', 'test', 'validation']:
        result[dataset_type] = dict()
        for u_cl in parameters['usedClasses']:
            result[dataset_type][u_cl] = dict()
            tmp = np.load(predDir+'pred_'+dataset_type+'__'+u_cl+'.npy')
            tmp_weights = np.load('./outputs/'+dnnTag+'/workdir/'+u_cl+'_weights_'+dataset_type+'.npy')
            result[dataset_type][u_cl]['predictions'] = tmp
            result[dataset_type][u_cl]['weights'] = tmp_weights.flatten()
    
    return result


def plot_PredictionsNormalized(dnnTag, dataset_type):  # dataset_type = train/test/validation

    print('Plotting normalized prediction per node for dataset:', dataset_type)

    parameters = get_Parameters(dnnTag)
    data = load_Predictions(dnnTag)

    for i in range(len(parameters['usedClasses'])):

        plt.clf()
        fig, ax = plt.subplots()

        for u_cl in reversed(parameters['usedClasses']):
            x = data[dataset_type][u_cl]['predictions'][:,i]
            w = data[dataset_type][u_cl]['weights']
            plt.hist(x, bins=100, weights=w, density=True, histtype='step', range=(0,1), label=u_cl, color=dict_Classes[u_cl]['color'])

        plt.legend(loc='upper center')
        plt.xlabel('NN output node '+str(i))
        plt.ylabel('Normalized number of events [a. u.]')

        insert_CMS(ax)
    
        saveFile = './outputs/'+dnnTag+'/plots/predictions_'+dataset_type+'_node'+str(i)+'.pdf'
        fig.savefig(saveFile)
        plt.close()


def plot_PredictionTrainVSTest(dnnTag):

    #### TO DO: Implement KS test and make this damn plot look good, incl. test dataset with proper error bars.

    print('Plotting normalized prediction for training vs. test dataset.')

    parameters = get_Parameters(dnnTag)
    data = load_Predictions(dnnTag)

    for i in range(len(parameters['usedClasses'])):

        plt.clf()
        fig, ax = plt.subplots()

        for u_cl in reversed(parameters['usedClasses']):
            x_train = data['train'][u_cl]['predictions'][:,i]
            w_train = data['train'][u_cl]['weights']
            plt.hist(x_train, bins=100, weights=w_train, density=True, histtype='step', range=(0,1), label=u_cl, color=dict_Classes[u_cl]['color'])
            x_test = data['test'][u_cl]['predictions'][:,i]
            w_test = data['test'][u_cl]['weights']
            plt.hist(x_test, bins=100, weights=w_test, density=True, histtype='step', range=(0,1), label=u_cl+' test', color=dict_Classes[u_cl]['color'], linestyle='--')
        
        plt.legend(loc='upper center')
        plt.xlabel('NN output node '+str(i))
        plt.ylabel('Normalized number of events [a. u.]')
        
        insert_CMS(ax)
        
        saveFile = './outputs/'+dnnTag+'/plots/predictions_KStest_node'+str(i)+'.pdf'
        fig.savefig(saveFile)
        plt.close()


def plot_PredictionsStacked(dnnTag, dataset_type):

    print('Plotting stacked prediction per node for dataset:', dataset_type)

    parameters = get_Parameters(dnnTag)
    data = load_Predictions(dnnTag)
    
    for i in range(len(parameters['usedClasses'])):
        
        plt.clf()
        fig, ax = plt.subplots()

        x = list()
        w = list()
        cl = list()
        colors = list()
        for u_cl in reversed(parameters['usedClasses']):
            x.append(data[dataset_type][u_cl]['predictions'][:,i])
            w.append(data[dataset_type][u_cl]['weights'])
            cl.append(u_cl)
            colors.append(dict_Classes[u_cl]['color'])

        plt.hist(x, bins=100, weights=w, stacked=True, range=(0,1), label=cl, color=colors)

        plt.legend(loc='upper center')
        plt.xlabel('NN output node '+str(i))
        plt.ylabel('Number of events')

        insert_CMS(ax)
    
        saveFile = './outputs/'+dnnTag+'/plots/predictions_'+dataset_type+'_node'+str(i)+'_stacked.pdf'
        fig.savefig(saveFile)
        plt.close()


if __name__ == '__main__':

    dnnTag = sys.argv[1] # dnn_yy-mm-dd-hh-mm-ss
    os.makedirs('outputs/'+dnnTag+'/plots', exist_ok=True)

    plot_Loss(dnnTag)
    plot_Metrics(dnnTag)
    for dataset_type in ['train', 'test', 'validation']:
        plot_PredictionsNormalized(dnnTag, dataset_type)
        plot_PredictionsStacked(dnnTag, dataset_type)
    plot_PredictionTrainVSTest(dnnTag)
