import matplotlib as mpl
mpl.use('Agg') # don't display plots while plotting ("batch mode")
import matplotlib.pyplot as plt

from config.SampleClasses import dict_Classes
from sklearn.metrics import roc_curve, auc

import sys
import os
import pickle
import numpy as np
import json


def insert_CMS(axisObject):

    axisObject.text(0.00,1.02,"CMS", transform=axisObject.transAxes, fontweight='bold', fontsize=16)
    axisObject.text(0.12,1.02,"Simulation, Work in progress", transform=axisObject.transAxes, style='italic', fontsize=12)


def insert_dnnTag(axisObject, dnnTag):

    axisObject.text(1.00,1.02,"Tag: "+dnnTag, transform=axisObject.transAxes, fontsize=8, horizontalalignment='right')


def insert_InfoBox(axisObject, customHist, kerasHist, plot_type='loss', n_last_epochs=20):

    train_x = np.array(customHist['train_'+str(plot_type)][-n_last_epochs:])
    valid_x = np.array(kerasHist['val_'+str(plot_type)][-n_last_epochs:])
    textstr = '\n'.join((
        r'After epoch '+str(len(customHist['train_'+str(plot_type)]))+':',
        r'Training set: %.3f %%' % (100*customHist['train_'+str(plot_type)][-1], ),
        r'Validation set: %.3f %%' % (100*kerasHist['val_'+str(plot_type)][-1], ),
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
    plt.ylim(bottom=0.00)
    ylabel = None
    if parameters.get('focal_loss'):
        if parameters.get('binary'):
            ylabel = 'Binary focal loss'
        else:
            ylabel = 'Categorical focal loss'
    else:
        if parameters.get('binary'):
            ylabel = 'Loss (binary crossentropy)'
        else:
            ylabel = 'Loss (categorical crossentropy)'
    plt.ylabel(ylabel)
    plt.xlabel('Number of training epochs')
    
    insert_dnnTag(ax, dnnTag)
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

    parameters = get_Parameters(dnnTag)

    plt.clf()
    fig, ax = plt.subplots()
    plt.grid()
    if parameters.get('binary'):
        x = (range(len(model_customHistory['train_binary_accuracy'])+1))[1:]
        plt.plot(x, model_customHistory['train_binary_accuracy'], label='Training set')
        plt.plot(x, model_history['val_binary_accuracy'], label='Validation set', linestyle='--')
        plt.ylabel('Binary accuracy')
    else:
        x = (range(len(model_customHistory['train_categorical_accuracy'])+1))[1:]
        plt.plot(x, model_customHistory['train_categorical_accuracy'], label='Training set')
        plt.plot(x, model_history['val_categorical_accuracy'], label='Validation set', linestyle='--')
        plt.ylabel('Categorical accuracy')
    plt.legend(loc='lower right')
    plt.ylim([0.00, 1.00])
    plt.xlabel('Number of training epochs')

    insert_dnnTag(ax, dnnTag)
    if parameters.get('binary'):
        insert_InfoBox(ax, model_customHistory, model_history, 'binary_accuracy')
    else:
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
            if parameters.get('binary'):
                result[dataset_type][u_cl]['predictions'] = tmp.flatten()
            else:
                result[dataset_type][u_cl]['predictions'] = tmp
            result[dataset_type][u_cl]['weights'] = tmp_weights.flatten()
    
    return result


def plot_PredictionsNormalized(dnnTag, dataset_type):  # dataset_type = train/test/validation

    print('Plotting normalized prediction per node for dataset:', dataset_type)

    parameters = get_Parameters(dnnTag)
    data = load_Predictions(dnnTag)

    if parameters.get('binary'):

        plt.clf()
        fig, ax = plt.subplots()

        for u_cl in reversed(parameters.get('usedClasses')):
            x = data[dataset_type][u_cl]['predictions']
            w = data[dataset_type][u_cl]['weights']
            plt.hist(x, bins=100, weights=w, density=True, histtype='step', range=(0,1), label=u_cl, color=dict_Classes[u_cl]['color'])

        plt.legend(loc='upper center', fontsize=5, ncol=2)
        plt.xlabel('NN output')
        plt.ylabel('Normalized number of events [a. u.]')

        insert_dnnTag(ax, dnnTag)
        insert_CMS(ax)

        saveFile = './outputs/'+dnnTag+'/plots/predictions_'+dataset_type+'_NNoutput.pdf'
        fig.savefig(saveFile)
        plt.close()

    else:

        for i in range(len(parameters.get('usedClasses'))):

            plt.clf()
            fig, ax = plt.subplots()

            for u_cl in reversed(parameters.get('usedClasses')):
                x = data[dataset_type][u_cl]['predictions'][:,i]
                w = data[dataset_type][u_cl]['weights']
                plt.hist(x, bins=100, weights=w, density=True, histtype='step', range=(0,1), label=u_cl, color=dict_Classes[u_cl]['color'])

            plt.legend(loc='upper center', fontsize=5, ncol=2)
            plt.xlabel('NN output node '+str(i))
            plt.ylabel('Normalized number of events [a. u.]')

            insert_dnnTag(ax, dnnTag)
            insert_CMS(ax)

            saveFile = './outputs/'+dnnTag+'/plots/predictions_'+dataset_type+'_node'+str(i)+'.pdf'
            fig.savefig(saveFile)
            plt.close()


def plot_PredictionTrainVSTest(dnnTag):

    #### TO DO: Implement KS test and make this damn plot look good, incl. test dataset with proper error bars.

    print('Plotting normalized prediction for training vs. test dataset.')

    parameters = get_Parameters(dnnTag)
    data = load_Predictions(dnnTag)

    if parameters.get('binary'):

        plt.clf()
        fig, ax = plt.subplots()

        for u_cl in reversed(parameters.get('usedClasses')):
                x_train = data['train'][u_cl]['predictions']
                w_train = data['train'][u_cl]['weights']
                plt.hist(x_train, bins=100, weights=w_train, density=True, histtype='step', range=(0,1), label=u_cl, color=dict_Classes[u_cl]['color'])
                x_test = data['test'][u_cl]['predictions']
                w_test = data['test'][u_cl]['weights']
                plt.hist(x_test, bins=100, weights=w_test, density=True, histtype='step', range=(0,1), label=u_cl+' test', color=dict_Classes[u_cl]['color'], linestyle='--')

        plt.legend(loc='upper center', fontsize=5, ncol=2)
        plt.xlabel('NN output')
        plt.ylabel('Normalized number of events [a. u.]')

        insert_dnnTag(ax, dnnTag)
        insert_CMS(ax)

        saveFile = './outputs/'+dnnTag+'/plots/predictions_KStest_'+dataset_type+'_NNoutput.pdf'
        fig.savefig(saveFile)
        plt.close()

    else:

        for i in range(len(parameters.get('usedClasses'))):

            plt.clf()
            fig, ax = plt.subplots()

            for u_cl in reversed(parameters.get('usedClasses')):
                x_train = data['train'][u_cl]['predictions'][:,i]
                w_train = data['train'][u_cl]['weights']
                plt.hist(x_train, bins=100, weights=w_train, density=True, histtype='step', range=(0,1), label=u_cl, color=dict_Classes[u_cl]['color'])
                x_test = data['test'][u_cl]['predictions'][:,i]
                w_test = data['test'][u_cl]['weights']
                plt.hist(x_test, bins=100, weights=w_test, density=True, histtype='step', range=(0,1), label=u_cl+' test', color=dict_Classes[u_cl]['color'], linestyle='--')

            plt.legend(loc='upper center', fontsize=5, ncol=2)
            plt.xlabel('NN output node '+str(i))
            plt.ylabel('Normalized number of events [a. u.]')

            insert_dnnTag(ax, dnnTag)
            insert_CMS(ax)

            saveFile = './outputs/'+dnnTag+'/plots/predictions_KStest_node'+str(i)+'.pdf'
            fig.savefig(saveFile)
            plt.close()


def plot_PredictionsStacked(dnnTag, dataset_type):

    print('Plotting stacked prediction per node for dataset:', dataset_type)

    parameters = get_Parameters(dnnTag)
    data = load_Predictions(dnnTag)

    if parameters.get('binary'):

        plt.clf()
        fig, ax = plt.subplots()

        x = list()
        w = list()
        cl = list()
        colors = list()
        for u_cl in reversed(parameters.get('usedClasses')):
            tmp_x = data[dataset_type][u_cl]['predictions']
            x.append(tmp_x)
            norm_factor = None
            if u_cl == 'TTbar':
                norm_factor = 0.784 ### highly dependent on phase space / selection !!!
                cl.append(u_cl+' x'+str(norm_factor))
            else:
                norm_factor = 1.0
                cl.append(u_cl)
            w.append(data[dataset_type][u_cl]['weights']*norm_factor)
            colors.append(dict_Classes[u_cl]['color'])

        plt.hist(x, bins=20, weights=w, stacked=True, range=(0,1), label=cl, color=colors)

        plt.legend(loc='upper center', fontsize=5, ncol=2)
        plt.xlabel('NN output')
        plt.ylabel('Number of events')

        insert_dnnTag(ax, dnnTag)
        insert_CMS(ax)

        saveFile = './outputs/'+dnnTag+'/plots/predictions_'+dataset_type+'_NNoutput_stacked.pdf'
        fig.savefig(saveFile)
        plt.close()

    else:

        for i in range(len(parameters.get('usedClasses'))):

            plt.clf()
            fig, ax = plt.subplots()

            x = list()
            w = list()
            cl = list()
            colors = list()
            for u_cl in reversed(parameters.get('usedClasses')):
                tmp_x = data[dataset_type][u_cl]['predictions'][:,i]
                x.append(tmp_x)
                norm_factor = None
                if u_cl == 'TTbar':
                    norm_factor = 0.784 ### highly dependent on phase space / selection !!!
                    cl.append(u_cl+' x'+str(norm_factor))
                else:
                    norm_factor = 1.0
                    cl.append(u_cl)
                w.append(data[dataset_type][u_cl]['weights']*norm_factor)
                colors.append(dict_Classes[u_cl]['color'])

            plt.hist(x, bins=50, weights=w, stacked=True, range=(0,1), label=cl, color=colors)

            plt.legend(loc='upper center', fontsize=5, ncol=2)
            plt.xlabel('NN output node '+str(i))
            plt.ylabel('Number of events')

            insert_dnnTag(ax, dnnTag)
            insert_CMS(ax)

            saveFile = './outputs/'+dnnTag+'/plots/predictions_'+dataset_type+'_node'+str(i)+'_stacked.pdf'
            fig.savefig(saveFile)
            plt.close()


def plot_ROC(dnnTag, dataset_type):

    print("Calculating and plotting ROC curves...")

    parameters = get_Parameters(dnnTag)
    data = load_Predictions(dnnTag)
    node_names = None
    if not parameters.get('binary'): # file does not exist in binary case
        node_names = np.load('./outputs/'+dnnTag+'/encoder_classes.npy')

    plt.clf()
    fig, ax = plt.subplots()

    plt.plot([0,1], [0,1], color='black', linestyle='--')

    loop_list = None
    if parameters.get('binary'):
        loop_list = [parameters.get('binary_signal')]
    else:
        loop_list = parameters.get('usedClasses')

    for process in loop_list: # loop over output class nodes; 'process' is the signal for which a ROC curve will be plotted

        node_id = None
        if not parameters.get('binary'):
            for node in range(len(node_names)):
                if node_names[node] == process:
                    node_id = node
                    print("Node ID for class "+process+":", node_id)
                    break

        y_score_list = list()
        y_true_list = list()
        weights_list = list()

        for u_cl in parameters.get('usedClasses'):
            if parameters.get('binary'):
                y_score_list.append(data[dataset_type][u_cl]['predictions'])
            else:
                y_score_list.append(data[dataset_type][u_cl]['predictions'][:,node_id])
            weights_list.append(data[dataset_type][u_cl]['weights'])
            if u_cl == process:
                y_true_list.append(np.ones(len(data[dataset_type][u_cl]['weights']), dtype=int))
            else:
                y_true_list.append(np.zeros(len(data[dataset_type][u_cl]['weights']), dtype=int))

        y_score = np.concatenate(y_score_list)
        y_true = np.concatenate(y_true_list)
        weights = np.concatenate(weights_list)

        fpr, tpr, thresholds = roc_curve(y_true, y_score, sample_weight=weights)
        roc_area = auc(fpr, tpr)

        # plot the ROC for process X in output node X!
        label = process+" (AUC = %.3f)" % roc_area
        plt.plot(fpr, tpr, color=dict_Classes[process]['color'], label=label)



    plt.grid()
    plt.legend(loc='lower right', fontsize=8)
    plt.xlabel("Background efficieny")
    plt.ylabel("Signal efficiency")
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    insert_dnnTag(ax, dnnTag)
    insert_CMS(ax)

    saveFile = './outputs/'+dnnTag+'/plots/roc.pdf'
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
    plot_ROC(dnnTag, 'train') # maybe use train+validation set?
