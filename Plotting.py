import matplotlib as mpl
mpl.use('Agg') # don't display plots while plotting ("batch mode")
import matplotlib.pyplot as plt

from config.SampleClasses import dict_Classes
from sklearn.metrics import roc_curve, auc
from scipy import stats
from ROOT import TH1F
from decimal import Decimal

import sys
import os
import pickle
import numpy as np
import json
import math


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
    axisObject.text(0.5, 0.3, textstr, transform=axisObject.transAxes, bbox=props)


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


def get_ClassEncoding(dnnTag):

    return np.load('./outputs/'+dnnTag+'/encoder_classes.npy')


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

        for u_cl in reversed(parameters.get('usedClasses')): # reversing ensures the correct order of stacking the processes in the plots
            x = None
            w = None
            if dataset_type == 'all':
                x = np.concatenate((data['train'][u_cl]['predictions'], data['test'][u_cl]['predictions'], data['validation'][u_cl]['predictions']), axis=0)
                #x = data['train'][u_cl]['predictions']
                #print(x)
                #x += data['test'][u_cl]['predictions']
                #x += data['validation'][u_cl]['predictions']
                w = np.concatenate((data['train'][u_cl]['weights'], data['test'][u_cl]['weights'], data['validation'][u_cl]['weights']), axis=0)
                #w = data['train'][u_cl]['weights']
                #w += data['test'][u_cl]['weights']
                #w += data['validation'][u_cl]['weights']
            else:
                x = data[dataset_type][u_cl]['predictions']
                w = data[dataset_type][u_cl]['weights']
            plt.hist(x, bins=100, weights=w, density=True, histtype='step', range=(0,1), label=u_cl, color=dict_Classes[u_cl]['color'])

        plt.legend(loc='upper center', fontsize=5, ncol=2)
        plt.xlabel('NN output')
        plt.ylabel('Normalized number of events [a. u.]')

        insert_dnnTag(ax, dnnTag)
        insert_CMS(ax)

        saveFile = './outputs/'+dnnTag+'/plots/predictions_'+dataset_type+'_NNoutput__norm.pdf'
        fig.savefig(saveFile)
        plt.close()

    else:

        encoding = get_ClassEncoding(dnnTag)

        for i in range(len(parameters.get('usedClasses'))):

            plt.clf()
            fig, ax = plt.subplots()

            for u_cl in reversed(parameters.get('usedClasses')):
                x = None
                w = None
                # x = data[dataset_type][u_cl]['predictions'][:,i]
                # w = data[dataset_type][u_cl]['weights']
                if dataset_type == 'all':
                    x = np.concatenate((data['train'][u_cl]['predictions'][:,i], data['test'][u_cl]['predictions'][:,i], data['validation'][u_cl]['predictions'][:,i]), axis=0)
                    w = np.concatenate((data['train'][u_cl]['weights'], data['test'][u_cl]['weights'], data['validation'][u_cl]['weights']), axis=0)
                else:
                    x = data[dataset_type][u_cl]['predictions'][:,i]
                    w = data[dataset_type][u_cl]['weights']
                plt.hist(x, bins=100, weights=w, density=True, histtype='step', range=(0,1), label=u_cl, color=dict_Classes[u_cl]['color'])

            plt.legend(loc='upper center', fontsize=5, ncol=2)
            plt.xlabel('NN output node '+str(i)+' ('+encoding[i]+')')
            plt.ylabel('Normalized number of events [a. u.]')

            insert_dnnTag(ax, dnnTag)
            insert_CMS(ax)

            saveFile = './outputs/'+dnnTag+'/plots/predictions_'+dataset_type+'_node'+str(i)+'_'+encoding[i]+'__norm.pdf'
            fig.savefig(saveFile)
            plt.close()


def plot_PredictionTrainVSTest(dnnTag):

    #### TO DO: Implement KS test and make this damn plot look good, incl. test dataset with proper error bars.

    print('Plotting normalized prediction for training vs. test dataset.')

    parameters = get_Parameters(dnnTag)
    data = load_Predictions(dnnTag)

    nbins = 50

    if parameters.get('binary'): # make another plot in which you stack all backgrounds such that it is indeed only signal vs. background KS test

        plt.clf()
        fig, ax = plt.subplots()

        for u_cl in reversed(parameters.get('usedClasses')):

            # if u_cl == 'QCD': continue

            x_train = data['train'][u_cl]['predictions']
            w_train = data['train'][u_cl]['weights']
            sum_w_train = np.sum(w_train)

            x_test = data['test'][u_cl]['predictions']
            w_test = data['test'][u_cl]['weights']
            sum_w_test = np.sum(w_test)

            plt.hist(x_train, bins=nbins, weights=w_train, density=True, histtype='step', range=(0,1), label=dict_Classes[u_cl]['latex']+' (train)', color=dict_Classes[u_cl]['color'])
            #plt.hist(x_test, bins=nbins, weights=w_test, density=True, histtype='step', range=(0,1), label=u_cl+' test', color=dict_Classes[u_cl]['color'], linestyle='--')

            w_test_density = w_test/sum_w_test*nbins
            test_binContents = list()
            test_binErrors = list()
            binCenters = list()

            # Using ROOT to properly calculate bin errors. Numpy does not have an easy solution to this
            t = TH1F('th1f_test_'+u_cl, 'no title necessary', nbins, 0, 1)
            for i in range(len(x_test)):
                t.Fill(x_test[i], w_test_density[i])
            for i in range(nbins):
                test_binContents.append(t.GetBinContent(i+1))
                test_binErrors.append(t.GetBinError(i+1))
                binCenters.append(i/nbins+1/nbins*0.5)

            h_test,  bin_edges_test  = np.histogram(x_test,  range=(0,1), bins=nbins, weights=w_test,  density=True)
            h_train, bin_edges_train = np.histogram(x_train, range=(0,1), bins=nbins, weights=w_train, density=True)
            # h_test and test_binContents should be the same!!!
            #print(h_test)
            #print(test_binContents)
            #print(test_binErrors)

            KSstat, KSpvalue = stats.ks_2samp(x_train, x_test)
            KS_alpha = 2*math.exp(-2*len(x_train)*len(x_test)/(len(x_train)+len(x_test))*KSstat*KSstat)
            #KSstat, KSpvalue = stats.ks_2samp(h_train, h_test)
            #KS_alpha = 2*math.exp(-2*len(h_train)*len(h_test)/(len(h_train)+len(h_test))*KSstat*KSstat)

            test_label = dict_Classes[u_cl]['latex']+' (test, KS = %.3f)' % Decimal(KSstat)
            plt.errorbar(binCenters, test_binContents, yerr=test_binErrors, label=test_label, color=dict_Classes[u_cl]['color'], linestyle='', marker='.')

        plt.legend(loc='upper center', fontsize=5, ncol=2)
        plt.xlabel('NN output')
        plt.ylabel('Normalized number of events [a. u.]')
        plt.ylim(bottom=0)

        insert_dnnTag(ax, dnnTag)
        insert_CMS(ax)

        saveFile = './outputs/'+dnnTag+'/plots/KStest.pdf'
        fig.savefig(saveFile)
        plt.close()

    else:

        encoding = get_ClassEncoding(dnnTag)

        for i in range(len(parameters.get('usedClasses'))):

            plt.clf()
            fig, ax = plt.subplots()

            for u_cl in reversed(parameters.get('usedClasses')):

                # if u_cl == 'QCD': continue

                x_train = data['train'][u_cl]['predictions'][:,i]
                w_train = data['train'][u_cl]['weights']
                sum_w_train = np.sum(w_train)

                x_test = data['test'][u_cl]['predictions'][:,i]
                w_test = data['test'][u_cl]['weights']
                sum_w_test = np.sum(w_test)

                plt.hist(x_train, bins=nbins, weights=w_train, density=True, histtype='step', range=(0,1), label=dict_Classes[u_cl]['latex']+' (train)', color=dict_Classes[u_cl]['color'])
                #plt.hist(x_test, bins=nbins, weights=w_test, density=True, histtype='step', range=(0,1), label=u_cl+' test', color=dict_Classes[u_cl]['color'], linestyle='--')

                w_test_density = w_test/sum_w_test*nbins
                test_binContents = list()
                test_binErrors = list()
                binCenters = list()

                # Using ROOT to properly calculate bin errors. Numpy does not have an easy solution to this
                t = TH1F('th1f_test_'+u_cl, 'no title necessary', nbins, 0, 1)
                for j in range(len(x_test)):
                    t.Fill(x_test[j], w_test_density[j])
                for j in range(nbins):
                    test_binContents.append(t.GetBinContent(j+1))
                    test_binErrors.append(t.GetBinError(j+1))
                    binCenters.append(j/nbins+1/nbins*0.5)

                h_test,  bin_edges_test  = np.histogram(x_test,  range=(0,1), bins=nbins, weights=w_test,  density=True)
                h_train, bin_edges_train = np.histogram(x_train, range=(0,1), bins=nbins, weights=w_train, density=True)

                KSstat, KSpvalue = stats.ks_2samp(x_train, x_test)
                KS_alpha = 2*math.exp(-2*len(x_train)*len(x_test)/(len(x_train)+len(x_test))*KSstat*KSstat)
                #KSstat, KSpvalue = stats.ks_2samp(h_train, h_test)
                #KS_alpha = 2*math.exp(-2*len(h_train)*len(h_test)/(len(h_train)+len(h_test))*KSstat*KSstat)

                test_label = dict_Classes[u_cl]['latex']+' (test, KS = %.3f)' % Decimal(KSstat)
                plt.errorbar(binCenters, test_binContents, yerr=test_binErrors, label=test_label, color=dict_Classes[u_cl]['color'], linestyle='', marker='.')

            plt.legend(loc='upper center', fontsize=5, ncol=2)
            plt.xlabel('NN output node '+str(i)+' ('+encoding[i]+')')
            plt.ylabel('Normalized number of events [a. u.]')
            plt.ylim(bottom=0)

            insert_dnnTag(ax, dnnTag)
            insert_CMS(ax)

            saveFile = './outputs/'+dnnTag+'/plots/KStest_node'+str(i)+'__'+encoding[i]+'.pdf'
            fig.savefig(saveFile)
            plt.close()

            # for u_cl in reversed(parameters.get('usedClasses')):
            #     x_train = data['train'][u_cl]['predictions'][:,i]
            #     w_train = data['train'][u_cl]['weights']
            #     plt.hist(x_train, bins=nbins, weights=w_train, density=True, histtype='step', range=(0,1), label=dict_Classes[u_cl]['latex'], color=dict_Classes[u_cl]['color'])
            #     x_test = data['test'][u_cl]['predictions'][:,i]
            #     w_test = data['test'][u_cl]['weights']
            #     plt.hist(x_test, bins=nbins, weights=w_test, density=True, histtype='step', range=(0,1), label=(dict_Classes[u_cl]['latex']+' test'), color=dict_Classes[u_cl]['color'], linestyle='--')
            #
            # plt.legend(loc='upper center', fontsize=5, ncol=2)
            # plt.xlabel('NN output node '+str(i)+' ('+u_cl+')')
            # plt.ylabel('Normalized number of events [a. u.]')
            #
            # insert_dnnTag(ax, dnnTag)
            # insert_CMS(ax)
            #
            # saveFile = './outputs/'+dnnTag+'/plots/predictions_KStest_node'+str(i)+'__'+u_cl+'.pdf'
            # fig.savefig(saveFile)
            # plt.close()


def plot_Significance(dnnTag):

    print('Plotting signal significance vs. cut on DNN output')

    parameters = get_Parameters(dnnTag)
    data = load_Predictions(dnnTag)

    if parameters.get('binary'):

        signal_name = parameters.get('binary_signal')

        hist_sig_train, bins = np.histogram(data['train'][signal_name]['predictions'], bins=50, range=(0,1), weights=data['train'][signal_name]['weights'])
        hist_sig_test, bins = np.histogram(data['test'][signal_name]['predictions'], bins=50, range=(0,1), weights=data['test'][signal_name]['weights'])
        hist_sig_validation, bins = np.histogram(data['validation'][signal_name]['predictions'], bins=50, range=(0,1), weights=data['validation'][signal_name]['weights'])

        hist_sig = hist_sig_train + hist_sig_test + hist_sig_validation
        #print(hist_sig)

        hist_bkg = np.zeros(50)
        for u_cl in parameters.get('usedClasses'):
            if u_cl == signal_name:
                continue
            tmp_hist_bkg_train, bins = np.histogram(data['train'][u_cl]['predictions'], bins=50, range=(0,1), weights=data['train'][u_cl]['weights'])
            tmp_hist_bkg_test, bins = np.histogram(data['test'][u_cl]['predictions'], bins=50, range=(0,1), weights=data['test'][u_cl]['weights'])
            tmp_hist_bkg_validation, bins = np.histogram(data['validation'][u_cl]['predictions'], bins=50, range=(0,1), weights=data['validation'][u_cl]['weights'])

            tmp_hist_bkg = tmp_hist_bkg_train + tmp_hist_bkg_test + tmp_hist_bkg_validation
            hist_bkg += tmp_hist_bkg
        #print(hist_bkg)

        # Now calculate cumulative histograms:
        cumu_sig = np.zeros(50)
        cumu_bkg = np.zeros(50)
        for i in range(50):
            cumu_sig[i] = hist_sig[i:50].sum()
            cumu_bkg[i] = hist_bkg[i:50].sum()
        #print(cumu_sig)
        #print(cumu_bkg)

        # Now calculate s / sqrt(s+b):
        significance = np.zeros(51) # add one extra bin for cut at DNN output = 1.0, resulting in s/sqrt(s+b) = 0
        for i in range(50):
            significance[i] = cumu_sig[i] / math.sqrt(cumu_sig[i] + cumu_bkg[i])
        #print(significance)

        # plotting
        plt.clf()
        fig, ax = plt.subplots()

        plt.plot(bins, significance)
        plt.xlabel('Cut on NN output')
        plt.ylabel('Signal significance')
        plt.grid()

        insert_dnnTag(ax, dnnTag)
        insert_CMS(ax)

        saveFile = './outputs/'+dnnTag+'/plots/significance.pdf'
        fig.savefig(saveFile)
        plt.close()

        # save max significance to file
        filename = './outputs/'+dnnTag+'/max_significance.txt'
        with open(filename, 'w') as f:
            f.write(str(significance.max()))

    else:

        print('  Not yet implemented for multi-class DNNs.')


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
        labels = list()
        for u_cl in reversed(parameters.get('usedClasses')):
            tmp_x = None
            if dataset_type == 'all':
                tmp_x = np.concatenate((data['train'][u_cl]['predictions'], data['test'][u_cl]['predictions'], data['validation'][u_cl]['predictions']), axis=0)
                #tmp_x = data['train'][u_cl]['predictions']
                #print(tmp_x)
                #tmp_x += data['test'][u_cl]['predictions']
                #tmp_x += data['validation'][u_cl]['predictions']
            else:
                tmp_x = data[dataset_type][u_cl]['predictions']
            x.append(tmp_x)
            norm_factor = None
            if u_cl == 'TTbar':
#                norm_factor = 0.784 ### highly dependent on phase space / selection !!!
                norm_factor = 1.
                cl.append(u_cl+' x'+str(norm_factor))
            else:
                norm_factor = 1.0
                cl.append(u_cl)
            tmp_w = None
            if dataset_type == 'all':
                tmp_w = np.concatenate((data['train'][u_cl]['weights'], data['test'][u_cl]['weights'], data['validation'][u_cl]['weights']), axis=0)
                #tmp_w = data['train'][u_cl]['weights']
                #tmp_w += data['test'][u_cl]['weights']
                #tmp_w += data['validation'][u_cl]['weights']
            else:
                tmp_w = data[dataset_type][u_cl]['weights']
            #w.append(data[dataset_type][u_cl]['weights']*norm_factor)
            w.append(tmp_w*norm_factor)
            colors.append(dict_Classes[u_cl]['color'])
            labels.append(dict_Classes[u_cl]['latex'])

        plt.hist(x, bins=50, weights=w, stacked=True, range=(0,1), label=labels, color=colors)

        plt.legend(loc='upper center', fontsize=5, ncol=2)
        plt.xlabel('NN output')
        plt.ylabel('Number of events')

        insert_dnnTag(ax, dnnTag)
        insert_CMS(ax)

        saveFile = './outputs/'+dnnTag+'/plots/predictions_'+dataset_type+'_NNoutput__stacked.pdf'
        fig.savefig(saveFile)
        plt.close()

    else:

        encoding = get_ClassEncoding(dnnTag)

        for i in range(len(parameters.get('usedClasses'))):

            plt.clf()
            fig, ax = plt.subplots()

            x = list()
            w = list()
            cl = list()
            colors = list()
            labels = list()
            for u_cl in reversed(parameters.get('usedClasses')):
                tmp_x = None
                if dataset_type == 'all':
                    tmp_x = np.concatenate((data['train'][u_cl]['predictions'][:,i], data['test'][u_cl]['predictions'][:,i], data['validation'][u_cl]['predictions'][:,i]), axis=0)
                else:
                    tmp_x = data[dataset_type][u_cl]['predictions'][:,i]
                x.append(tmp_x)
                norm_factor = None
                if u_cl == 'TTbar':
                    norm_factor = 1. ### highly dependent on phase space / selection !!!
                    cl.append(u_cl+' x'+str(norm_factor))
                else:
                    norm_factor = 1.0
                    cl.append(u_cl)
                tmp_w = None
                if dataset_type == 'all':
                    tmp_w = np.concatenate((data['train'][u_cl]['weights'], data['test'][u_cl]['weights'], data['validation'][u_cl]['weights']), axis=0)
                else:
                    tmp_w = data[dataset_type][u_cl]['weights']
                w.append(tmp_w*norm_factor)
                colors.append(dict_Classes[u_cl]['color'])
                labels.append(dict_Classes[u_cl]['latex'])

            plt.hist(x, bins=50, weights=w, stacked=True, range=(0,1), label=labels, color=colors)

            plt.legend(loc='upper center', fontsize=5, ncol=2)
            plt.xlabel('NN output node '+str(i)+' ('+encoding[i]+')')
            plt.ylabel('Number of events')

            insert_dnnTag(ax, dnnTag)
            insert_CMS(ax)

            saveFile = './outputs/'+dnnTag+'/plots/predictions_'+dataset_type+'_node'+str(i)+'_'+encoding[i]+'__stacked.pdf'
            fig.savefig(saveFile)
            plt.close()


def plot_TTbarConstPredictionsStacked(dnnTag, dataset_types=['train, test, validation'], nbins=20):

    print('Plotting TTbar-constant stacked prediction')

    parameters = get_Parameters(dnnTag)
    data = load_Predictions(dnnTag)

    if parameters.get('binary'):

        x_TTbar = np.array([])
        w_TTbar = np.array([])

        for dt in dataset_types:
            np.append(x_TTbar, data[dataset_type]['TTbar']['predictions'])
            np.append(w_TTbar, data[dataset_type]['TTbar']['weights'])

        int_TTbar = np.sum(w_TTbar)
        TTbar_bin_content = int_TTbar/nbins

        binEdges_predValues = list()

        x_TTbar_indices = x_TTbar.argsort()
        x_TTbar = x_TTbar[x_TTbar_indices[::1]]
        w_TTbar = w_TTbar[x_TTbar_indices[::1]]

        #### go on from here

        plt.clf()
        fig, ax = plt.subplots()

        x = list()
        w = list()
        cl = list()
        colors = list()
        labels = list()
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
            labels.append(dict_Classes[u_cl]['latex'])

        plt.hist(x, bins=50, weights=w, stacked=True, range=(0,1), label=labels, color=colors)

        plt.legend(loc='upper center', fontsize=5, ncol=2)
        plt.xlabel('NN output')
        plt.ylabel('Number of events')

        insert_dnnTag(ax, dnnTag)
        insert_CMS(ax)

        saveFile = './outputs/'+dnnTag+'/plots/predictions_'+dataset_type+'_NNoutput_stacked.pdf'
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
        print('  Events with negative weights:', len(weights[weights <= 0]), '/', len(weights), '('+str(len(weights[weights <= 0])/len(weights)*100)+' %)')
        print('  Sum of negative weights / positive weights:', (weights[weights <= 0]).sum(), '/', (weights[weights > 0]).sum(), ' | ratio:', (weights[weights <= 0]).sum()/(weights[weights > 0]).sum())
        # weights = np.ones(len(weights))
        print('  CAVEAT: Will only use positively weighted events to calculate ROC!')
        y_score = y_score[weights > 0]
        y_true = y_true[weights > 0]
        weights = weights[weights > 0]

        fpr, tpr, thresholds = roc_curve(y_true, y_score, sample_weight=weights)
        roc_area = auc(fpr, tpr)

        # plot the ROC for process X in output node X!
        label = dict_Classes[process]['latex']+" (AUC = %.3f)" % roc_area
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

    # plot_Loss(dnnTag)
    # plot_Metrics(dnnTag)
    # for dataset_type in ['train', 'test', 'validation', 'all']:
    #     plot_PredictionsNormalized(dnnTag, dataset_type)
    #     plot_PredictionsStacked(dnnTag, dataset_type)
    # plot_Significance(dnnTag) # only works for binary DNNs so far. For multiclass, nothing will happen
    # plot_PredictionTrainVSTest(dnnTag)
#    plot_TTbarConstPredictionsStacked(dnnTag)
    plot_ROC(dnnTag, 'train') # maybe use train+validation set?
