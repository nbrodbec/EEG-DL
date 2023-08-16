import os
import requests
import pyedflib
import numpy as np

def get_physionet_dataset():
    '''
    This function pulls the edf files from physionet dataset.
    All the files will be saved under "physionet_dataset" in the directory where the function is run.
    '''
    basePath = 'physionet_dataset/'
    baseUrl = 'https://archive.physionet.org/pn4/eegmmidb/'

    if not os.path.exists('physionet_dataset/'):
        os.mkdir('physionet_dataset')

    for i in range (1, 110):
        subject = 'S' + ('0' * (3 - len(str(i)))) + str(i)
        if not os.path.exists(basePath + subject):
            os.mkdir(basePath + subject)
        for j in range (1, 15):
            trial = 'R' + ('0' * (2 - len(str(j)))) + str(j) + '.edf'
            if not os.path.exists(basePath + subject + '/' + trial):
                fileUrl = baseUrl + subject + '/' + subject + trial
                file = requests.get(fileUrl)
                open(basePath + subject + '/' + trial, 'wb').write(file.content)

def extract_data_edf(subject, files=range(1, 15)):
    '''
    This function extracts all the data from one patient, and appends the appropriate labels for each file.
    Note that one of the papers excluded the trials from subject 89 due to differences in the recording results,
    so it is probably best to not use that.
    
    input: A string specifying the patient to use. Please conform with the naming convention of the folders
    ('S001', 'S012', 'S102').
    output: A numpy array of shape (x, 65). x depends on the number of samples in each of the 14 files.
    '''
    basePath = 'Download_Raw_EEG_Data/'
    electrodes = 64

    allData = np.zeros((0, electrodes + 1))
    
    for i in files:
        trial = 'R' + ('0' * (2 - len(str(i)))) + str(i) + '.edf'

        sig = pyedflib.EdfReader(basePath + subject + trial)
        n = sig.signals_in_file
        sigbuf = np.zeros((n, sig.getNSamples()[0]))
        for j in np.arange(n):
            sigbuf[j, :] = sig.readSignal(j)
        #annotation = [start time, duration, annotation(T0/T1/T2)]
        annotations = sig.read_annotation()
        data = np.asarray(sigbuf)

        rowOfLabels = np.zeros((1, data.shape[1]))
        previous = 0

        #T1 and in this set: left fist
        #T1 and not in this set: both fists
        #T2 and in this set: right fist
        #T2 and not in this set: both feet
        checkRun = {3, 4, 7, 8, 11, 12}
        sample_rate = 160

        for annotation in annotations:
            duration = int(float(annotation[1]) * sample_rate)
            note = annotation[2].decode('utf-8')
            label = 0
            if note == 'T0':
                continue
                # label = 0
            elif note == 'T1':
                if i in checkRun:
                    label = 1
                else:
                    label = 3
            else:
                if i in checkRun:
                    label = 2
                else:
                    label = 4

            rowOfLabels[0][previous:previous + duration] = label
            previous = previous + duration

        rowOfLabels[0][previous:] = label
        data = np.append(data, rowOfLabels, axis=0)
        data = np.transpose(data)
        allData = np.append(allData, data, axis=0)

    return allData