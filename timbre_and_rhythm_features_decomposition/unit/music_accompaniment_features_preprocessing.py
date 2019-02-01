#!/usr/bin/env python
# coding: utf-8

get_ipython().run_line_magic('pylab', 'inline')
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib.pyplot as plt
from .audiofile_read import *
from .rp_extract import rp_extract
#from rp_plot import *
import librosa
import os
import pandas as pd
import scipy
import glob


def features_extraction_accompaniment(track_path):
    # adapt the fext array to your needs:
    fext = ['rp','ssd','rh','mvd'] # sh, tssd, trh
    samplerate, samplewidth, wavedata = audiofile_read(track_path, normalize=False)
    features = rp_extract(wavedata,
                      samplerate,
                      extract_rp   = ('rp' in fext),          # extract Rhythm Patterns features
                      extract_ssd  = ('ssd' in fext),           # extract Statistical Spectrum Descriptor
                      #extract_sh   = ('sh' in fext),          # extract Statistical Histograms
                      extract_tssd = ('tssd' in fext),          # extract temporal Statistical Spectrum Descriptor
                      extract_rh   = ('rh' in fext),           # extract Rhythm Histogram features
                      extract_trh  = ('trh' in fext),          # extract temporal Rhythm Histogram features
                      extract_mvd  = ('mvd' in fext),        # extract Modulation Frequency Variance Descriptor
                      spectral_masking=True,
                      transform_db=True,
                      transform_phon=True,
                      transform_sone=True,
                      fluctuation_strength_weighting=True,
                      skip_leadin_fadeout=1,
                      step_width=1)
    res = []
    for key in fext:
        res.extend(features[key])
    return res



def features_extraction_voice(track_path):
#     mfccs = librosa.feature.mfcc(wavedata, sr=sr,n_mfcc=26)
#     deltas = librosa.feature.delta(mfccs)
    wavedata, sr = librosa.load(track_path)
    discretization = np.linspace(0, len(wavedata),4,dtype=int)
    mfccs = []
    deltas = []
    for i in range(len(discretization) - 1):
        mfccs_tmp = librosa.feature.mfcc(wavedata[discretization[i]:discretization[i+1]], sr=sr,n_mfcc=26)
        mfccs.extend(np.median(mfccs_tmp,axis=1)[:12])
        deltas.extend(np.median(mfccs_tmp,axis=1)[:12])
    return mfccs + deltas


def track_preprocessing(track_path):
    return features_extraction_accompaniment(track_path) + features_extraction_voice(track_path)


def create_tracks_features_space(features_data_folder, audio_folder_path):
    genres = os.listdir(audio_folder_path+'/')
    for genre in genres:
        tracks = os.listdir(audio_folder_path+'/'+i+'/')
        for track in tracks:
            tmp = track_preprocessing(audio_folder_path+'/'+genre+'/'+track)
            pd.DataFrame(tmp).to_csv(features_data_folder+'/'+track[:-4]+'.csv', header=None, index=None)


def calculate_distances(features_data_folder):
    d = {}
    import os
    del os
    import os
    for file in os.listdir(features_data_folder+'/'):
        d[file] = pd.read_csv(features_data_folder+'/'+file,header=None, engine='python')[0].values
    distances = pd.DataFrame(columns=d.keys(),index=d.keys())
    for basic_track in d:
        for compared_track in d:
            distances.loc[basic_track,compared_track] = scipy.spatial.distance.cosine(d[basic_track],d[compared_track])
    distances.to_excel('tracks_similarity_matrix.xlsx')


def formulate_recommendations():
    recommendations = {}
    distances = pd.read_excel('tracks_similarity_matrix.xlsx')
    for col in distances.columns:
        recommendations[col] = distances[col].sort_values()[:20].index
    pd.DataFrame(recommendations).to_excel('recommendations_for_tracks.xlsx',index=None)


def recommendation_for_new_track(tracks_path, features_data_folder):
    tp = track_preprocessing(tracks_path)
    d = {}
    for file in os.listdir(features_data_folder+'/'):
        d[file] = pd.read_csv(features_data_folder+'/'+file,header=None, engine='c')[0].values
    dist = pd.DataFrame(columns=['dist'], index=d.keys())

    for col in d.keys():
        dist.loc[col,'dist'] = scipy.spatial.distance.cosine(d[col],tp)
    return dist['dist'].sort_values()[:20].index


def add_track_in_base(track_path, features_data_folder):
    tp = track_preprocessing(track_path)
    if track_path.split('/')[-1][:-4]+'.csv' not in os.listdir(features_data_folder+'/'):
        pd.DataFrame(tp).to_csv(features_data_folder+'/'+track_path.split('/')[-1][:-4]+'.csv',header=None,index=None)
        calculate_distances()
        formulate_recommendations()
        print('added')
    else:
        print('already in db')


def create_database_from_folder(audio_folder_path, features_data_folder):
    #tracks = os.listdir(audio_folder_path)
    tracks = glob.glob(audio_folder_path + '/**/*.mp3', recursive=True)
    #print(tracks)
    for track in tracks:
        if track[-4:].lower()=='.mp3':
            tmp = track_preprocessing(track)
            track_name_data = track.split("/")
            track_name_data = track_name_data[::-1]
            pd.DataFrame(tmp).to_csv(features_data_folder+'/'+track_name_data[0][:-4]+'.csv', header=None, index=None)
