import os 
import sys
import argparse 
sys.path.append("../")
import logging 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import uproot3
import pytest
import json 
import warnings 
import requests
from particle import Particle

#from parameterized import parameterized, param

# this is used to redefine the way the layers are numbered
# the convention used so far is easy for validation
# redefined to make it easier to adapt to GNN. The GNN code should be adapted instead, this transform is very slow! 

def redefine_layer_id(volume_id, layer_id): 
    """Instead of numbering by volume and then layer, so each volume counts layer from 1, make every layer a unique number""" 
    # left side endcap
    if volume_id == 1: 
        layer_id = layer_id + 4
    # right side endcap 
    elif volume_id == 3: 
        layer_id = layer_id + 4+12 
    return layer_id



# a pytest fixture gives data to other functions dependent on it 
#l1 = np.arange(0,1000, 100)
#l2 = np.arange(100,1100, 100)

#l1 = np.arange(0,1000, 10)
#l2 = np.arange(10,1100, 10)

#event_start_stop = list(zip(l1, l2))
directory = r'/home/lhv14/validationNtuple/ntuple_PU200/new_simids/'
files = os.listdir(directory)

pixelBarrel = pd.read_csv('PixelBarrel.csv', index_col = None, header=None).T


@pytest.fixture(scope="session", params = files)
def start_stop_point(request): 
    return request.param

#def pytest_configure(): 
#    file = uproot3.open(pytestconfig.getoption("filename"))[b'ntuplizer;1'][b'tree;84']
#
#    # Flattening the data takes a bit of time and probably won't scale well, but much easier to work with 
#    #data = file.pandas.df(["*"], flatten=True, entrystart=int(pytestconfig.getoption("event_start_point")), entrystop=int(pytestconfig.getoption("event_stop_point")))
#    data = file.pandas.df(["*"], flatten=True, entrystart = 0, entrystop=100)
#    #print(len(data))
#    # remove all hits not associated to track 
#    #data = data[data['particle_id'] > -1]
#    data['r'] = np.sqrt(data['x']**2 + data['y']**2)
#    data['new_layer_ids'] = data.apply(lambda x: redefine_layer_id(x.volume_id, x.layer_id), axis=1)
#    pytest.data = data 
#


@pytest.fixture(scope="session")
def data(pytestconfig, start_stop_point):
   """Reads in the data. The default and arguments you can parse are defined in conftest.py. This is automatically linked at runtime""" 


   # NB! The path needs changing with different tree structures! 
   #file = uproot3.open(pytestconfig.getoption("filename"))[b'ntuplizer;1'][b'tree;83']
   data = pd.read_hdf(directory+start_stop_point)
   # Flattening the data takes a bit of time and probably won't scale well, but much easier to work with 
   #data = file.pandas.df(["*"], flatten=True, 
   #data = file.pandas.df(["*"], flatten=True, entrystart=int(pytestconfig.getoption("event_start_point")), entrystop=int(pytestconfig.getoption("event_stop_point")))
   #data = file.pandas.df(["*"], flatten=True, entrystart = start_stop_point[0], entrystop=start_stop_point[1])
   #print(len(data))
   # remove all hits not associated to track 
   #data = data[data['particle_id'] > -1]
   #data['r'] = np.sqrt(data['x']**2 + data['y']**2)
   #data['new_layer_ids'] = data.apply(lambda x: redefine_layer_id(x.volume_id, x.layer_id), axis=1)
   # save file after transformations 
  # pd.to_pickle(data, "ntuple_PU200_event{}_to{}.pkl".format(0, 100))
   #print(np.unique(data.index.get_level_values(0)))
   #data = pd.read_pickle(directory+file)
   return data


# the function name data is called to give the data 
def test_missing_values(data): 
    """Checks if there are any NaN values in the data""" 
    
    missing_values = data.isnull().values.any()
    assert missing_values ==False, "There are missing values in the data"



def test_duplicates(data): 
    """Checks for duplicated rows or duplicated hitids within each event""" 

    # check if any duplicated rows 
    duplications = data.duplicated().any()
    # check if any duplicated hit ids within an event 
    agg_by_entry_hitid  = data.groupby(['hit_id']).agg({'hit_id':'count'})
    duplicated_hitid_in_event = agg_by_entry_hitid['hit_id'].max() > 1
    assert  (duplications == False ) and (duplicated_hitid_in_event == False), "There is duplicated data. Duplicated rows returned "+ duplications+ " and duplicated hitids in event returned "+ duplicated_hitid_in_event  

def test_run_lumi(data): 
    """Checks that there are single values for lumi and run throughout the dataset""" 
    
    assert (len(data['run'].unique()) == 1) and (len(data['lumi'].unique()) == 1), "There are multiple run numbers or lumi values within this data"   



def test_evt_ids(data): 
    """Check that evt is the same throguhout each event"""
    
    # check that the length of the unique values of evt is the same as the length of the event index 
    # this will also check that the same evt id is consistent through each event 

    agg_by_evt = data.groupby(['evt']).agg({'evt':'count'})
    num_unique_evt = len(agg_by_evt.index.get_level_values(1))
    num_unique_events = len(data.index.get_level_values(0).unique())
    assert  num_unique_evt == num_unique_events, "The number of unique evt identifiers does not match the number of events in the data. For evt this is " +  str(num_unique_evt) + " and there are " + str(num_unique_events) + " events" 
   

def test_nhit(data): 
    """Number of hits in each event must be bigger than 80 000 (very arbitrary number)"""
    # it's assumed this number is constant through each event, could be double checked 
    assert (False in data['nhit'].unique() > 80000) == False, "There is one or several events that has less than 80 000 hits" 


def test_hit_position_pixel_barrel(data): 
    """ Tests that all hits are within the r posisiton of the layers in the pixel barrel """ 

    # a couple of transformations to make table easier to read
    pixelBarrel.columns= pixelBarrel.iloc[0,:]
    pixelBarrel = pixelBarrel.shift(-1).iloc[0:-1] 
    # make same units 
    pixelBarrel = pixelBarrel.apply(lambda x: x/10 if x.name in ['r', 'z_max'] else x)
    checks_passed = []
    for i in range(1,5):
        #instead of checking all values we can check just max and min, if they pass, all other hits will pass 
        # the positions in the tables are the middle (?) of the layers, so include a range of values hits may lie within. Numbers arbitrarily chosen based on what's in data

        checks_passed.append(min(data[(data['volume_id']== 2) & (data['layer_id'] == i)]['r']) > (pixelBarrel['r'][i-1] - 0.4))
        checks_passed.append(max(data[(data['volume_id']== 2) & (data['layer_id'] == i)]['r']) < (pixelBarrel['r'][i-1] + 0.4))
    
    assert False not in checks_passed, "There are hits in the pixel barrel that are more than 0.4 cm away from the hit layer position" 


def test_hit_position_pixel_endcaps(data): 
    """ Tests that all hits are within the z position of the pixel endcaps """ 

    pixelEndcaps = pd.read_csv('PixelEndcap.csv', index_col = None, header=None).T
    pixelEndcaps.columns= pixelEndcaps.iloc[0,:]
    pixelEndcaps = pixelEndcaps.shift(-1).iloc[0:-1] 
    pixelEndcaps = pixelEndcaps.apply(lambda x: x/10 if x.name in ['r', 'z'] else x)
    pixelEndcaps['Disk'] = range(1,13)
    checks_passed = []
    for i in range(1,12):
        # check both positive and negative endcaps, so volume 1 and 3  

        checks_passed.append(min(data[(data['volume_id']== 1) & (data['layer_id'] == i)]['z']) > (-pixelEndcaps['z'][i-1] - 0.8))
        checks_passed.append(max(data[(data['volume_id']== 1) & (data['layer_id'] == i)]['z']) < (-pixelEndcaps['z'][i-1] + 0.8))
    
        checks_passed.append(min(data[(data['volume_id']== 3) & (data['layer_id'] == i)]['z']) > (pixelEndcaps['z'][i-1] - 0.8))
        checks_passed.append(max(data[(data['volume_id']== 3) & (data['layer_id'] == i)]['z']) < (pixelEndcaps['z'][i-1] + 0.8))
    
    assert False not in checks_passed, "There are hits in the pixel endcap that are more than 0.8 cm away from the hit layer position" 


def test_pdg_id(data): 
    """Checks that all the pdg_ids are within the pdg convention.""" 
    
    passed_list = []
    for pid in data.pdg_id.unique():
        if pid != 0: 
            try: 
                # this throws an error if the pid is not recognised 
                Particle.from_pdgid(pid)
            except: 
                passed_list.append(pid)
        
    assert len(passed_list)==0, "There are particles with ids that are not recognised "+ str(passed_list) 

def test_pt_eta_phi(data): 
    """Checks that the sim pt eta and phi are within the accepted values """ 
    pt_passed = max(data.sim_pt) < 150
    eta_passed = (min(data.sim_eta) > -10) & (max(data.sim_eta) < 10)
    phi_passed = (min(data.sim_phi) > -np.pi) & (max(data.sim_phi) < np.pi)
    assert pt_passed & eta_passed & phi_passed, "Pt less than 100 GeV returned "+ str(pt_passed) + " pseudorapidity between -10 and 10 returned "+ str(eta_passed) + " phi between -pi and pi returned " + str(phi_passed)

def test_nhits_per_track(data): 
    """Checks the number of hits per track. It checks how many tracks have number of hits > 50 and < 3. Events outside these boundaries will pass, but a warning is printed""" 
    particle_id_by_event = data.groupby(['entry', 'particle_id']).agg({'particle_id':'count'})
    number_tracks_w_more_than_40_hits = particle_id_by_event[particle_id_by_event['particle_id'] > 50]
    number_tracks_w_less_than_3_hits = particle_id_by_event[particle_id_by_event['particle_id'] < 3]
    if len(number_tracks_w_more_than_40_hits) > 0: 
        warnings.warn(UserWarning("Warning, there are "+ str(len(number_tracks_w_more_than_40_hits))+ " tracks with more than 40 hits. These  are: ", number_tracks_w_more_than_40_hits))
    if len(number_tracks_w_less_than_3_hits) > 0: 
        warnings.warn(UserWarning("Warning, there are "+ str(len(number_tracks_w_less_than_3_hits))+ " tracks with less than 3 hits. These  are: ", number_tracks_w_less_than_3_hits))
    pass


def test_volume_and_layer_id(data): 
    """Check that all volume and layer ids are within accepted values""" 
    volume_accepted = (max(data['volume_id']) == 3) & (min(data['volume_id']) ==1)
    layer_accepted_barrel = (max(data[data['volume_id']==2]['layer_id']) == 4) & min(data[data['volume_id']==2]['layer_id']) == 1
    layer_accepted_endcaps = (max(data[data['volume_id']==1]['layer_id']) == 12) & min(data[data['volume_id']==1]['layer_id']) == 1
    layer_accepted_endcaps_2 = (max(data[data['volume_id']==3]['layer_id']) == 12) & min(data[data['volume_id']==3]['layer_id']) == 1

    assert volume_accepted & layer_accepted_barrel & layer_accepted_endcaps & layer_accepted_endcaps_2, "There are layers that are not within the specified region of interest" 


    
