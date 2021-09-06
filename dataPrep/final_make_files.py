import pandas as pd
import numpy as np
import uproot3
from sklearn import preprocessing
import tables
from tqdm import tqdm
from timeit import default_timer as timer

pd.options.mode.chained_assignment = None  # default='warn'

file = uproot3.open("ntuple_PU200_numEvent1000.root")[b'ntuplizer;1'][b'tree;83']


def create_unique_label_background_tracks(x, y):
    x = -le.transform(y)
    return x


for i in tqdm(range(1,1000)): 
    data = file.pandas.df(["*"], flatten=True,  entrystart=i, entrystop=i+1)   # not the default
    
    data['r'] = np.sqrt(data['x']**2 + data['y']**2)
    
    data['layer_id'] = np.where(data['volume_id'] == 1, data['layer_id'] + 4, data['layer_id'])
    data['layer_id'] = np.where(data['volume_id'] == 3, data['layer_id'] + 4 + 12, data['layer_id'])
    
    background = data[data['particle_id']==-1]
    tracks = data[data['particle_id']!=-1]
    #del tracks['particle_id']
    #indices = tracks.groupby(['particle_id', 'sim_pt']).agg({'x':'count'}).reset_index()
    #del indices['x']
    #indices = indices.reset_index().rename(columns = {'index':'sim_id'})
    #tracks = tracks.merge(indices, on = ['particle_id', 'sim_pt'], how = 'right')
    
    le = preprocessing.LabelEncoder()
    le.fit(background['sim_pt'].unique())
    background['particle_id'] = create_unique_label_background_tracks(background['particle_id'], background['sim_pt'])
    final_df = tracks.append(background)
    final_df = final_df.reset_index(drop=True)
    
    final_df.to_hdf( "ntuple_PU200/new_simids/ntuple_PU200_event{}.h5".format(i), complevel=0, key='df', mode='w')
