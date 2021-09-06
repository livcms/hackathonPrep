"""
Data preparation script for GNN tracking.
This script processes the TrackML dataset and produces graph data on disk.
"""

# System
import os
import sys
import time
import argparse
import logging
import multiprocessing as mp
from functools import partial
sys.path.append("../")

# Externals
import yaml
import pickle
import numpy as np
import pandas as pd
import trackml.dataset
import time

# Locals
from collections import namedtuple
import numpy as np

Graph = namedtuple('Graph', ['x', 'edge_attr', 'edge_index', 'y', 'pid', 'pt', 'eta'])

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser('prepare.py')
    add_arg = parser.add_argument
    add_arg('config', nargs='?', default='configs/prepare_trackml.yaml')
    add_arg('--n-workers', type=int, default=1)
    add_arg('--task', type=int, default=0)
    add_arg('--n-tasks', type=int, default=1)
    add_arg('-v', '--verbose', action='store_true')
    add_arg('--show-config', action='store_true')
    add_arg('--interactive', action='store_true')
    add_arg('--start-evtid', type=int, default=1000)
    add_arg('--end-evtid', type=int, default=3000)
    return parser.parse_args()





def calc_dphi(phi1, phi2):
    """Computes phi2-phi1 given in range [-pi,pi]"""
    dphi = phi2 - phi1
    dphi[dphi > np.pi] -= 2*np.pi
    dphi[dphi < -np.pi] += 2*np.pi
    return dphi


def select_hits(data, pt_min):
    
    #data_sub = data[data['evt'] == evtid]
    pt_cut_data = data[data['sim_pt'] > pt_min]
    
    return pt_cut_data 


def calc_eta(r, z):
    theta = np.arctan2(r, z)
    return -1. * np.log(np.tan(theta / 2.))



def select_segments(hits1, hits2, phi_slope_max, z0_max,
                    layer1, layer2,
                    remove_intersecting_edges=False):
    """
    Construct a list of selected segments from the pairings
    between hits1 and hits2, filtered with the specified
    phi slope and z0 criteria.
    Returns: pd DataFrame of (index_1, index_2), corresponding to the
    DataFrame hit label-indices in hits1 and hits2, respectively.
    """

    # Start with all possible pairs of hits
    keys = ['evt', 'r', 'phi', 'z']
    hit_pairs = hits1[keys].reset_index().merge(
        hits2[keys].reset_index(), on='evt', suffixes=('_1', '_2'))
    #print(hit_pairs)
    # Compute line through the points
    dphi = calc_dphi(hit_pairs.phi_1, hit_pairs.phi_2)
    dz = hit_pairs.z_2 - hit_pairs.z_1
    dr = hit_pairs.r_2 - hit_pairs.r_1
    eta_1 = calc_eta(hit_pairs.r_1, hit_pairs.z_1)
    eta_2 = calc_eta(hit_pairs.r_2, hit_pairs.z_2)
    deta = eta_2 - eta_1
    dR = np.sqrt(deta**2 + dphi**2)
    phi_slope = dphi / dr
    z0 = hit_pairs.z_1 - hit_pairs.r_1 * dz / dr
    
    # Apply the intersecting line cut
    intersected_layer = dr.abs() < -1
    #if remove_intersecting_edges:

        # Innermost barrel layer --> innermost L,R endcap layers
     #   if (layer1 == 0) and (layer2 == 11 or layer2 == 4):
     #       z_coord = 71.56298065185547 * dz/dr + z0
     #       intersected_layer = np.logical_and(z_coord > -490.975,
    #                                           z_coord < 490.975)
      #  if (layer1 == 1) and (layer2 == 11 or layer2 == 4):
      #      z_coord = 115.37811279296875 * dz / dr + z0
      #      intersected_layer = np.logical_and(z_coord > -490.975,
    #                                           z_coord < 490.975)

    # Filter segments according to criteria
    #good_seg_mask = (phi_slope.abs() > -1000000) 
    good_seg_mask = ((phi_slope.abs() < phi_slope_max) &
                     (z0.abs() < z0_max) &
                     (intersected_layer == False))

    dr = dr[good_seg_mask]
    dphi = dphi[good_seg_mask]
    dz = dz[good_seg_mask]
    dR = dR[good_seg_mask]
    return hit_pairs[['subentry_1', 'subentry_2']][good_seg_mask], dr, dphi, dz, dR


def construct_graph(hits, layer_pairs, phi_slope_max, z0_max,
                    feature_names, feature_scale, prefix,
                    remove_intersecting_edges = False):
    """Construct one graph (e.g. from one event)"""
    
    t0 = time.time()
    #print(hits.index.get_level_values(0).unique())
    hits = hits.droplevel(0)
    #print(hits.index)
    # Loop over layer pairs and construct segments
    layer_groups = hits.groupby('new_layer_ids')
    segments = []
    seg_dr, seg_dphi, seg_dz, seg_dR = [], [], [], []
    for (layer1, layer2) in layer_pairs:
        # Find and join all hit pairs
        try:
            hits1 = layer_groups.get_group(layer1)
            hits2 = layer_groups.get_group(layer2)
        # If an event has no hits on a layer, we get a KeyError.
        # In that case we just skip to the next layer pair
        except KeyError as e:
            logging.info('skipping empty layer: %s' % e)
            continue
        # Construct the segments
        selected, dr, dphi, dz, dR = select_segments(hits1, hits2, phi_slope_max, z0_max,
                                                     layer1, layer2, 
                                                     remove_intersecting_edges=remove_intersecting_edges)
        segments.append(selected)
        seg_dr.append(dr)
        seg_dphi.append(dphi)
        seg_dz.append(dz)
        seg_dR.append(dR)
        
    # Combine segments from all layer pairs

    segments = pd.concat(segments)
    seg_dr, seg_dphi = pd.concat(seg_dr), pd.concat(seg_dphi)
    seg_dz, seg_dR = pd.concat(seg_dz), pd.concat(seg_dR)

    # Prepare the graph matrices
    n_hits = hits.shape[0]
    n_edges = segments.shape[0]
    X = (hits[feature_names].values / feature_scale).astype(np.float32)
    edge_attr = np.stack((seg_dr/feature_scale[0], 
                          seg_dphi/feature_scale[1], 
                          seg_dz/feature_scale[2], 
                          seg_dR))

    y = np.zeros(n_edges, dtype=np.float32)

    # We have the segments' hits given by dataframe label,
    # so we need to translate into positional indices.
    # Use a series to map hit label-index onto positional-index.
    hit_idx = pd.Series(np.arange(n_hits), index=hits.index)
    seg_start = hit_idx.loc[segments.subentry_1].values
    seg_end = hit_idx.loc[segments.subentry_2].values
    edge_index = np.stack((seg_start, seg_end))

    # Fill the segment, particle labels
    pid = hits.particle_id
    # this is for partilce? 
    pt = hits.sim_pt
    eta = hits.sim_eta
    unique_pid_map = {pid_old: pid_new 
                      for pid_new, pid_old in enumerate(np.unique(pid.values))}
    pid_mapped = pid.map(unique_pid_map)
    #print(hits.particle_id.droplevel(0), '\n',  segments)
    #hits = hits.droplevel(0)

    pid1 = hits.particle_id.loc[segments.subentry_1].values
    pid2 = hits.particle_id.loc[segments.subentry_2].values
    y[:] = (pid1 == pid2)

    # Correct for multiple true barrel-endcap segments
    layer1 = hits.new_layer_ids.loc[segments.subentry_1].values
    layer2 = hits.new_layer_ids.loc[segments.subentry_2].values
    #print("layer 1: ",layer1," \n layer 2 :", layer2)
    true_layer1 = layer1[y>0.5]
    true_layer2 = layer2[y>0.5]
    true_pid1 = pid1[y>0.5]
    true_pid2 = pid2[y>0.5]
    true_z1 = hits.z.loc[segments.subentry_1].values[y>0.5]
    true_z2 = hits.z.loc[segments.subentry_2].values[y>0.5]
    pid_lookup = {}
    for p, pid in enumerate(np.unique(true_pid1)):
        pid_lookup[pid] = [0, 0, -1, -1]
        l1, l2 = true_layer1[true_pid1==pid], true_layer2[true_pid2==pid]
        z1, z2 = true_z1[true_pid1==pid], true_z2[true_pid2==pid]
        for l in range(len(l1)):
            barrel_to_LEC = (l1[l] in [1,2,3, 4] and l2[l]==5)
            barrel_to_REC = (l1[l] in [1,2,3, 4] and l2[l]==17)
    #        if (barrel_to_LEC or barrel_to_REC):
     #           temp = pid_lookup[pid]
      #          temp[0] += 1
       #         if abs(temp[1]) < abs(z1[l]):
        #            if (temp[0] > 1):
         #               
          #              print(" *** adjusting y for triangle edge pattern")
           #             print("     > replacing edge (l1={}, l2={}, z1={:.2f})\n"
            #                  .format(temp[2], temp[3], temp[1]),  
             #                 "      with      edge (l1={}, l2={}, z1={:.2f})"
              #                .format(l1[l], l2[l], abs(z1[l]))
               #         )

#                           y[(pid1==pid2) & (pid1==pid) & 
 #                         (layer1==temp[2]) & (layer2==temp[3])] = 0
#
 #                   temp[1] = abs(z1[l])
  #                  temp[2] = l1[l]
   #                 temp[3] = l2[l]
#
 #               pid_lookup[pid] = temp
    
    print("... completed in {0} seconds".format(time.time()-t0))
    return Graph(X, edge_attr, edge_index, y, pid_mapped, pt, eta)


def split_detector_sections(hits, phi_edges, eta_edges):
    """Split hits according to provided phi and eta boundaries."""
    hits_sections = []
    # Loop over sections
    for i in range(len(phi_edges) - 1):
        phi_min, phi_max = phi_edges[i], phi_edges[i+1]
        # Select hits in this phi section
        phi_hits = hits[(hits.phi > phi_min) & (hits.phi < phi_max)]
        # Center these hits on phi=0
        centered_phi = phi_hits.phi - (phi_min + phi_max) / 2
        phi_hits = phi_hits.assign(phi=centered_phi, phi_section=i)
        for j in range(len(eta_edges) - 1):
            eta_min, eta_max = eta_edges[j], eta_edges[j+1]
            # Select hits in this eta section
            eta = calc_eta(phi_hits.r, phi_hits.z)
            sec_hits = phi_hits[(eta > eta_min) & (eta < eta_max)]
            hits_sections.append(sec_hits.assign(eta_section=j))

    return hits_sections


def process_event(file, output_dir, pt_min, n_eta_sections, n_phi_sections,
                  eta_range, phi_range, phi_slope_max, z0_max, phi_reflect,
                  endcaps, remove_intersecting_edges):


    data = pd.read_hdf(file)
    data = data[['evt', 'hit_id', 'x', 'y', 'z', 'r','sim_type', 'sim_id', 'sim_pt', 'sim_eta', 'sim_phi', 'particle_id', 'volume_id', 'layer_id', 'new_layer_ids']]
    data['phi'] = np.arctan2(data.y, data.x)                 

    #evtid = prefix 
    prefix = file.split("ntuple_PU200_")[1].split('.h5')[0].strip()
    #print("the prefix is ", prefix)
    hits = select_hits(data, pt_min)
    phi_edges = np.linspace(*phi_range, num=n_phi_sections+1)
    eta_edges = np.linspace(*eta_range, num=n_eta_sections+1)

    hits_sections = split_detector_sections(hits, phi_edges, eta_edges)

        # Graph features and scale
    feature_names = ['r', 'phi', 'z']
    feature_scale = np.array([1000., np.pi / n_phi_sections, 1000.])
    if (phi_reflect): feature_scale[1] *= -1

    # Define adjacent layers
    n_det_layers = 5
    l = np.arange(1,n_det_layers)
    layer_pairs = np.stack([l[:-1], l[1:]], axis=1)


    n_det_layers = 18
    EC_L = np.arange(5, 17)
    EC_L_pairs = np.stack([EC_L[:-1], EC_L[1:]], axis=1)
    layer_pairs = np.concatenate((layer_pairs, EC_L_pairs), axis=0)
    EC_R = np.arange(17, 29)
    EC_R_pairs = np.stack([EC_R[:-1], EC_R[1:]], axis=1)
    layer_pairs = np.concatenate((layer_pairs, EC_R_pairs), axis=0)
    barrel_EC_L_pairs = np.array([(0,5), (1,5), (2,5), (3,5)])
    barrel_EC_R_pairs = np.array([(0,17), (1,17), (2,17), (3,17)])
    layer_pairs = np.concatenate((layer_pairs, barrel_EC_L_pairs), axis=0)
    layer_pairs = np.concatenate((layer_pairs, barrel_EC_R_pairs), axis=0)

    graphs = [construct_graph(section_hits, layer_pairs=layer_pairs,
                          phi_slope_max=phi_slope_max, z0_max=z0_max,
                          feature_names=feature_names,
                          feature_scale=feature_scale,
                          prefix=prefix,
                          remove_intersecting_edges=remove_intersecting_edges)
          for section_hits in hits_sections]


    try:
#        base_prefix = os.path.basename(prefix)
 #       print(base_prefix)
 #       print(prefix)
        filenames = [os.path.join(output_dir, '%s_g%03i' % (prefix, i))
                     for i in range(len(graphs))]
 #       print(filenames)
    except Exception as e:
        logging.info(e)
        print("I am in exception")
    
    logging.info('Event %i, writing graphs', prefix)    
    for graph, filename in zip(graphs, filenames):
        np.savez(filename, ** dict(x=graph.x, edge_attr=graph.edge_attr,
                                   edge_index=graph.edge_index, 
                                   y=graph.y, pid=graph.pid, pt=graph.pt, eta=graph.eta))


def main(): 
#    file = pd.read_pickle("~/validationNtuple/ntuple.pkl") 
 

    args = parse_args()
    with open(args.config) as f:
        config = yaml.load(f, yaml.FullLoader)
    if args.task == 0:
        logging.info('Configuration: %s' % config)

    input_dir = config['input_dir']
    all_files = [os.path.join(input_dir, file) for file in os.listdir(input_dir)]
    #print(all_files)
    
    
    output_dir = "output" 
    os.makedirs(output_dir, exist_ok=True) 
        

    #file_prefixes = np.unique(data['evt'])
    t0 = time.time()
    with mp.Pool(processes=args.n_workers) as pool:
        process_func = partial(process_event, output_dir=output_dir,
                               phi_range=(-np.pi, np.pi), **config['selection'])
        pool.map(process_func, all_files)
    t1 = time.time()
    print("Finished in", t1-t0, "seconds")
    

if __name__ == '__main__':
    main()


