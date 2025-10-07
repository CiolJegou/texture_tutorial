import h5py
import hdf5plugin
import matplotlib.pyplot as plt
import numpy as np
import pyFAI
import fabio
import pickle
import os
from tqdm import tqdm
import sys

#Function to load 1 frame
def load_frame(position_number, rotation_number):
    datafile_fn = f'/data/visitor/{experiment}/id11/{date}/RAW_DATA/{sample}/{sample}_{dataset}/scan{position_number+1:04d}/{detector}_0000.h5'
    with h5py.File(datafile_fn, 'r') as file:
        data = np.array(file['entry_0000/measurement/data'][rotation_number,:,:])
    return data

#Function to process 1 frame
def process_frame(image_data):
    #Load integration metadata from the example
    integration_args2D = {'npt_rad':integ_metadata['npt_rad'], 'npt_azim':integ_metadata['npt_azim'],
                          'unit': integ_metadata['unit'],'radial_range':integ_metadata['radial_range'],
                          'mask':mask, 'polarization_factor':integ_metadata['polarization_factor']}
    
    cake_data, _, _  = ai.integrate2d(image_data, **integration_args2D)
    cake_data = cake_data - phase_data['bg'][np.newaxis, :]
    output_data = []
    for peak_mask in phase_data['peak_masks']:
        this_peak_data = np.sum(cake_data[:, peak_mask], axis = -1)
        output_data.append(this_peak_data)

    return np.stack(output_data, axis = -1)

#load the position index from the batch file
position_index = int(sys.argv[1])

#No space around the "=" sign 
sample='textom_posth14_post_beamdown'
dataset='z12_redo'
experiment='ma6795'
date='20250826'
detector='eiger'
data_key=f'/measurement/{detector}'
number_of_rotations=1800

# Load metadata files
path_texture = 'auxillary_files/'
with open(path_texture + 'phase_data.npy', 'rb') as fid:
    phase_data = pickle.load(fid)

with open(path_texture + 'integration_metadata.npy', 'rb') as fid:
    integ_metadata = pickle.load(fid)

# Set up azimuthal integrator
mask = fabio.open(integ_metadata['mask_path']).data
ai = pyFAI.load(integ_metadata['poni_path'])

# Allocate arrays for data
n_hkl = len(phase_data['hkl_list'])
full_data_array = np.zeros((number_of_rotations, 1, integ_metadata['npt_azim'], n_hkl))
# full_COM_array = np.zeros((number_of_rotations, 1, integ_metadata['npt_azim'], n_hkl))

#For loop to go through all rotation angles
for rotation_index in range(number_of_rotations):
    # read out data
    try:
        #This loads the frame 1 by 1 because sometimes the eiger misses a frame 
        data = load_frame(position_index, rotation_index)
        data=np.where(data>2e5,1e5,data)
    except FileNotFoundError:
        print(f'Rotation {rotation_index}: Y Position: {position_index}')
        data = load_frame(position_index-1, rotation_index)
        data=np.where(data>2e5,1e5,data)
    #This output is a 2d array of size (n_azim, n_hkl)
    output = process_frame(data)

    com_data = np.sum(output, axis = 0)
    full_data_array[rotation_index, 0, :, :] = output
    # full_COM_array[rotation_index, 0, :, :] = output/com_data         

#Create a folder to save the integrated intensities
output_folder = f'/data/visitor/{experiment}/id11/{date}/PROCESSED_DATA/integrated_intensities/{sample}/{sample}_{dataset}/'
    
# Write this position to the datafile
with h5py.File(output_folder+f'scan{position_index+1:04d}.h5', 'w') as file:
    grp = file.create_group(f'pos_{position_index}')

    grp.create_dataset('intensity', data = full_data_array)
    # grp.create_dataset('COM', data = full_COM_array)


