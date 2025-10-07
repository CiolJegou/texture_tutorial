import h5py
import sys
from odftt.tomography_models import FromArrayModel
from odftt.optimization import FISTA
import numpy as np

from mumott import Geometry 

z_slice_index = int(sys.argv[1])
maxiter = int(sys.argv[2])
sample = sys.argv[3]

metadata_folder = 'auxillary_files/'
with h5py.File(metadata_folder+f'{sample}_corrected_data.h5', 'r') as file:
    data_array = np.array(file['data_array'][:,:,z_slice_index:z_slice_index+1,:,:])    
    gap_mask = np.array(file['gap_mask'])

weigth_array = np.ones(data_array.shape)
weigth_array[:,:,:,~gap_mask] = False

# Should maybe be read from a file instead
geometry = Geometry()
geometry.read(metadata_folder+'2d.mumottgeometry')

# Model quantitites
basis_function_arrys = []
with h5py.File(metadata_folder+'ODF_model_data.h5', 'r') as file:
    step_size_parameter = float(file['step_size_parameter'][...])
    for rot_index in range(data_array.shape[0]):
        basis_function_arrys.append(np.array(file[f'pf_matrices/rot_{rot_index}']))

model_RBF = FromArrayModel(basis_function_arrys, geometry)

# Do reconstruction
optimizer = FISTA(model_RBF, data_array, weights = weigth_array, maxiter = maxiter, step_size_parameter=step_size_parameter)
x_RBF, convergence_curve = optimizer.optimize()

with h5py.File(f'reconstruction_output_z{z_slice_index}.h5', 'w') as file:
    file.create_dataset('coefficients', data=x_RBF)
    file.create_dataset('convergence_curve', data=convergence_curve)
