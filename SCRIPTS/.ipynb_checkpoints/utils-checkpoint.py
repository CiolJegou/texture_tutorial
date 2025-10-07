import numpy as np

from matplotlib import pyplot as plt
from matplotlib.widgets import PolygonSelector
from matplotlib.path import Path
import fileinput
import os
# from mumott.methods.projectors import SAXSProjector as Projector

def get_param(file):
    start_k = file.find('_k')+2
    end_k = file.find('_g')
    start_g = file.find('id')+2
    end_g = file.find('.npy')
    
    kernel = float(file[start_k:end_k])
    grid_res = int(file[start_g:end_g])
    return kernel, grid_res

# def get_odf(grid_res = 15, kernel = 0.1):
#     # Define reconstruction model
#     grid_resolution_parameter = grid_res # This determines the numebr of grid points
#     kernel_sigma = kernel # This determines the width of teh individual basisfunctions
#     grid = grids.hopf_grid(grid_resolution_parameter, point_groups.octahedral)
#     odf = odfs.GaussianRBF(grid, point_groups.octahedral, kernel_sigma)
#     return odf

# def get_recon(file):
#     x = np.load(file)
#     k, g = get_param(file)
#     odf = get_odf(grid_res = g, kernel = k)
#     return x, odf

def plot_mask_result(image, mask):
    fig, axs = plt.subplots(1, 3, sharex=True, sharey=True, layout='constrained', figsize=(10, 4))
    axs[0].imshow(image, origin='lower')
    axs[1].imshow(mask, origin='lower')
    axs[2].imshow(np.where(mask, image, 0), origin='lower')
    axs[0].set_title('Reconstruction')
    axs[1].set_title('Mask')
    axs[2].set_title('Masked reconstruction')
    fig.supxlabel("<-- Sample Y axis")
    fig.supylabel("Sample x axis (Beam) >")
    plt.show()
    
class InteractiveMask:
    def __init__(self, image):
        self.image = image
        self.polygon_points = None  # Store polygon points
        self.fig, self.ax = plt.subplots(layout='constrained')
        self.ax.imshow(image, origin='upper')
        self.selector = PolygonSelector(self.ax, self.on_select)
        
        # Add instruction text
        self.ax.set_title("Draw mask and press Enter to finish")
        
        # Connect event to close plot on Enter key
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.fig.supxlabel("<-- Sample Y axis")
        self.fig.supylabel("Sample x axis (Beam) >")
        
        plt.show(block=False)  # Non-blocking for Jupyter

    def on_select(self, verts):
        """Callback function for polygon selection."""
        self.polygon_points = np.array(verts, dtype=np.int32)

    def on_key(self, event):
        """Closes the figure when Enter is pressed."""
        if event.key == "enter":
            plt.close(self.fig)  # Close the figure

    def get_mask(self, doplot=True):
        """Waits for user to draw polygon and then returns the mask."""
        # Generate mask after plot is closed
        mask = np.zeros_like(self.image, dtype=bool)

        if self.polygon_points is not None and len(self.polygon_points) > 0:
            yy, xx = np.meshgrid(np.arange(self.image.shape[0]), np.arange(self.image.shape[1]), indexing="ij")
            polygon_path = Path(self.polygon_points)
            mask = polygon_path.contains_points(np.vstack((xx.ravel(), yy.ravel())).T).reshape(self.image.shape).astype(bool)
        
        if doplot:
            plot_mask_result(self.image, mask)
        
        return mask
    
def write_init(sample = 'sample', dataset = 'dataset',experiment = 'ma6510', date = '24121997', detector = 'eiger', number_of_rotations = 3620, outfile = None):
    cur_wd = os.getcwd()
    print(f'Current working directory : {cur_wd}')
    #Create a copy of the function file
    if outfile == None:
        copy_file='batch_scripts/integrate_one_projection_' + 'copy.py'
        copy_slurm='batch_scripts/schedule_integration_' + 'copy.slurm'
    else:
        copy_file= 'batch_scripts/' + outfile + '.py'
        copy_slurm='batch_scripts/' + outfile + '.slurm'
        
    os.system(f'cp batch_scripts/integrate_one_projection.py {copy_file}')
    #Change the different input variables in the integration function
    for line in fileinput.input(copy_file, inplace = True): 
        if 'sample='in line:
            print(f"sample='{sample}'", end='\n')
        elif 'dataset='in line:
            print(f"dataset='{dataset}'", end='\n')
        elif 'experiment='in line:
            print(f"experiment='{experiment}'", end='\n')
        elif 'date='in line:
            print(f"date='{date}'", end='\n')
        elif 'detector='in line:
            print(f"detector='{detector}'", end='\n')
        elif 'number_of_rotations='in line:
            print(f"number_of_rotations={str(number_of_rotations)}", end='\n')
        else:
            print(line, end= '')
            
    os.system(f'cp batch_scripts/schedule_integration.slurm {copy_slurm}')
    #Change the different input variables in the integration function
    for line in fileinput.input(copy_slurm, inplace = True): 
        if 'cd 'in line:
            print(f"cd {cur_wd}", end='\n')
        elif 'python 'in line:
            print(f"python {copy_file} $1", end='\n')
        elif '#SBATCH --output=slurm_logfiles/slurm-%j.out' in line:
            print(f"#SBATCH --output=slurm_logfiles/{dataset}-%j.out", end='\n')
        else:
            print(line, end= '')
            
def read_init(init_file = '0_init.txt'):
    #If nothing is specified, the outfile is the same as the dataset name
    if outfile is None:
        outfile = dataset
    
    #Change the different input variables in the integration function
    for line in fileinput.input('batch_scripts/integrate_one_projection.py', inplace = True): 
        if 'sample='in line:
            print(f"sample='{sample}'", end='\n')
        elif 'dataset='in line:
            print(f"dataset='{dataset}'", end='\n')
        elif 'dataset='in line:
            print(f"date='{date}'", end='\n')
        else:
            print(line, end= '')
            
    return sample, dataset, exp_number, exp_time

def downsample(full_sinogram, bin_factor_rot = 6, bin_factor_dty = 4, bin_factor_azim = 2, verbose = True):
    #If the number of rotation is not divisible by the bin factor, we remove the last points of the sinogram

    # Bin rotation angle
    down_sinogram = np.sum(full_sinogram.reshape((full_sinogram.shape[0]//bin_factor_rot, bin_factor_rot, *full_sinogram.shape[1:])), axis = 1)
    if verbose:
        print('Downsampling rotation : {}'.format(down_sinogram.shape))

    rem_dty = bin_factor_dty*(down_sinogram.shape[1]//bin_factor_dty)
    # Bin position
    down_sinogram = down_sinogram[:,:rem_dty,...]
    down_sinogram = np.sum(down_sinogram.reshape((down_sinogram.shape[0], down_sinogram.shape[1]//bin_factor_dty, bin_factor_dty, *down_sinogram.shape[2:])), axis = 2)
    if verbose:
        print('Downsampling translations : {}'.format(down_sinogram.shape))

    # Bin azimuth
    down_sinogram = np.sum(down_sinogram.reshape((*down_sinogram.shape[:3], down_sinogram.shape[3]//bin_factor_azim, bin_factor_azim, down_sinogram.shape[4])), axis = 4)
    if verbose:
        print('Downsampling azimuthal bins : {}'.format(down_sinogram.shape))
    
    return down_sinogram

def downsample_image(full_sinogram, bin_factor_rot = 6, bin_factor_azim = 2, verbose = True):
    #If the number of rotation is not divisible by the bin factor, we remove the last points of the sinogram
    # Bin rotation angle
    down_sinogram = np.sum(full_sinogram.reshape((full_sinogram.shape[0]//bin_factor_rot, bin_factor_rot, *full_sinogram.shape[1:])), axis = 1)
    if verbose:
        print('Downsampling rotation : {}'.format(down_sinogram.shape))

    # Bin azimuth
    down_sinogram = np.sum(down_sinogram.reshape((down_sinogram.shape[0], down_sinogram.shape[1]//bin_factor_azim, bin_factor_azim, down_sinogram.shape[2])), axis = 2)
    if verbose:
        print('Downsampling azimuthal bins : {}'.format(down_sinogram.shape))
    
    return down_sinogram


# def FBP(data, geometry):
#     projector = Projector(geometry)
#     filt = np.abs(np.fft.fftfreq(geometry.projection_shape[0]))
#     filt = np.clip(filt, 0, 0.25)
#     filtered_data = np.real(np.fft.ifft(np.fft.fft(data, axis = 1)*filt[np.newaxis,:,np.newaxis], axis = 1))
#     fbp = projector.adjoint(np.ascontiguousarray(filtered_data[...,np.newaxis]))[:,:,:,0]
#     return fbp

# def Rz(angle):
#     R = np.array([[np.cos(angle), -np.sin(angle), 0],
#                   [np.sin(angle), np.cos(angle), 0],
#                   [0, 0, 1],])
#     return R
