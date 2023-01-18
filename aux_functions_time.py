import numpy as np
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import os
from pyDOE import lhs
import nibabel as nib
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import meshio
from nibabel.nifti1 import save
from nilearn.image.image import load_img
import pyvista
import scipy.io as spio
import pydicom

def create_stack(data_path):

    result_img = load_img( os.path.join(data_path, 't*.nii.gz') )
    name = os.path.join( data_path, 'imt.nii.gz')
    save(result_img, name)

    return result_img

def im_and_segm_mesh_hernan(crop_x_in, crop_x_end, crop_y_in, crop_y_end, data_path):
    
    # Get volunteer string from data_path
    volunteer = os.path.basename( os.path.normpath( data_path ))

    #path = 'MRI_images/registered_data/'+volunteer+'/'+volunteer+'.mat'
    path = os.path.join('registered_data',volunteer,volunteer+'.mat')
    v = spio.loadmat(path,struct_as_record=True)

    # Get information
    Xi = v['Xi']  # x coordinate
    Yi = v['Yi']  # y coordinate
    Zi = v['Zi']  # z coordinate
    I = v['Ii']   # image intensity

    Xi = np.transpose(Xi, [2,0,1])
    Yi = np.transpose(Yi, [2,0,1])
    Zi = np.transpose(Zi, [2,0,1])

    Xi = Xi[:, crop_y_in:crop_y_end, crop_x_in:crop_x_end]
    Yi = Yi[:, crop_y_in:crop_y_end, crop_x_in:crop_x_end]
    Zi = Zi[:, crop_y_in:crop_y_end, crop_x_in:crop_x_end]

    Xi = -Xi.ravel()[:,None]
    Yi = -Yi.ravel()[:,None]
    Zi = Zi.ravel()[:,None]
    # Mesh coordinates in a list
    im_mesh = np.hstack((Xi, Yi, Zi))

    # Nodes in volumetric mesh of left ventricle 
    vol_mesh = meshio.read( os.path.join(data_path, volunteer+'_vol.vtk') )
    # Segmentation mesh
    segm_mesh = vol_mesh.points

    return im_mesh, segm_mesh

def im_and_segm_mesh(imt_data, crop_x_in, crop_x_end, crop_y_in, crop_y_end, data_path):
    
    # Get volunteer string from data_path
    volunteer = os.path.basename( os.path.normpath( data_path ))

    # Header of images to get information
    header = imt_data.header

    pix_dim = header['pixdim']
    # Pixel spacing of images
    pixsp_x = pix_dim[1]
    pixsp_y = pix_dim[2]
    # Slice thickness
    pixsp_z = pix_dim[3]

    dim = header['dim']

    Nx = dim[1]
    Ny = dim[2]
    Nz = dim[3]

    xp = np.arange(Nx)
    yp = np.arange(Ny)
    zp = np.arange(Nz)

    Y, Z, X = np.meshgrid(yp, zp, xp)

    X = X[:, crop_y_in:crop_y_end, crop_x_in:crop_x_end]
    Y = Y[:, crop_y_in:crop_y_end, crop_x_in:crop_x_end]
    Z = Z[:, crop_y_in:crop_y_end, crop_x_in:crop_x_end]

    X = X.ravel()[:,None]
    Y = Y.ravel()[:,None]
    Z = Z.ravel()[:,None]
    # Pixel coordinates in a list
    im_pix = np.hstack((X, Y, Z))
    im_mesh = pixel_to_mesh(im_pix, pixsp_x, pixsp_y, pixsp_z)

    # Nodes in volumetric mesh of left ventricle 
    vol_mesh = meshio.read( os.path.join(data_path, volunteer+'_vol.vtk') )
    # Segmentation mesh
    segm_mesh = vol_mesh.points

    return im_mesh, segm_mesh

def background_mesh(data_path, imt_data, crop_x_in, crop_x_end, crop_y_in, crop_y_end, bg_mesh_file, slices):
    
    if os.path.exists( bg_mesh_file ):
        bg_mesh = np.load( bg_mesh_file )
    else:
        generate_background_mesh(data_path, imt_data, crop_x_in, crop_x_end, crop_y_in, crop_y_end, slices)
        bg_mesh = np.load( bg_mesh_file )

    return bg_mesh

def generate_background_mesh(data_path, imt_data, crop_x_in, crop_x_end, crop_y_in, crop_y_end, slices):

    def inTet(tetCoords,point):
        m = np.hstack([tetCoords, np.ones((4,1))])
        D0 = np.linalg.det(m)
        for i in range(4):
            n = m.copy()
            n[i,:3] = point
            D = np.linalg.det(n)
            if D != 0 and np.sign(D) != np.sign(D0):
                return False
        return True

    # Get volunteer string from data_path
    volunteer = os.path.basename( os.path.normpath( data_path ))

    header = imt_data.header

    pix_dim = header['pixdim']

    # Pixel spacing of images
    pixsp_x = pix_dim[1]
    pixsp_y = pix_dim[2]
    # Slice thickness
    pixsp_z = pix_dim[3]

    init_point = np.array([crop_x_in, crop_y_in, 0])
    factor = np.array([crop_x_end-crop_x_in, crop_y_end-crop_y_in, slices])

    vol_file = os.path.join( data_path, volunteer+'_vol.vtk')
    vol_mesh = meshio.read(vol_file)
    n_points = len(vol_mesh.points)

    #bg_file = os.path.join( data_path, volunteer+'_bg.vtk')
    bg_file = os.path.join( data_path, volunteer+'_surf_vol.vtk')
    bg_mesh = meshio.read(bg_file)
    cells = bg_mesh.cells[1][1]
    nodes = bg_mesh.points

    i = 0

    size = 200000

    candidate_points = init_point + lhs(3,size) * factor
    candidate_mesh = pixel_to_mesh(candidate_points, pixsp_x, pixsp_y, pixsp_z)
    bg_points = np.zeros((0,3))

    for point_mesh in candidate_mesh:
        
        dist_list = np.linalg.norm(nodes - point_mesh, axis= 1) 
        nearest_node_index = np.argmin(dist_list)

        nearest_tetra = np.where(cells == nearest_node_index)[0]

        boolean = np.array([])
        for index in nearest_tetra:
            tetra_coords = nodes[cells[index, :]]
            boolean = np.append(boolean, inTet(tetra_coords, point_mesh))
        
        if ~boolean.any():
            bg_points = np.vstack((bg_points, point_mesh))
            i+=1
        if i == n_points:
            break

    crop_str = str(crop_x_in)+'_'+str(crop_x_end)+'_'+str(crop_y_in)+'_'+str(crop_y_end)
    bg_points_file = os.path.join(data_path, 'background_points_'+crop_str+'.npy') 
    np.save( bg_points_file, bg_points)

def boolean_mask(data_path, imt_data, crop_x_in, crop_x_end, crop_y_in, crop_y_end, bool_mask_file):

    if os.path.exists(bool_mask_file):
        # Background nodes to real coordinates
        bool_mask = np.load( bool_mask_file )
    else:
        generate_boolean_mask(data_path, imt_data, crop_x_in, crop_x_end, crop_y_in, crop_y_end)
        bool_mask = np.load( bool_mask_file )

    return bool_mask

def generate_boolean_mask(data_path, imt_data, crop_x_in, crop_x_end, crop_y_in, crop_y_end):

    def inTet(tetCoords,point):
        m = np.hstack([tetCoords, np.ones((4,1))])
        D0 = np.linalg.det(m)
        for i in range(4):
            n = m.copy()
            n[i,:3] = point
            D = np.linalg.det(n)
            if D != 0 and np.sign(D) != np.sign(D0):
                return False
        return True

    # Get volunteer string from data_path
    volunteer = os.path.basename( os.path.normpath( data_path ))

    header = imt_data.header

    pix_dim = header['pixdim']

    # Pixel spacing of images
    pixsp_x = pix_dim[1]
    pixsp_y = pix_dim[2]
    # Slice thickness
    pixsp_z = pix_dim[3]

    dim = header['dim']

    Nx = dim[1]
    Ny = dim[2]
    Nz = dim[3]

    xp = np.arange(Nx)
    yp = np.arange(Ny)
    zp = np.arange(Nz)

    Y, Z, X = np.meshgrid(yp, zp, xp)

    X = X[:, crop_y_in:crop_y_end, crop_x_in:crop_x_end]
    Y = Y[:, crop_y_in:crop_y_end, crop_x_in:crop_x_end]
    Z = Z[:, crop_y_in:crop_y_end, crop_x_in:crop_x_end]

    Xr = X.ravel()[:,None]
    Yr = Y.ravel()[:,None]
    Zr = Z.ravel()[:,None]
    
    # Pixel coordinates in a list
    im_pix = np.hstack((Xr, Yr, Zr))
    im_mesh = pixel_to_mesh(im_pix, pixsp_x, pixsp_y, pixsp_z)

    #bg_file = os.path.join( data_path, volunteer+'_bg.vtk')
    bg_file = os.path.join( data_path, volunteer+'_surf_vol.vtk')
    bg_mesh = meshio.read(bg_file)
    cells = bg_mesh.cells[1][1]
    nodes = bg_mesh.points

    # 1 if the point is in the LV, 0 if not 
    boolean_mask = np.zeros(len(im_pix), dtype=bool) 

    #for point_mesh in im_mesh:
    for i in range(len(im_mesh)):

        point_mesh = im_mesh[i]

        dist_list = np.linalg.norm(nodes - point_mesh, axis= 1) 
        nearest_node_index = np.argmin(dist_list)

        nearest_tetra = np.where(cells == nearest_node_index)[0]

        boolean = np.array([])
        for index in nearest_tetra:
            tetra_coords = nodes[cells[index, :]]
            boolean = np.append(boolean, inTet(tetra_coords, point_mesh))
        
        if boolean.any():
            boolean_mask[i] = 1

    crop_str = str(crop_x_in)+'_'+str(crop_x_end)+'_'+str(crop_y_in)+'_'+str(crop_y_end)
    
    boolean_mask_file = os.path.join( data_path, 'boolean_mask_'+crop_str+'.npy')
    boolean_mask = np.reshape(boolean_mask, X.shape)
    np.save(boolean_mask_file, boolean_mask)

def pixel_to_mesh(pixel, px, py, pz):
    return pixel * np.array([-px, -py, pz])

def compare_images(ground_truth, prediction):
    
    mse_im = np.sqrt((np.square(ground_truth-prediction)).mean() / (np.square(ground_truth)).mean())
    ssim_im = ssim(ground_truth, prediction)

    print('Image | MSE: {0:.5e}, SSIM: {1:.5f}'.format(mse_im, ssim_im))

def read_lmks_mesh(file_name):
    """ Read Landmarks in mesh coordinates and 
    transform them to real coordinates for comparison in mm."""

    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(file_name)
    reader.ReadAllScalarsOn()
    reader.ReadAllVectorsOn()
    reader.Update()

    polydata = reader.GetOutput()
    points = polydata.GetPoints()
    array = points.GetData()
    lmks_mesh = vtk_to_numpy(array)

    return lmks_mesh

def mse_ssim(model_dir, frames, mse_pred_list, mse_temp_list, ssim_pred_list, ssim_temp_list):
    
    """
    Plot figures comparing reference image with template image with MSE and SSIM
    """

    fig_path = os.path.join( model_dir, 'Figures' )

    if not os.path.exists( fig_path ):
        os.makedirs( fig_path )

    fig, ax = plt.subplots(1, 2, figsize=(10,5))
    ax[0].set_title('MSE')
    ax[0].plot(np.arange(1, frames+1), mse_pred_list, label='Reference vs Warped Template')
    ax[0].plot(np.arange(1, frames+1), mse_temp_list, label='Reference vs Template')
    ax[0].set_xticks(np.arange(1, frames+1, 2))
    ax[0].set_xlabel('Frame')
    ax[0].legend()

    ax[1].set_title('SSIM')
    ax[1].plot(np.arange(1, frames+1), ssim_pred_list, label='Reference vs Warped Template')
    ax[1].plot(np.arange(1, frames+1), ssim_temp_list, label='Reference vs Template')
    ax[1].set_xticks(np.arange(1, frames+1, 2))
    ax[1].set_xlabel('Frame')
    ax[1].set_ylim([0.75,1])
    ax[1].legend()

    fig_name = 'MSE_SSIM.pdf'
    plt.savefig( os.path.join(fig_path, fig_name) )

def global_strain(data_path, model_dir, frames, pinn_strains):

    """
    pinn_strains shape: [9, frames]
    pinn_strains[0, :]: mean_rr_pinn
    pinn_strains[1, :]: median_rr_pinn
    pinn_strains[2, :]: std_rr_pinn 
    """
    # Strains given by PINN model
    mean_rr_pinn = pinn_strains[0, :]
    median_rr_pinn = pinn_strains[1, :]
    std_rr_pinn = pinn_strains[2, :] 

    mean_cc_pinn = pinn_strains[3, :]
    median_cc_pinn = pinn_strains[4, :]
    std_cc_pinn = pinn_strains[5, :] 

    mean_ll_pinn = pinn_strains[6, :]
    median_ll_pinn = pinn_strains[7, :]
    std_ll_pinn = pinn_strains[8, :] 

    #Strains given by INRIA
    mesh_inria_path = os.path.join(data_path, 'MESH_INRIA')

    mean_rr_inria = np.zeros([frames])
    median_rr_inria = np.zeros([frames])
    std_rr_inria = np.zeros([frames])

    mean_cc_inria = np.zeros([frames])
    median_cc_inria = np.zeros([frames])
    std_cc_inria = np.zeros([frames])

    mean_ll_inria = np.zeros([frames])
    median_ll_inria = np.zeros([frames])
    std_ll_inria = np.zeros([frames])

    # Strains given by UPF
    mesh_upf_path = os.path.join(data_path, 'MESH_UPF')

    mean_rr_upf = np.zeros([frames])
    median_rr_upf = np.zeros([frames])
    std_rr_upf = np.zeros([frames])

    mean_cc_upf = np.zeros([frames])
    median_cc_upf = np.zeros([frames])
    std_cc_upf = np.zeros([frames])

    mean_ll_upf = np.zeros([frames])
    median_ll_upf = np.zeros([frames])
    std_ll_upf = np.zeros([frames])

    # Calculate mean, median and str for INRIA and UPF
    for i in range(0, frames):
        
        strain_inria = pyvista.PolyData( os.path.join(mesh_inria_path, 'finalMesh0{:02d}.vtk'.format(i)))
        
        E_rr = strain_inria.point_arrays['radStrain']
        E_cc = strain_inria.point_arrays['circStrain']
        E_ll = strain_inria.point_arrays['longStrain']

        mean_rr_inria[i] = np.mean(E_rr)
        median_rr_inria[i] = np.median(E_rr)
        std_rr_inria[i] = np.std(E_rr)

        mean_cc_inria[i] = np.mean(E_cc)
        median_cc_inria[i] = np.median(E_cc)
        std_cc_inria[i] = np.std(E_cc)

        mean_ll_inria[i] = np.mean(E_ll)
        median_ll_inria[i] = np.median(E_ll)
        std_ll_inria[i] = np.std(E_ll)

        strain_upf = pyvista.PolyData( os.path.join(mesh_upf_path, 'finalMesh0{:02d}.vtk'.format(i)))
        
        E_rr = strain_upf.point_arrays['radStrain']
        E_cc = strain_upf.point_arrays['circStrain']
        E_ll = strain_upf.point_arrays['longStrain']

        mean_rr_upf[i] = np.mean(E_rr)
        median_rr_upf[i] = np.median(E_rr)
        std_rr_upf[i] = np.std(E_rr)

        mean_cc_upf[i] = np.mean(E_cc)
        median_cc_upf[i] = np.median(E_cc)
        std_cc_upf[i] = np.std(E_cc)

        mean_ll_upf[i] = np.mean(E_ll)
        median_ll_upf[i] = np.median(E_ll)
        std_ll_upf[i] = np.std(E_ll)

    #########################
    volunteer = os.path.basename( os.path.normpath( data_path ))

    # Plot for mean strains
    fig, axes = plt.subplots(1,3, figsize=(12,5))

    fig.suptitle('Radial, Circumferential and Longitudinal Mean Strains. Volunteer '+volunteer, fontsize=16)
    # Radial Strain

    # Inria
    axes[0].fill_between(np.linspace(0,1,frames), mean_rr_inria+std_rr_inria, mean_rr_inria-std_rr_inria, alpha=.5)
    axes[0].plot(np.linspace(0,1,frames), mean_rr_inria, label='INRIA')
    # UPF
    axes[0].fill_between(np.linspace(0,1,frames), mean_rr_upf+std_rr_upf, mean_rr_upf-std_rr_upf, alpha=.5)
    axes[0].plot(np.linspace(0,1,frames), mean_rr_upf, label='UPF')
    # PINN
    axes[0].fill_between(np.linspace(0,1,frames), mean_rr_pinn+std_rr_pinn, mean_rr_pinn-std_rr_pinn, alpha=.5)
    axes[0].plot(np.linspace(0,1,frames), mean_rr_pinn, label='PINN')

    axes[0].set_xticks(np.linspace(0,1,3))
    axes[0].set_title('Radial Strain. Mean + Std')
    axes[0].legend()

    # Circumferential strain

    # Inria
    axes[1].fill_between(np.linspace(0,1,frames), mean_cc_inria+std_cc_inria, mean_cc_inria-std_cc_inria, alpha=.5)
    axes[1].plot(np.linspace(0,1,frames), mean_cc_inria, label='INRIA')
    # UPF
    axes[1].fill_between(np.linspace(0,1,frames), mean_cc_upf+std_cc_upf, mean_cc_upf-std_cc_upf, alpha=.5)
    axes[1].plot(np.linspace(0,1,frames), mean_cc_upf, label='UPF')
    # PINN
    axes[1].fill_between(np.linspace(0,1,frames), mean_cc_pinn+std_cc_pinn, mean_cc_pinn-std_cc_pinn, alpha=.5)
    axes[1].plot(np.linspace(0,1,frames), mean_cc_pinn, label='PINN')

    axes[1].set_xticks(np.linspace(0,1,3))
    axes[1].set_title('Circumferential Strain. Mean + Std')
    axes[1].legend()

    # Longitudinal strain

    #Inria
    axes[2].fill_between(np.linspace(0,1,frames), mean_ll_inria+std_ll_inria, mean_ll_inria-std_ll_inria, alpha=.5)
    axes[2].plot(np.linspace(0,1,frames), mean_ll_inria, label='INRIA')
    #UPF
    axes[2].fill_between(np.linspace(0,1,frames), mean_ll_upf+std_ll_upf, mean_ll_upf-std_ll_upf, alpha=.5)
    axes[2].plot(np.linspace(0,1,frames), mean_ll_upf, label='UPF')
    #PINN
    axes[2].fill_between(np.linspace(0,1,frames), mean_ll_pinn+std_ll_pinn, mean_ll_pinn-std_ll_pinn, alpha=.5)
    axes[2].plot(np.linspace(0,1,frames), mean_ll_pinn, label='PINN')

    axes[2].set_xticks(np.linspace(0,1,3))
    axes[2].set_title('Longitudinal Strain. Mean + Std')
    axes[2].legend()

    fig_path = os.path.join( model_dir, 'Figures' )
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)

    fig_name = 'strains_mean_std.pdf'
    plt.savefig( os.path.join(fig_path, fig_name) )

    #########################
    # Plot for median strains
    fig, axes = plt.subplots(1,3, figsize=(12,5))

    fig.suptitle('Radial, Circumferential and Longitudinal Median Strains. Volunteer '+volunteer, fontsize=16)
    # Radial Strain

    # Inria
    axes[0].fill_between(np.linspace(0,1,frames), median_rr_inria+std_rr_inria, median_rr_inria-std_rr_inria, alpha=.5)
    axes[0].plot(np.linspace(0,1,frames), median_rr_inria, label='INRIA')
    # UPF
    axes[0].fill_between(np.linspace(0,1,frames), median_rr_upf+std_rr_upf, median_rr_upf-std_rr_upf, alpha=.5)
    axes[0].plot(np.linspace(0,1,frames), median_rr_upf, label='UPF')
    # PINN
    axes[0].fill_between(np.linspace(0,1,frames), median_rr_pinn+std_rr_pinn, median_rr_pinn-std_rr_pinn, alpha=.5)
    axes[0].plot(np.linspace(0,1,frames), median_rr_pinn, label='PINN')

    axes[0].set_xticks(np.linspace(0,1,3))
    axes[0].set_title('Radial Strain. Median + Std')
    axes[0].legend()

    # Circumferential strain

    # Inria
    axes[1].fill_between(np.linspace(0,1,frames), median_cc_inria+std_cc_inria, median_cc_inria-std_cc_inria, alpha=.5)
    axes[1].plot(np.linspace(0,1,frames), median_cc_inria, label='INRIA')
    # UPF
    axes[1].fill_between(np.linspace(0,1,frames), median_cc_upf+std_cc_upf, median_cc_upf-std_cc_upf, alpha=.5)
    axes[1].plot(np.linspace(0,1,frames), median_cc_upf, label='UPF')
    # PINN
    axes[1].fill_between(np.linspace(0,1,frames), median_cc_pinn+std_cc_pinn, median_cc_pinn-std_cc_pinn, alpha=.5)
    axes[1].plot(np.linspace(0,1,frames), median_cc_pinn, label='PINN')

    axes[1].set_xticks(np.linspace(0,1,3))
    axes[1].set_title('Circumferential Strain. Median + Std')
    axes[1].legend()

    # Longitudinal strain

    #Inria
    axes[2].fill_between(np.linspace(0,1,frames), median_ll_inria+std_ll_inria, median_ll_inria-std_ll_inria, alpha=.5)
    axes[2].plot(np.linspace(0,1,frames), median_ll_inria, label='INRIA')
    #UPF
    axes[2].fill_between(np.linspace(0,1,frames), median_ll_upf+std_ll_upf, median_ll_upf-std_ll_upf, alpha=.5)
    axes[2].plot(np.linspace(0,1,frames), median_ll_upf, label='UPF')
    #PINN
    axes[2].fill_between(np.linspace(0,1,frames), median_ll_pinn+std_ll_pinn, median_ll_pinn-std_ll_pinn, alpha=.5)
    axes[2].plot(np.linspace(0,1,frames), median_ll_pinn, label='PINN')

    axes[2].set_xticks(np.linspace(0,1,3))
    axes[2].set_title('Longitudinal Strain. Median + Std')
    axes[2].legend()

    fig_path = os.path.join( model_dir, 'Figures' )
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)

    fig_name = 'strains_median_std.pdf'
    plt.savefig( os.path.join(fig_path, fig_name) )
    plt.close()

def read_lmks(file_name, px, py, pz, A, b):
    """ Read Landmarks in mesh coordinates and 
    transform them to real coordinates for comparison in mm."""

    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(file_name)
    reader.ReadAllScalarsOn()
    reader.ReadAllVectorsOn()
    reader.Update()

    polydata = reader.GetOutput()
    points = polydata.GetPoints()
    array = points.GetData()
    lmks_mesh = vtk_to_numpy(array)

    lmks_pixel = mesh_to_pixel(lmks_mesh, px, py, pz)
    lmks_real = pixel_to_real( lmks_pixel, A, b)

    return lmks_real

def box_plots(data_path, model_dir):

    volunteers = ["v1","v2","v4","v5","v6","v7","v8","v9","v10","v11","v12","v13","v14","v15","v16"]
    tag_ff = ["22","28","25","22","22","30","30","29","26","31","23","37","28","20","24"]
    ssfp_ff = ["29","29","29","29","29","29","29","29","29","29","29","29","29","29","29"]
    tag_es = ["10","10","10","10","10","11","10","10","10","11","10","10","11","08","09"]
    ssfp_es = ["10","11","11","11","11","09","09","10","10","09","11","09","10","11","11"]

    im_dicom_list = ['IM_0048', 'IM_0098', 'IM_0237', 'IM_1099', 'IM_1455', 'IM_1279', 'IM_0415', 'IM_0066', 'IM_1597', 'IM_0561', 'IM_0066', 'IM_0066', 'IM_0098', 'IM_1506', 'IM_1460']

    # Get volunteer string from data_path
    volunteer = os.path.basename( os.path.normpath( data_path ))
    i = np.argwhere(np.array(volunteers)==volunteer)[0][0]

    if volunteer == 'v2':
        t1 = nib.load(os.path.join(data_path, 't01.nii.gz'))
        header = t1.header

        pix_dim = header['pixdim']
        # Pixel spacing of images
        pixsp_x = pix_dim[1]
        pixsp_y = pix_dim[2]
        # Slice thickness
        pixsp_z = pix_dim[3]

        A = t1.affine[:-1,:-1].astype(np.float32)
        b = t1.affine[:-1,-1].astype(np.float32)

    else:

        #im_path = os.path.join('MRI_images', 'nifti_images_2', volunteer)
        im_path = os.path.join('nifti_images', volunteer)
        #data_path = os.path.join('MRI_data', volunteer)
        imt_path = os.path.join(im_path,'imt.nii.gz')
        imt_data_dcm = nib.load( imt_path )

        # Pixels
        header = imt_data_dcm.header

        pix_dim = header['pixdim']
        # Pixel spacing of images
        pixsp_x = pix_dim[1]
        pixsp_y = pix_dim[2]
        # Slice thickness
        pixsp_z = pix_dim[3]

        # Get matrix A
        A = imt_data_dcm.affine[:-1,:-1] * np.array([-1, -1, 1])

        # Get vector b
        im_dicom = im_dicom_list[i]

        dicom_path = os.path.join('dicom_images', volunteer, 'cSAX', 'time_1', im_dicom)
        #dicom_path = os.path.join('MRI_images', 'dicom_images_im1', volunteer, 'time_1', im_dicom)
        #dicom_path = os.path.join('MRI_data/dicom_images', volunteer, 'time_1', im_dicom)
        ds = pydicom.dcmread(dicom_path)

        b = ds.ImagePositionPatient * np.array([-1, -1, 1])

    # Ground truth (using INRIA COORDINATES)
    gt_inria_lmks_path = os.path.join(data_path, 'LMKS_GT/INRIA_COORDINATES')

    # Final Frame
    file_name = os.path.join(gt_inria_lmks_path, 'obs1_groundTruth{0}.vtk'.format(tag_ff[i]))
    lmks_gt_obs1_ff_i = read_lmks(file_name, pixsp_x, pixsp_y, pixsp_z, A, b)

    file_name = os.path.join(gt_inria_lmks_path, 'obs2_groundTruth{0}.vtk'.format(tag_ff[i]))
    lmks_gt_obs2_ff_i = read_lmks(file_name, pixsp_x, pixsp_y, pixsp_z, A, b)

    # ES Frame
    file_name = os.path.join(gt_inria_lmks_path, 'obs1_groundTruth{0}.vtk'.format(tag_es[i]))
    lmks_gt_obs1_es_i = read_lmks(file_name, pixsp_x, pixsp_y, pixsp_z, A, b)

    file_name = os.path.join(gt_inria_lmks_path, 'obs2_groundTruth{0}.vtk'.format(tag_es[i]))
    lmks_gt_obs2_es_i = read_lmks(file_name, pixsp_x, pixsp_y, pixsp_z, A, b)

    # Observer 1
    file_name = os.path.join(model_dir, 'lmks', 'lmks_results{0}.vtk'.format(int(ssfp_ff[i])+1))
    lmks_model = meshio.read(file_name)
    lmks_pinn_ff = mesh_to_real(lmks_model.points, pixsp_x, pixsp_y, pixsp_z, A, b)

    file_name = os.path.join(model_dir, 'lmks', 'lmks_results{0}.vtk'.format(int(ssfp_es[i])+1))
    lmks_model = meshio.read(file_name)
    lmks_pinn_es = mesh_to_real(lmks_model.points, pixsp_x, pixsp_y, pixsp_z, A, b)

    dist_gt_pinn_ff_i = np.linalg.norm(lmks_gt_obs1_ff_i - lmks_pinn_ff, axis=1)
    dist_gt_pinn_es_i = np.linalg.norm(lmks_gt_obs1_es_i - lmks_pinn_es, axis=1)

    dist_gt_pinn_i = np.reshape( np.stack((dist_gt_pinn_ff_i, dist_gt_pinn_es_i,\
                dist_gt_pinn_ff_i, dist_gt_pinn_es_i)), [-1])

    dist_gt_inria_i = np.load( os.path.join(data_path, 'lmks_gt_inria.npy') )

    dist_gt_upf_i = np.load( os.path.join(data_path, 'lmks_gt_upf.npy') )

    dist_gt_deepstr_i = np.load( os.path.join(data_path, 'lmks_deep_strain.npy') )

    fig_path = os.path.join( model_dir, 'Figures' )
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)

    # Box plot without outliers
    plt.figure()
    plt.boxplot(np.stack( (dist_gt_upf_i, dist_gt_inria_i, dist_gt_deepstr_i, dist_gt_pinn_i) ).T, labels = ['UPF', 'INRIA', 'CarMEN', 'PINN'], whis = 'range' )
    plt.title('Box plot for volunteer '+volunteer)
    plt.ylabel('Euclidian distance (mm)')
    plt.yticks(np.arange(0,16))
    plt.axhline( np.median(dist_gt_pinn_i), linestyle = '--' )
    fig_name = 'box_plot_'+volunteer+'.pdf'
    plt.savefig( os.path.join(fig_path, fig_name) )

    # Box plot with outliers
    plt.figure()
    plt.boxplot(np.stack( (dist_gt_upf_i, dist_gt_inria_i, dist_gt_deepstr_i, dist_gt_pinn_i) ).T, labels = ['UPF', 'INRIA', 'CarMEN','PINN'])

    plt.title('Box plot for volunteer '+volunteer)
    plt.ylabel('Euclidian distance (mm)')
    plt.yticks(np.arange(0,16))
    #plt.yticks(np.arange(0, np.max((dist_gt_upf_i, dist_gt_inria_i, dist_gt_pinn_i)) ))
    #plt.ylim([0, 16])
    plt.axhline( np.median(dist_gt_pinn_i), linestyle = '--' )
    fig_name = 'box_plot_outl_'+volunteer+'.pdf'
    plt.savefig( os.path.join(fig_path, fig_name) )

    np.save( os.path.join(data_path, 'lmks_gt_pinn.npy'), dist_gt_pinn_i)
############################

def mesh_to_pixel(surf_mesh, px, py, pz):
    return surf_mesh * np.array([-1/px, -1/py, 1/pz])

def pixel_to_real(pixel, A, b):
    return (np.matmul(pixel, A.T) + b).astype(np.float32)

def real_to_pixel(real, A_inv, b):
    return (np.matmul(real - b, A_inv.T)).astype(np.float32)

def numpy_to_nifti(im, nifti_name, affine):
    im_ni = nib.Nifti1Image(np.transpose(im, [2,1,0]), affine)
    nib.save(im_ni, nifti_name)

def plot_image(X, Y, Z, path_fig, image_pred, title, name, levels, vmin, vmax, slices, save=True):
    fig, axes = plt.subplots(len(slices), figsize=(15,10))

    imz = axes[0].contourf(X[slices[0],:,:], Y[slices[0],:,:], image_pred[slices[0],:,:], levels=levels, origin='lower', extent=[0,1,0,1])
    axes[0].set_title(title[0]+'. Slice z={0:.3f}'.format(slices[0]))
    axes[0].set_box_aspect(1)

    imy = axes[1].contourf(X[:,slices[1],:], Z[:,slices[1],:], image_pred[:,slices[1],:], levels=levels, origin='lower', extent=[0,1,0,1])
    axes[1].set_title(title[0]+'. Slice y={0:.3f}'.format(slices[1]))
    axes[1].set_box_aspect(1)

    imx = axes[2].contourf(Y[:,:,slices[2]], Z[:,:,slices[2]], image_pred[:,:,slices[2]], levels=levels, origin='lower', extent=[0,1,0,1])
    axes[2].set_title(title[0]+'. Slice x={0:.3f}'.format(slices[2]))
    axes[2].set_box_aspect(1)

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.03, 0.7])
    imx.set_clim(vmin, vmax)
    imy.set_clim(vmin, vmax)
    imz.set_clim(vmin, vmax)
    fig.colorbar(imx, cax=cbar_ax)

    #plt.tight_layout()
    if save: 
        path_save = os.path.join(path_fig, name)
        plt.savefig(path_save, bbox_inches='tight')

def read_upf_lmks(file_name, px, py, pz, A, b):
    """ Read Landmarks in mesh coordinates and 
    transform them to real coordinates for comparison in mm."""

    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(file_name)
    reader.ReadAllScalarsOn()
    reader.ReadAllVectorsOn()
    reader.Update()

    polydata = reader.GetOutput()
    points = polydata.GetPoints()
    array = points.GetData()
    lmks_mesh = vtk_to_numpy(array) * np.array([-1,-1,1]) + b * np.array([-1, -1, -1])

    lmks_pixel = mesh_to_pixel(lmks_mesh, px, py, pz)
    lmks_real = pixel_to_real( lmks_pixel, A, b)
    #lmks_real = pixel_to_real( lmks_mesh, A, b)

    return lmks_real

def mesh_to_real(points_mesh, px, py, pz, A, b):
    points_pixel = mesh_to_pixel(points_mesh, px, py, pz)
    points_real = pixel_to_real(points_pixel, A, b)

    return points_real

def deform_np(im, X, Y, Z, u1, u2, u3, crop_x_in, crop_y_in):
    depth = im.shape[0]
    height = im.shape[1]
    width = im.shape[2]

    X_new = (X - crop_x_in + u1)# * (width-1)
    Y_new = (Y - crop_y_in + u2)# * (height-1)
    Z_new = (Z + u3)# * (depth-1)

    #indices
    X0 = np.floor(X_new)
    X1 = X0 + 1
    Y0 = np.floor(Y_new)
    Y1 = Y0 + 1
    Z0 = np.floor(Z_new)
    Z1 = Z0 + 1

    X0 = np.clip(X0, 0, width - 1)
    X1 = np.clip(X1, 0, width - 1)
    Y0 = np.clip(Y0, 0, height - 1)
    Y1 = np.clip(Y1, 0, height - 1) 
    Z0 = np.clip(Z0, 0, depth - 1)
    Z1 = np.clip(Z1, 0, depth - 1) 

    Xd = (X_new - X0).ravel()
    Yd = (Y_new - Y0).ravel()
    Zd = (Z_new - Z0).ravel()

    Xdd = (X1 - X_new).ravel()
    Ydd = (Y1 - Y_new).ravel()
    Zdd = (Z1 - Z_new).ravel()

    X0 = X0.astype(int)
    X1 = X1.astype(int)
    Y0 = Y0.astype(int)
    Y1 = Y1.astype(int)
    Z0 = Z0.astype(int)
    Z1 = Z1.astype(int)  

    i000 = X0.ravel() + width*Y0.ravel() + width*height*Z0.ravel() 
    i100 = X1.ravel() + width*Y0.ravel() + width*height*Z0.ravel() 
    i001 = X0.ravel() + width*Y0.ravel() + width*height*Z1.ravel() 
    i101 = X1.ravel() + width*Y0.ravel() + width*height*Z1.ravel() 
    i010 = X0.ravel() + width*Y1.ravel() + width*height*Z0.ravel() 
    i110 = X1.ravel() + width*Y1.ravel() + width*height*Z0.ravel() 
    i011 = X0.ravel() + width*Y1.ravel() + width*height*Z1.ravel() 
    i111 = X1.ravel() + width*Y1.ravel() + width*height*Z1.ravel() 

    imr = im.ravel()

    im00 = imr[i000]*Xdd + imr[i100]*Xd
    im10 = imr[i010]*Xdd + imr[i110]*Xd
    im01 = imr[i001]*Xdd + imr[i101]*Xd
    im11 = imr[i011]*Xdd + imr[i111]*Xd

    im0 = im00*Zdd + im01*Zd
    im1 = im10*Zdd + im11*Zd

    return (im0*Ydd+im1*Yd).reshape(depth, height, width)

def im_and_segm_real(t1, crop_x_in, crop_x_end, crop_y_in, crop_y_end, data_path):
    # Header of images to get information
    header = t1.header

    dim = header['dim']

    Nx = dim[1]
    Ny = dim[2]
    Nz = dim[3]

    xp = np.arange(Nx)
    yp = np.arange(Ny)
    zp = np.arange(Nz)

    Y, Z, X = np.meshgrid(yp, zp, xp)

    X = X[:, crop_y_in:crop_y_end, crop_x_in:crop_x_end]
    Y = Y[:, crop_y_in:crop_y_end, crop_x_in:crop_x_end]
    Z = Z[:, crop_y_in:crop_y_end, crop_x_in:crop_x_end]

    X = X.ravel()[:,None]
    Y = Y.ravel()[:,None]
    Z = Z.ravel()[:,None]
    # Pixel coordinates in a list
    im_pix = np.hstack((X, Y, Z))

    # Affine transformation from pixel coords to real coords. Ax+b

    A = t1.affine[:-1,:-1].astype(np.float32)
    b = t1.affine[:-1,-1].astype(np.float32)

    pix_dim = header['pixdim']

    # Pixel spacing of images
    pixsp_x = pix_dim[1]
    pixsp_y = pix_dim[2]
    # Slice thickness
    pixsp_z = pix_dim[3]

    im_real = pixel_to_real(im_pix, A, b)

    # Nodes in volumetric mesh of left ventricle 
    vol_mesh = meshio.read( os.path.join(data_path, 'v1_vol.vtk') )

    # Segmentation mesh to real coordinates
    segm_mesh = vol_mesh.points
    segm_pix = mesh_to_pixel(segm_mesh, pixsp_x, pixsp_y, pixsp_z)
    segm_real = pixel_to_real(segm_pix, A, b)

    return im_real, segm_real

def background_real(volunteer, t1, crop_x_in, crop_x_end, crop_y_in, crop_y_end, bg_mesh_file):
    
    header = t1.header

    A = t1.affine[:-1,:-1].astype(np.float32)
    b = t1.affine[:-1,-1].astype(np.float32)

    pix_dim = header['pixdim']

    # Pixel spacing of images
    pixsp_x = pix_dim[1]
    pixsp_y = pix_dim[2]
    # Slice thickness
    pixsp_z = pix_dim[3]

    if os.path.exists(bg_mesh_file):
        # Background nodes to real coordinates
        bg_mesh = np.load( bg_mesh_file )
    else:
        generate_background_mesh(volunteer, t1, crop_x_in, crop_x_end, crop_y_in, crop_y_end)
        bg_mesh = np.load( bg_mesh_file )

    bg_pixel = mesh_to_pixel(bg_mesh, pixsp_x, pixsp_y, pixsp_z)
    bg_real = pixel_to_real(bg_pixel, A, b)

    return bg_real

