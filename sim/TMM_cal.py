import numpy as np
import pandas as pd
from tmm_fast.vectorized_tmm_dispersive_multistack import coh_vec_tmm_disp_mstack as tmm


def TMM_cal(bit_string):
    qv_ii = bit_string
    num_qubits = np.size(qv_ii)
    # simulation parameters (wavelength, incident angle, number of layers, layer thickness)
    wl = np.linspace(300,2500,2201) * (10**(-9))       # wavelength, unit (nm)
    theta = np.linspace(0,0,1) * (np.pi/180)           # incident angle, unit (deg)
    mode = 'T'
    num_layers = int(num_qubits/2)
    layer_thickness = 50
    num_stacks = 1                   # default
    total_layers = num_layers + 2    # default
    upper_medium = 1.4     # upper medium, air = 0, PDMS = 1.45
    lower_medium = 1.45    # substrate, SiO2 = 1.45
    
    # define ideal state
    solar_spectrum = pd.read_csv('sim/solar_spectrum.txt', sep=',', header=None)
    solar_spectrum = solar_spectrum.values.astype(np.float32)
    ideal_T = np.zeros((1,np.size(wl)))
    ideal_T[0, 0:400-300] = 0
    ideal_T[0, 400-300:750-300+1] = 1
    ideal_T[0, 750-300+1:] = 0
    ideal_trans_energy = np.multiply(ideal_T, solar_spectrum)
    
    # thickness - preset
    thickness = np.ones((1,total_layers))
    for i in range(1,total_layers):
        thickness[:,i] = layer_thickness
    thickness[:,0] = np.inf
    thickness[:,-1] = np.inf
    thickness = thickness * (10**(-9))  # unit (nm)
    thickness = np.array(thickness)
    
    # Call refractive indices of dielectric materials from saved data (must align with wavelength range)
    dielectric_ref = pd.read_csv('sim/dielectric_ref.txt', sep=',', header=None)
    dielectric_ref = dielectric_ref.values.astype(np.float32)  # 0: SiO2, 1: Si3N4, 2: Al2O3, 3: TiO2
    
    mater_index = np.zeros((num_layers+1, np.size(wl)))  # first layer is an upper layer (e.g., air or PDMS)
    for nl in range(0,num_layers):
        if qv_ii[nl*2] == 0:
            if qv_ii[nl*2+1] == 0:
                mater_index[nl+1] = dielectric_ref[0,:]  # '00' = SiO2
            else:
                mater_index[nl+1] = dielectric_ref[1,:]  # '01' = Si3N4
        elif qv_ii[nl*2] == 1:
            if qv_ii[nl*2+1] == 0:
                mater_index[nl+1] = dielectric_ref[2,:]  # '10' = Al2O3
            else:
                mater_index[nl+1] = dielectric_ref[3,:]  # '11' = TiO2
    
    # create layer configurations
    # regractive index
    M = np.ones((num_stacks, total_layers, wl.shape[0]))
    for i in range(1, M.shape[1]-1):
        M[:,0] = upper_medium     # upper medium, air = 0, PDMS = 1.45
        M[:,i] = mater_index[i]
        M[:,-1] = lower_medium    # substrate, SiO2 = 1.45
    
    # TMM simulation
    results = tmm('p', M, thickness, theta, wl, device='cpu', timer=True)  # polarization = 's' or 'p', device = 'cpu' or 'cuda'
    
    # extract optical characteristics
    x = wl * (10**9)
    T = results[0]['T']
    T_0 = T[0][0]  # at 0th angle
    R = results[0]['R']
    R_0 = R[0][0]  # at 0th angle
    
    # FOM definition, save FOM & qv_ii
    trans_energy = np.multiply(T[0], solar_spectrum)
    FOM_cal = np.sum(np.square(np.subtract(trans_energy,ideal_trans_energy)))
    FOM = FOM_cal/np.sum(np.square(solar_spectrum))*10
    FOM = FOM.reshape(1,1)
    
    return FOM

