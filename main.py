#%%
%reload_ext autoreload
%autoreload 2

import os
import json
import numpy as np
from datetime import datetime 
from matplotlib import colors
import matplotlib.pyplot as plt
from joblib import dump, load
import cupy as cp

from GeneratePSD import GeneratePSD
from GeneratePhase import *

# Loading influence functions

# Open json file
with open('settings.json') as f:
    settings = json.load(f)
    
for i in settings.keys():
    settings[i] = os.path.normpath(settings[i])
    
#%%
try:
    base = settings['base_path']

    choice = 0

    if choice == 0:
        # Boosted HO
        path_IFs   = os.path.normpath(os.path.join(base,'NFM_CM_files_181121/'))
        path_eigen = os.path.normpath(os.path.join(base,'NFM_CM_files_181121/'))
        path_root  = os.path.normpath(os.path.join(base,'LTAO_PSD_HarnessData/0026_HOCM_newref_3_HO_KI_-0.4/'))
        CMLT_paths    = [ os.path.join(base, 'LTAO_PSD_HarnessData/ReferenceCMs', 'LTAO_CM'+str(i)+'_Ref_4Lay1300_nt100_np2p5.fits') for i in range(1,5) ]
        CM_path = CMLT_paths

    elif choice == 1:
        # Boosted HO + misregistration
        path_root  = os.path.normpath(os.path.join(base,'LTAO_PSD_HarnessData/0026_HOCM_newref_3_HO_KI_-0.4/'))
        path_eigen = os.path.normpath(os.path.join(base,'NFM_CM_files_181121/'))
        path_IFs   = os.path.normpath(os.path.join(base,'NFM_CM_files_181121/'))
        CMLTmis_paths  = [ os.path.join(base, 'LTAO_PSD_HarnessData/ReferenceCMs', 'LTAO_CM'+str(i)+'_Ref_4Lay1300_nt100_np2p5_0misreg.fits') for i in range(1,5) ]
        CM_path = CMLTmis_paths

    else:#elif choice == 2:
        # GLAO
        path_root  = os.path.normpath(os.path.join(base,'LTAO_PSD_HarnessData/Day1_nMaxModes/0020_HOCM_1_HO_KI_-0.4/'))
        CMGLAO_paths  = [ os.path.join(base, 'LTAO_PSD_HarnessData/ReferenceCMs', 'LTAO_CM'+str(i)+'_GLAO_E75_nt300.fits') for i in range(1,5) ]
        CM_path = CMGLAO_paths

    IF_entries = {
        'eigmod': 'eigenmodes',
        'eigval': 'eigenvalues',
        'usdact': 'UsedActMap'
    }

    PSD_generator = GeneratePSD(path_root, path_IFs, path_eigen, CM_path)
    if choice == 0 or choice == 1 or choice == 2:
        PSD_generator.IF_entries = IF_entries

    #PSD_generator.exclude_modes = list(set([0,1,2])) #,33,34,56,57]))

    PSD = PSD_generator.ComputePSD()
    KL_pupil = PSD_generator.pupil

    path_save = os.path.join(settings['LIFT_path'], 'AOF_PSD.pkl')
    dump(PSD, path_save, compress='gzip')
    
    path_save = os.path.join(settings['LIFT_path'], 'KL_pupil.pkl')
    dump(KL_pupil, path_save, compress='gzip')
    
except:
    path_load = os.path.join(settings['LIFT_path'],  'AOF_PSD.pkl')
    PSD = load(path_load)
    path_load = os.path.join(settings['LIFT_path'],  'KL_pupil.pkl')
    KL_pupil = load(path_load)
    

plt.figure(dpi=200)
PSD_size = PSD.shape[0]
crop = 44
zoomed = slice(PSD_size//2-crop,PSD_size//2+crop)
zoomed = (zoomed,zoomed)
PSD_cropped = PSD[zoomed]
plt.imshow(np.log(PSD_cropped)) #, norm=norm)
plt.show()


#%%
seeing = 1.5 #0.35

PSD_total, PSD_vK = HO_PSD_vs_seeing(PSD, seeing, return_vK=True)

def binning(inp, N, regime='sum'):
    if N == 1:
        return inp
    
    xp = cp.get_array_module(inp)
    out = xp.stack(xp.split(xp.stack(xp.split(xp.atleast_3d(inp), inp.shape[0]//N, axis=0)), inp.shape[1]//N, axis=2))
    if    regime == 'max':  func = xp.max
    elif  regime == 'min':  func = xp.min
    elif  regime == 'mean': func = xp.mean
    else: func = xp.sum
        
    return xp.squeeze( xp.transpose( func(out, axis=(2,3), keepdims=True), axes=(1,0,2,3,4)) )


bin_factor = 4

PSD_total = binning(PSD_total, bin_factor, regime='mean') / bin_factor**4
pupil = binning(cp.array(KL_pupil), bin_factor, regime='max')

phases = PSDrealizationBatchGPU(cp.array(PSD_total), batch_size=100, return_CPU=False)

WFE_expected = HO_WFE_vs_seeing(seeing)


phases = remove_piston_and_TT(phases, pupil)
STDs = np.sqrt( np.sum(phases**2, axis=(0,1)) / np.sum(pupil) )
STDs.mean()

print(f'Generated WFE: {STDs.mean()}, expected WFE: {WFE_expected}')

plt.imshow(phases[...,0].get())
plt.colorbar()
plt.show()


#%%
SPARTAunit2Microns = 35 # 35 microns per SPARTA unit
Surface2WF = 2 # multiplicative factor to go from surface to WF
A = SPARTAunit2Microns * Surface2WF

dk = 1.0/8.0 # PSD spatial frequency step [m^-1]
pitch = 0.2 # [m]
kc = 1/(2*pitch)

def radial_profile(data, center):
    y, x = np.indices((data.shape))
    r = np.sqrt( (x-center[0])**2 + (y-center[1])**2 )
    r = r.astype('int')

    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr
    return radialprofile 


full_profile = radial_profile(PSD_total, (240//2, 240//2))[:240//2]
vK_profile   = radial_profile(PSD_vK, (240//2, 240//2))[:240//2]
k = np.arange(240//2)*dk

plt.figure(dpi=200)
plt.plot(k, vK_profile, label='Von Karman PSD', linestyle='--', color='gray')
plt.plot(k, full_profile, label='Telemetry PSD + synth von Karman')
plt.axvline(x=kc, color='black', linestyle='--', label='Cut-off freq.')
plt.xlim([k[1], k.max()])
plt.title('Telemetry PSD')
plt.yscale('log')
plt.xscale('log')
plt.xlabel(r'Spatial frequency, $m^{-1}$')
plt.legend()
plt.grid()
plt.show()

'''
phase_test_batch1 = PSDrealizationBatchGPU(PSD_total)
phase_test_batch2 = PSDrealizationBatchGPU(PSD_total)
phase_test_batch  = np.dstack([phase_test_batch1, phase_test_batch2]) # total 1000 phase screen generated

import pickle
with open(settings['data_path']+'IRLOS/Synthetic/LIFT_dataset/phase_screens.pickle', 'wb') as handle:
    pickle.dump(phase_test_batch, handle, protocol=pickle.HIGHEST_PROTOCOL)

phase_test = phase_test_batch[:,:,22+26]
pupil = PSD_generator.pupil
print(phase_test[np.where(pupil>0.0)].std())
plt.imshow(phase_test*pupil)
'''
#%%

filename = settings['data_path']+'LTAO_PSD_HarnessData/Day1_nMaxModes/0020_HOCM_1_HO_KI_-0.4/0020_HOCM_1_HO_KI_-0.4.txt'

def ReadObservationFile(filename):
    with open(filename) as f:
        lines = f.readlines()
    file_data = {}
    for i,line in enumerate(lines):
        if line.find('timestamp') == -1:
            entry, value = line.replace('\n','').split(':')
            value = value.split(' ')[1:]
            value = np.array([float(val) for val in value])
            file_data[entry] = value
            if len(value) == 1:
                file_data[entry] = value[0]
        else:
            entry, value = line.replace('\n','').split(' ')
            file_data[entry] = datetime.fromisoformat(value)
    return file_data

data_dict = ReadObservationFile(filename)

#%%
wvl = 5e-7 #[m]
k2 = (2*np.pi/5e-7)**2

rad2arc = 3600 * 180/np.pi

F  = data_dict['frac_cn2 vector']
h  = data_dict['h vector'] * 1e3
a0 = data_dict['seeing DET1'] / rad2arc

r0 = 0.976*wvl/a0

F = F[:sum(F.astype(bool))-1]
h = h[:sum(h.astype(bool))-1]

H = ( np.trapz(F*h**(5/3),h) / np.trapz(F,h) )**(3/5)
Theta_0 = 0.314 * r0/H * rad2arc #[asec]

print(Theta_0)


#%%
pitch = 0.2 #[m]
D = 8 #[m]
sampling = PSD_size/240
dk = 1/D/sampling # PSD spatial frequency step
kc = 1/(2*pitch)

def radial_profile(data, center):
    y, x = np.indices((data.shape))
    r = np.sqrt( (x-center[0])**2 + (y-center[1])**2 )
    r = r.astype('int')

    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr
    return radialprofile 

PSD_profile = radial_profile(PSD, (PSD_size//2,PSD_size//2))[:PSD_size//2]

k = np.arange(PSD_size//2)*dk

plt.plot(k, PSD_profile, label='Telemetry PSD')
plt.axvline(x=kc, color='black', linestyle='--', label='Cut-off freq.')
plt.xlim([k[1],k.max()])
plt.title('Telemetry PSD')
plt.yscale('log')
plt.xscale('log')
plt.ylim([10,8e6])
plt.xlabel(r'Spatial frequency, $m^{-1}$')
plt.legend()
plt.grid()
plt.show()
# %%
