#%%
from datetime import datetime 
from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np
import os

from GeneratePSD import GeneratePSD

#%%
# Loading influence functions
base       = os.path.normpath('C:/Users/akuznets/Data/AOF/')
path_IFs   = os.path.normpath(os.path.join(base,'Modes/NFM_CM_files_181121/'))
path_eigen = os.path.normpath(os.path.join(base,'Modes/NFM_CM_files_181121/'))
path_root  = os.path.normpath(os.path.join(base,'LTAO_PSD_HarnessData/0026_HOCM_newref_3_HO_KI_-0.4/'))
#path_root  = os.path.normpath(os.path.join(base,'LTAO_PSD_HarnessData/Day1_nMaxModes/0020_HOCM_1_HO_KI_-0.4/'))
CMLT_paths = [ os.path.join(path_root, 'LTAO_CM'+str(i)+'_Ref_4Lay1300_nt100_np2p5.fits') for i in range(1,5) ]

IF_entries = {
    'eigmod': 'eigenmodes',
    'eigval': 'eigenvalues',
    'usdact': 'UsedActMap'
}
PSD_generator = GeneratePSD(path_root, path_IFs, path_eigen, CMLT_paths)
PSD_generator.IF_entries = IF_entries
PSD_generator.exclude_modes = list(set([0,1,2])) #,33,34,56,57]))

PSD = PSD_generator.ComputePSD()


PSD_size = PSD.shape[0]
crop = 44
zoomed = slice(PSD_size//2-crop,PSD_size//2+crop)
zoomed = (zoomed,zoomed)
PSD_cropped = PSD[zoomed]
norm = colors.LogNorm(vmin=6, vmax=15)
plt.imshow(np.log(PSD_cropped), norm=norm)
plt.colorbar()
plt.show()

#%%

filename = 'C:/Users/akuznets/Data/AOF/LTAO_PSD_HarnessData/Day1_nMaxModes/0020_HOCM_1_HO_KI_-0.4/0020_HOCM_1_HO_KI_-0.4.txt'

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
