#%%
from matplotlib import colors
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from matplotlib import colors

import os
from tqdm import tqdm

import cupy as cp
from cupyx.scipy.fft import get_fft_plan
import cupyx.scipy.fft as gfft

#%%

# Loading influence functions
#path_IFs = os.path.normpath('C:\\Users\\akuznets\\Data\\AOF\\Modes\\KL_modes_new (Nov 2021)\\DM influence functions')
#path_IFs = os.path.normpath('C:\\Users\\akuznets\\Data\\AOF\\Modes\\LWE simulation (Dec 2021)\\IF')
path_IFs = os.path.normpath('C:\\Users\\akuznets\\Data\\AOF\\Modes\\NFM_CM_files_181121\\IF_2021-11-18_18-15-54')
allDM = np.zeros([240, 240, 1156]) # command 2 phase (3D cube)

for cpt in tqdm(range(1156)):
    with fits.open(os.path.join(path_IFs, 'Mode_'+str(cpt+1)+'.fits')) as HDUL:
	    allDM[:,:,cpt] = np.array(HDUL[0].data)

allDM2 = allDM.reshape(240*240, 1156) # command 2 phase (2D matrix)

#%
pathMat = os.path.normpath('C:\\Users\\akuznets\\Data\\AOF\\LTAO_PSD_HarnessData')
#pathFile='0026_HOCM_newref_3_HO_KI_-0.4'
pathFile='0601_HOCM_2_HO_KI_-0.4'

slopes = []
CMLT = []
CM = []

def ReadFITSdata(filename):
    with fits.open(filename) as HDUL:
        return HDUL[0].data

with fits.open(os.path.join(pathMat, pathFile, pathFile+'.fits')) as HDUL:
    dat = np.array(HDUL[1].data)

# Iterate over WFSs
for i in range(1,5):
    # Read slopes per WFS
    slopes.append(dat['WFS'+str(i)+'_Gradients'])
    # Read LTAO control matricies
    CM.append( ReadFITSdata(os.path.join(pathMat, pathFile, 'LGSRecn.REC'+str(i)+'.HOCM.fits')) )
    # Read another (?????) LTAO control matricies
    CMLT.append( ReadFITSdata(os.path.join(pathMat, 'LTAO_CM'+str(i)+'_Ref_4Lay1300_nt100_np2p5.fits')) )

# Load the rest
#eigenmodes  = ReadFITSdata(os.path.join(pathMat, pathFile, 'RTC.KL.eigenmodes.fits'))
#eigenvalues = ReadFITSdata(os.path.join(pathMat, pathFile, 'RTC.KL.eigenvalues.fits'))
#Act_valid   = ReadFITSdata(os.path.join(pathMat, pathFile, 'RTC.USED_ACT_MAP.fits'))

eigenmodes  = ReadFITSdata(os.path.normpath('C:\\Users\\akuznets\\Data\\AOF\\Modes\\NFM_CM_files_181121\\eigenmodes.fits'))
eigenvalues = ReadFITSdata(os.path.normpath('C:\\Users\\akuznets\\Data\\AOF\\Modes\\NFM_CM_files_181121\\eigenvalues.fits'))
Act_valid   = ReadFITSdata(os.path.normpath('C:\\Users\\akuznets\\Data\\AOF\\Modes\\NFM_CM_files_181121\\UsedActMap.fits'))

Act_valid = np.array(Act_valid)-1
Act_broken = list(set(range(1156))-set(Act_valid.tolist()[0]))
Act_broken.sort()

KLmod = np.zeros([1156, eigenmodes.shape[0]])
KLmod[Act_valid,:] = eigenmodes
Cmd2KL = np.linalg.pinv(KLmod)

KL2phase = (allDM2 @ KLmod)
pupil = np.array([KL2phase[:,0]!=0]).reshape([240,240])*1
KL_modes_cube = KL2phase.reshape([240, 240, KLmod.shape[1]])


#%%
'''
mempool = cp.get_default_memory_pool()
pinned_mempool = cp.get_default_pinned_memory_pool()

num_modes = 200

oversampling = 4

size = int(240*oversampling)

ROI = slice(size//2-240//2, size//2+240//2)

FFT_KL_padded = cp.zeros([size,size,num_modes], dtype=cp.float32)
FFT_KL_padded[ROI,ROI,:num_modes] = cp.array(KL_modes_cube[:,:,:num_modes], dtype=cp.float32)

plan = get_fft_plan(FFT_KL_padded, axes=(0,1), value_type='C2C') # for batched, C2C, 2D transform

FFT_KL = gfft.fftshift(1/size*gfft.fft2(gfft.fftshift(FFT_KL_padded), axes=(0,1), plan=plan))
FFT_KL_stack = cp.asnumpy(cp.abs(FFT_KL))

del FFT_KL, FFT_KL_padded

mempool.free_all_blocks()
pinned_mempool.free_all_blocks() # clear GPU memory


PSD_size = FFT_KL_stack.shape[0]
crop = 44*2
zoomed = slice(PSD_size//2-crop,PSD_size//2+crop)

for i in range(FFT_KL_stack.shape[2]):
    plt.figure(dpi=300)
    plt.imshow(np.log(FFT_KL_stack[zoomed,zoomed,i]))
    plt.title('KL'+str(i))
    plt.savefig('C:\\Users\\akuznets\\Desktop\\buf\\modes\\KL'+str(i)+'.png')

for i in range(KL_modes_cube.shape[2]):
    plt.figure(dpi=300)
    plt.imshow(KL_modes_cube[:,:,i])
    plt.title('KL'+str(i))
    plt.savefig('C:\\Users\\akuznets\\Desktop\\buf\\modes\\KL'+str(i)+'.png')
'''

#%% ----------------------------------- Computing KL ids --------------------------------------------

#convert_factor = 2*np.pi/0.5e-6*1e-6;  #[microns] -> [rad] at lambda=500 [nm]
SPARTAunit2Microns = 35 # 35 microns per SPARTA unit
Surface2WF = 2 # multiplicative factor to go from surface to WF
A = SPARTAunit2Microns * Surface2WF * 1e3 #* convert_factor

use_GPU = True

if use_GPU:
    print('Computing KL modes coefficients from WFS slopes (on GPU)...')
    mempool = cp.get_default_memory_pool()
    pinned_mempool = cp.get_default_pinned_memory_pool()

    KL_ids     = cp.zeros([10000, 1150, 4], dtype=cp.float32)
    slopes_buf = cp.zeros([10000, 2480], dtype=cp.float32)
    CMLT_buf   = cp.zeros([1156,  2480], dtype=cp.float32)
    Cmd2KL_buf = cp.array(Cmd2KL, dtype=cp.float32)

    for i in range(4):
        slopes_buf = cp.array(slopes[i], dtype=cp.float32)
        CMLT_buf   = cp.array(CMLT[i], dtype=cp.float32)
        KL_ids[:,:,i] = slopes_buf @ CMLT_buf.T @ Cmd2KL_buf.T
    KL_ids = KL_ids.mean(axis=2) # averaging KL_ids from each WFS
    del Cmd2KL_buf, CMLT_buf, slopes_buf
    #KL_ids = cp.asnumpy(KL_ids)
else:
    print('Computing KL modes coefficients from WFS slopes...')
    KL_ids = []
    for i in range(4):
        KL_ids.append(slopes[i] @ CMLT[i].T @ Cmd2KL.T)
    KL_ids = np.dstack(KL_ids)
    KL_ids = KL_ids.mean(axis=2) # averaging KL_ids from each WFS
print('Done!')

#include = 19
#exclude = list(set(range(20))-set([include]))
exclude = list(set([0,1,2])) #,33,34,56,57]))
#KL_ids[:,exclude] = 0.0

# ----------------------------------- Computing PSD --------------------------------------------
if not use_GPU:
    print('Computing residual phase sreens from KL modes coefficients...')
    phase_screens = KL2phase @ KL_ids.T * A 
    phase_screens_cube = phase_screens.reshape([240,240,phase_screens.shape[1]])
    print('Done!')

    print('Computing FFTs of each phase screen and spectra...')
    spectra = np.zeros([480, 480, phase_screens_cube.shape[2]], dtype=np.complex64)
    phase = np.zeros([480, 480], dtype=np.complex64)
    for i in tqdm(range(phase_screens_cube.shape[2])):
        phase[240//2:240//2+240, 240//2:240//2+240] = phase_screens_cube[:,:,i] # zero padding
        spectra[:,:,i] = np.abs(np.fft.fftshift( 1/480. * np.fft.fft2(np.fft.fftshift(phase)) ))
    print('Done!\nComputing PSD...')
    PSD = spectra.var(axis=2)
    del spectra, phase
    print('Done!')

else: 
    KL2phase_buf = cp.array(KL2phase, dtype=cp.float32)

    batch_size = 500
    N = slopes[0].shape[0]

    print('Computing FFTs of each phase screen and spectra (on GPU)...')
    temp_mean          = cp.zeros([480, 480], dtype=cp.float32)
    FFT_batch          = cp.zeros([480, 480, batch_size], dtype=cp.complex64)
    phase_padded_batch = cp.zeros([480, 480, batch_size], dtype=np.float32)
    variance_batches   = cp.zeros([480, 480, N//batch_size], dtype=cp.float32)

    plan = get_fft_plan(phase_padded_batch, axes=(0,1), value_type='C2C') # for batched, C2C, 2D transform

    for i in tqdm(range(N//batch_size)):
        #phase_batch = cp.array( phase_screens_cube[:,:,i*batch_size:(i+1)*batch_size], dtype=cp.float32 ) # zero padding
        #phase_batch = (KL2phase_buf @ KL_ids[i*batch_size:(i+1)*batch_size,:].T * A).reshape([240,240,batch_size])
        phase_padded_batch[240//2:240//2+240, 240//2:240//2+240] = \
            (KL2phase_buf @ KL_ids[i*batch_size:(i+1)*batch_size,:].T * A).reshape([240,240,batch_size])
            #phase_batch
        FFT_batch = cp.abs( gfft.fftshift(1/480.*gfft.fft2(gfft.fftshift(phase_padded_batch), axes=(0,1), plan=plan)) )
        temp_mean = FFT_batch.mean(axis=2, keepdims=True)
        variance_batches[:,:,i] = cp.sum( (FFT_batch-temp_mean)**2, axis=2 )

    PSD = cp.asnumpy( variance_batches.sum(axis=2) ) / N

    del phase_padded_batch, FFT_batch, variance_batches, temp_mean, plan, KL2phase_buf #, phase_batch
    mempool.free_all_blocks()
    pinned_mempool.free_all_blocks() # clear GPU memory
    print('Done!')

#import pickle

#with open('C:\\Users\\akuznets\\Desktop\\buf\\PSD_ex'+str(include)+'.pickle', 'wb') as handle:
#with open('C:\\Users\\akuznets\\Desktop\\buf\\PSD_ex'+str(exclude)+'.pickle', 'wb') as handle:
#    pickle.dump(PSD, handle, protocol=pickle.HIGHEST_PROTOCOL)

#num_modes = 3
#max_spectrums = np.zeros([num_modes])

#for i in range(10,11):
#    test = KL_ids[:,i]
#    KL_fft = np.fft.fft(test)
#    test = np.abs(test)
#    #max_spectrums[i] = test.max()
#    plt.plot(test, label='KL '+str(i)) #+3))
#    plt.ylim([0, 0.004])
#    plt.xlim([0,KL_ids.shape[0]])
#    plt.legend()
#    plt.show()

#plt.figure(dpi=300)
#plt.semilogy(max_spectrums)
#plt.grid()
#plt.xlabel('KL mode')
#plt.ylabel(r"Max val. of FFT($KL_{id}$) spectrum")
#plt.show()

#%
'''
list_files = os.listdir('C:\\Users\\akuznets\\Desktop\\buf\\PSDs')
data = []
for file in list_files:
    with open(os.path.join('C:\\Users\\akuznets\\Desktop\\buf\\PSDs',file), 'rb') as handle:
        data.append( pickle.load(handle) )
data = np.dstack(data)

titles = ['All included'] #,'KL0-20 exluded']
for i in range(data.shape[2]):
    #titles.append('KL0-20 ex, #'+str(i)+' in')
    titles.append('KL'+str(i)+' excluded')

max_val = 15 #data.max()
min_val = 6#data.min()

crop = 44
PSD_size = data.shape[0]
zoomed = slice(PSD_size//2-crop, PSD_size//2+crop)

from matplotlib import colors, scale
norm=colors.LogNorm(vmin=min_val, vmax=max_val)

for i in range(data.shape[2]):
    plt.figure(dpi=300)
    plt.imshow(np.log(data[zoomed,zoomed,i]), norm=norm)
    plt.colorbar()
    plt.title(titles[i])
    plt.savefig('C:\\Users\\akuznets\\Desktop\\buf\\gif\\'+titles[i]+'.png')


#%

files = os.listdir('C:\\Users\\akuznets\\Desktop\\buf\\gif\\')

image_stack = []

for file in files:
    temp = plt.imread( os.path.join('C:\\Users\\akuznets\\Desktop\\buf\\gif\\', file) )
    image_stack.append(temp)


def save_GIF_RGB(images_stack, duration=1e3, downscale=4, path='test.gif'):
    from PIL import Image
    gif_anim = []
    
    def remove_transparency(img, bg_colour=(255, 255, 255)):
        alpha = img.convert('RGBA').split()[-1]
        bg = Image.new("RGBA", img.size, bg_colour + (255,))
        bg.paste(img, mask=alpha)
        bg = bg.convert('RGB')
        return bg
    
    for layer in images_stack:
        im = Image.fromarray(np.uint8(layer*255))
        im.thumbnail((im.size[0]//downscale, im.size[1]//downscale), Image.ANTIALIAS)
        gif_anim.append( remove_transparency(im) )
        gif_anim[0].save(path, save_all=True, append_images=gif_anim[1:], optimize=True, duration=duration, loop=0)


save_GIF_RGB(image_stack, duration=1e3, path='C:\\Users\\akuznets\\Desktop\\buf\\gif\\result.gif')
'''
#%
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

#test = KL2phase[:,exclude] @ cp.asnumpy(KL_ids[:,exclude].T) * A
#
#test_mean = test.mean(axis=1).reshape([240,240])
#
#plt.imshow(test_mean)

#%

#buf = np.zeros([480, 480], dtype=np.complex64)
#
#buf[240//2:240//2+240, 240//2:240//2+240] = test_mean # zero padding
#test_FFT = np.abs(np.fft.fftshift( 1/480. * np.fft.fft2(np.fft.fftshift(buf)) ))
#
#plt.imshow(np.log(test_FFT[zoomed]))

#%%
pitch = 0.2 #[m]
D = 8 #[m]
sampling = PSD_size/pupil.shape[0]
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
