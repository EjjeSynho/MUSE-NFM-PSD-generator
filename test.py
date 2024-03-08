#%%
import numpy as np
import cupy as cp
import scipy.special as spc
import scipy.ndimage
import matplotlib.pyplot as plt
from cupyx.scipy.fft import get_fft_plan
import cupyx.scipy.fft as gfft
import cupyx.scipy.ndimage
from tqdm import tqdm

xp = cp

double_precision = True

if double_precision:
    datafloat     = xp.float64
    datacomplex   = xp.complex128
    datafloat_cpu = np.float64
else:
    datafloat     = xp.float32
    datacomplex   = xp.complex64
    datafloat_cpu = np.float32    


def freq_array(N, dx):
    """Generate spatial frequency arrays."""
    df = 1 / (N*dx)  # Spatial frequency interval [1/m]
    fx = (np.arange(-N//2, N//2, 1) + N%2) * df
    fy = (np.arange(-N//2, N//2, 1) + N%2) * df
    fx, fy = np.meshgrid(fx, fy, indexing='ij')
    return fx, fy, np.sqrt(fx**2 + fy**2), df


def vonKarmanPSD(k, r0, L0):
    """Calculate the von Karman PSD."""
    cte = (24*spc.gamma(6/5)/5)**(5/6)*(spc.gamma(11/6)**2./(2.*np.pi**(11/3)))
    PSD = r0**(-5/3)*cte*(k**2 + 1/L0**2)**(-11/6)
    PSD[PSD.shape[0]//2, PSD.shape[1]//2] = 0  # Avoid division by zero at the origin
    return PSD


def mask_circle(N, r, center=(0,0), centered=True):
    """Generates a circular mask of radius r in a grid of size N."""
    factor = 0.5 * (1-N%2)
    if centered:
        coord_range = np.linspace(-N//2+N%2+factor, N//2-factor, N)
    else:
        coord_range = np.linspace(0, N-1, N)
    xx, yy = np.meshgrid(coord_range-center[1], coord_range-center[0])
    pupil_round = np.zeros([N, N], dtype=np.int32)
    pupil_round[np.sqrt(yy**2+xx**2) < r] = 1
    return pupil_round

def generate_phase_screen(PSD, N):
    random_phase = np.exp(2j * np.pi * np.random.rand(N, N))
    complex_spectrum = np.fft.ifftshift(np.sqrt(PSD) * random_phase)
    phase_screen = np.fft.ifft2(complex_spectrum) * PSD.size
    phase_screen_nm = np.real(phase_screen)  
    return phase_screen_nm

#%%
# Parameters
D  = 8.0  # Size of the phase screen [m]
r0 = 0.1  # Fried parameter [m]
L0 = 50.0 # Outer scale [m]

dx = r0 / 3.0 # Spatial sampling interval [m/pixel], make sure r0 is Nyquist sampled
dt = 0.001 # Time step [s/step]

boiler = lambda f: f * 2*dx # just radial linear gradient map with 1.0 at the N/2 distance from the center

#%%
# Initialize cascade parameters
n_cascades = 3
factor = 3**(n_cascades-1)
N  = D / dx # [pixels] Number of grid points
N_min = np.ceil(N/factor)
N_min += 1 - N_min%2 # Make sure that the minimal size in the cascade is odd so it has a central piston pixel
N = int(N_min * factor) # Make sure size is subdividable by 3, because each cascade zooms 3X
num_screens = 200 #[step]
wvl_atmo = 500.0 # [nm]
rad2nm = wvl_atmo / 2.0 / np.pi # [nm/rad]

fx_, fy_, f_, df_ = freq_array(N, dx) # [1/m]
PSD_test = vonKarmanPSD(f_, r0, L0) * df_**2 * rad2nm**2 # [nm^2/m^2]

# Pre-allocate arrays
arrays_shape = [N, N, n_cascades]
PSD_temporal = np.zeros(arrays_shape, dtype=datafloat_cpu) # Describes the temporal evolution of the PSD
PSDs = np.zeros(arrays_shape, dtype=datafloat_cpu) # Contains spatial frequencies
fx   = np.zeros(arrays_shape, dtype=datafloat_cpu)
fy   = np.zeros(arrays_shape, dtype=datafloat_cpu)
f    = np.zeros(arrays_shape, dtype=datafloat_cpu)
df   = np.zeros(n_cascades,   dtype=datafloat_cpu)


select_middle = lambda N: np.s_[N//2-N//6:N//2+N//6+N%2, N//2-N//6:N//2+N//6+N%2, :]
crops = [np.s_[0:N, 0:N, :]] # selects the whole image

for i in range(0, n_cascades):
    crops.append(select_middle(N // 3**i)) # selects thw middle 1/3 quandrant of the image
    dx_ = dx * 3**i # every next cascade zooms out by 3 to capture more low frequencies
    fx[...,i], fy[...,i], f[...,i], df[...,i] = freq_array(N, dx_) # [1/m]
    PSDs[...,i] = vonKarmanPSD(f[...,i], r0, L0) * df[i]**2 * rad2nm**2 # PSD spatial [nm^2/m^2]
    PSD_temporal[...,i] = boiler(f[...,i]) # PSD temporal [??/??]
_ = crops.pop(-1)

# Masks are used to supress the spatial frequencies that belong to other cascades
mask_outer = mask_circle(N, N/2, centered=True)
mask_inner = mask_circle(N, N/6, centered=True)
masks = np.ones(arrays_shape, dtype=datafloat_cpu)

PSDs[...,:-1] *= (mask_outer-mask_inner)[..., np.newaxis]
PSDs[...,-1]  *=  mask_outer # The last cascade has the most information about the low frequencies

PSD_temporal[...,:-1] *= (mask_outer-mask_inner)[..., np.newaxis]
PSD_temporal[...,-1]  *=  mask_outer

#%%
def screens_from_PSD_and_phase(PSD, PSD_phase):
    """Generate phase screens batch from PSD and random phase."""
    dimensions = PSD_phase.shape
    PSD_realizations = cp.zeros(dimensions, dtype=datacomplex)
    phase_batch = cp.zeros(dimensions, dtype=datafloat)
    
    plan = get_fft_plan(PSD_realizations, axes=(0,1), value_type='C2C') # for batched, C2C, 2D transform

    PSD_ = cp.atleast_3d(cp.asarray(PSD, dtype=datafloat))
    PSD_realizations = cp.sqrt(PSD_) * PSD_phase

    # Perform batched FFT
    phase_batch = cp.real( 
        gfft.ifft2(
            gfft.ifftshift(
                PSD_realizations, axes=(0,1))
            , axes=(0,1), plan=plan
        ) * PSD.size
    )
    return phase_batch


def zoomX3(x, iters=0, interp_order=3):
    """Scales up the resolution of the phase screens stack 'x' by a factor of 3."""	
    if iters > 0:
        zoom_factor = (3**iters, 3**iters, 1)
        if cp.get_array_module(x) == np:
            return scipy.ndimage.zoom(x, zoom_factor, order=interp_order)
        else:
            return cupyx.scipy.ndimage.zoom(x, zoom_factor, order=interp_order)
    else:
        return x


def zoomX3_FFT(phase_screens_batch):
    N_ = phase_screens_batch.shape[0]
    hanning_window = cp.array( (np.hanning(N_).reshape(-1, 1) * np.hanning(N_))[..., None], dtype=datafloat )
    hanning_window /= 2
    hanning_window += 1
    
    original_height, original_width, num_screens = phase_screens_batch.shape
    new_height = int(original_height * 3)
    new_width  = int(original_width  * 3)
    
    fft_batch = cp.fft.fft2(phase_screens_batch, axes=(0, 1))
    fft_shifted_batch = cp.fft.fftshift(fft_batch, axes=(0, 1))
    
    padded_fft_batch = cp.zeros((new_height, new_width, num_screens), dtype=datacomplex)
    pad_height_start = (new_height - original_height) // 2
    pad_width_start  = (new_width  - original_width)  // 2
    
    padded_fft_batch[
        pad_height_start : pad_height_start + original_height,
        pad_width_start  : pad_width_start  + original_width,
    :] = fft_shifted_batch # * hanning_window
    
    ifft_shifted_batch = cp.fft.ifftshift(padded_fft_batch, axes=(0, 1))
    interpolated_batch = cp.fft.ifft2(ifft_shifted_batch, axes=(0, 1))
    
    return cp.real(interpolated_batch) * 3**2


#%%
mempool = cp.get_default_memory_pool()
pinned_mempool = cp.get_default_pinned_memory_pool()

rng = np.random.default_rng() # random generation on the CPU is faster

noise_realization = cp.asarray( np.random.normal(size=(N, N, num_screens)), dtype=datafloat_cpu ) # random per screen
random_phase_all  = cp.exp(2j*cp.pi*noise_realization)

wind_speed = 40 * 0 # [m/s]
wind_direction = 45 #[degree]
boiling_factor = datafloat(0.5) * 100.0

# Frames IDs
screen_id = cp.arange(num_screens, dtype=datafloat)[None, None, :] # 1 x 1 x N_screens

# Generate tip/tilt modes to simulate directional moving of the frozen flow
coords  = np.linspace(0, N-1, N) - N//2 + 0.5 * (1-N%2)
[xx,yy] = np.meshgrid( coords, coords, copy=False)
center_grid = lambda x: x / (N//2 - 0.5*(1-N%2)) / 2.0
tip  = cp.array( center_grid(xx)[..., None], dtype=datafloat ) # 1/N-th of it shifts phase screen by 1 pixel
tilt = cp.array( center_grid(yy)[..., None], dtype=datafloat )

#%%
# Initial random realization of PSD at t=0
init_noise = cp.asarray( rng.normal(size=(N,N,1)), dtype=datafloat )

# Spatially-dependant phase retardation of the PSD's realisation phase, used to simulate boiling
random_retardation = cp.asarray(np.random.uniform(0, 1, size=(N,N,1)), dtype=datafloat)

# Generate cascaded phase screens (W x H x N_screens x N_cascades)
screens_cascade = cp.zeros([N, N, num_screens, n_cascades], dtype=datafloat)

mask_cascade = [0,0,1]

for i in range(n_cascades):
    # Due to zooming out the lower frequencies, the wind speed must to be slowed down
    V = cp.array(wind_speed / dx * dt, dtype=datafloat) # [pixels/step]
    Vx_ = V * cp.cos(cp.deg2rad(cp.array(wind_direction, dtype=datafloat))) / 3**i
    Vy_ = V * cp.sin(cp.deg2rad(cp.array(wind_direction, dtype=datafloat))) / 3**i

    # Temporal PSD defines boiling per spatial frequency
    PSD_temporal_ = cp.array(PSD_temporal[...,i][..., None], datafloat)
    evolution = tip*Vx_ + tilt*Vy_ + random_retardation * boiling_factor * PSD_temporal_
    random_phase = cp.exp(2j*cp.pi * (screen_id*evolution + init_noise) )
    
    # screens_cascade[...,i] = zoomX3( screens_from_PSD_and_phase(PSDs[...,i], random_phase)[crops[i]], i )
    screens_cascade[...,i] = zoomX3( screens_from_PSD_and_phase(PSDs[...,i], random_phase_all)[crops[i]], i ) * mask_cascade[i]

screens_cascade[...,-1] -= screens_cascade[...,-1].mean(axis=(0,1), keepdims=True) # remove piston again
screens_cascade = screens_cascade.sum(axis=-1) # sum all cascades to W x H x N_screens

mempool.free_all_blocks()
pinned_mempool.free_all_blocks() # clear GPU memory

#%%
from matplotlib import cm
from skimage.transform import rescale
import cv2
from tqdm import tqdm

colormap = cm.viridis
scale_factor = 2
normalizer = cp.abs(screens_cascade).max().get()

output_video_path = 'screens.mp4'
height, width, layers = screens_cascade.shape

fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec used to compress the frames
video  = cv2.VideoWriter(output_video_path, fourcc, 20.0, (width*scale_factor, height*scale_factor))

print('Writing video...')
for i in tqdm(range(layers)):
    buf = (screens_cascade[..., i].get() + normalizer) / 2 / normalizer
    buf = rescale(buf, scale_factor, order=1)
    frame = (colormap(buf) * 255).astype(np.uint8)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    video.write(frame)

video.release()

#%%
fx_,  fy_,  f_,  df_  = freq_array(N, dx)
fx_3, fy_3, f_3, df_3 = freq_array(N, 3*dx)

PSD_test   = vonKarmanPSD(f_,  r0, L0) * df_**2  * rad2nm**2
PSD_test_1 = vonKarmanPSD(f_,  r0, L0) * df_**2  * rad2nm**2
PSD_test_2 = vonKarmanPSD(f_,  r0, L0) * df_**2  * rad2nm**2
PSD_test_3 = vonKarmanPSD(f_3, r0, L0) * df_3**2 * rad2nm**2

mask_out = mask_circle(N, N/2, centered=True)
mask_in  = mask_circle(N, N/6, centered=True)

PSD_test_2 *= mask_out-mask_in
PSD_test_1 *= mask_in
PSD_test_3 *= mask_out

# PSD_test_2 *=0
el_croppo = select_middle(N)

noise_realization = cp.asarray( np.random.normal(size=(f_.shape[0], f_.shape[1], num_screens)), dtype=datafloat_cpu ) # random per screen
random_phase_all  = cp.exp(2j*cp.pi*noise_realization)

phase_batch_1 = screens_from_PSD_and_phase(PSD_test_1, random_phase_all)
phase_batch_2 = screens_from_PSD_and_phase(PSD_test_2, random_phase_all)

phase_batch = phase_batch_1 + phase_batch_2

# phase_batch_3 = zoomX3( screens_from_PSD_and_phase(PSD_test_3, random_phase_all)[el_croppo], 2, interp_order=3 )
phase_batch_3 = screens_from_PSD_and_phase(PSD_test_3, random_phase_all) #* cp.array(mask_in[...,None])
# phase_batch_3 = zoomX3_FFT(phase_batch_3[el_croppo])
phase_batch_3 = zoomX3(phase_batch_3[el_croppo], 1, 4)

# phase_batch_3 = phase_batch_3[el_croppo]
# phase_batch_3 = zoomX3( phase_batch_3, 1, interp_order=3 )
phase_batch_3 -= phase_batch_3.mean(axis=(0,1), keepdims=True)
phase_batch_3 += phase_batch_2

#%%

# _ = plt.hist(phase_batch_1.get().flatten(), bins=100, alpha=0.5, label='1')
# # _ = plt.hist(cp.abs(phase_batch_1).get().flatten(), bins=100, alpha=0.5, label='3')
# plt.show()
# _ = plt.hist(phase_batch_3.get().flatten(), bins=100, alpha=0.5, label='3')
# # _ = plt.hist(cp.abs(phase_batch_3).get().flatten(), bins=100, alpha=0.5, label='3')
# plt.show()


#%
i = np.random.randint(0, num_screens)
plt.imshow(phase_batch[...,i].get())
plt.show()
plt.imshow(phase_batch_3[...,i].get())
plt.show()


#%%
def PSD_to_phase(phase_batch): 
    N_ = phase_batch.shape[0]
    hanning_window = (np.hanning(N_).reshape(-1, 1) * np.hanning(N_))[..., None]
    hanning_window = cp.array(hanning_window, dtype=datafloat) * 1.6322**2 # Corrective factor

    temp_mean = cp.zeros([N_, N_], dtype=datafloat)
    FFT_batch = cp.zeros([N_, N_, num_screens], dtype=datacomplex)

    plan = get_fft_plan(FFT_batch, axes=(0,1), value_type='C2C') # for batched, C2C, 2D transform

    FFT_batch = gfft.fftshift(
        gfft.fft2(gfft.fftshift(phase_batch*hanning_window, axes=(0,1)), axes=(0,1), plan=plan) / N_**2, axes=(0,1)        
    )
    temp_mean = FFT_batch.mean(axis=(0,1), keepdims=True)
    return 2 * cp.mean( cp.abs(FFT_batch-temp_mean)**2, axis=2 ).get()


def PSD_to_phase_advanced(phase_batch):
    N_, _, N_screens = phase_batch.shape
    batch_size = 32
    pad_size   = N_//2+N_%2
    N_batches  = N_screens // batch_size

    hanning_window = cp.array( (np.hanning(N_).reshape(-1, 1) * np.hanning(N_))[..., None], dtype=datafloat )

    temp_mean = cp.zeros([pad_size*2+N_, pad_size*2+N_], dtype=datafloat)
    FFT_batch = cp.zeros([pad_size*2+N_, pad_size*2+N_, batch_size], dtype=datacomplex)
    variance_batches = cp.zeros([pad_size*2+N_, pad_size*2+N_], dtype=datafloat)

    plan = get_fft_plan(FFT_batch, axes=(0,1), value_type='C2C') # for batched, C2C, 2D transform

    pad_dims = ( (pad_size, pad_size), (pad_size,pad_size), (0, 0) )

    for i in range(N_batches):
        buf = cp.pad( phase_batch[..., i*batch_size:(i+1)*batch_size]*hanning_window, pad_dims, 'constant', constant_values=(0,0))
        FFT_batch = gfft.fftshift(
            gfft.fft2(gfft.fftshift(buf, axes=(0,1)), axes=(0,1), plan=plan) / N_**2, axes=(0,1)        
        )
        temp_mean = FFT_batch.mean(axis=(0,1), keepdims=True)
        variance_batches += 2 * cp.mean( cp.abs(FFT_batch-temp_mean)**2, axis=2 )
        variance_batches = variance_batches / N_batches

    return cupyx.scipy.ndimage.zoom(variance_batches, N_/(pad_size*2+N_), order=3)

#%%
PSD_out   = PSD_to_phase(phase_batch)
PSD_out_3 = PSD_to_phase(phase_batch_3)

plt.imshow(np.log(PSD_out))
plt.show()
plt.imshow(np.log(PSD_out_3))
plt.show()


#%%


_,_,f_over, df_over = freq_array(N*17, dx)
PSD_ultimate = vonKarmanPSD(f_over, r0, L0) * df_over**2 * rad2nm**2 * 17**2

PSD_test   = vonKarmanPSD(f_, r0, L0) * df_**2  * rad2nm**2

def radial_profile(data, center):
    y, x = np.indices((data.shape))
    r = np.sqrt( (x-center[0])**2 + (y-center[1])**2 )
    r = r.astype('int')
    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr
    return radialprofile 

def radialize_PSD(PSD, grid, label=''):
    PSD_profile = radial_profile(PSD, (PSD.shape[0]//2, PSD.shape[1]//2))[:PSD.shape[1]//2]
    grid_profile = grid[grid.shape[0]//2, grid.shape[1]//2:-1]
    plt.plot(grid_profile, PSD_profile, label=label)

radialize_PSD(PSD_ultimate, f_over, 'Ultimate PSD')
radialize_PSD(PSD_test, f_,  'Intial PSD')
radialize_PSD(PSD_out_3, f_, 'Reconstructed PSD')

print(PSD_out.sum() / PSD_test.sum())

plt.grid()
plt.legend()
plt.yscale('log')
plt.xscale('log')
# plt.ylim(1e-3, 1e6)
plt.show()
#%%
