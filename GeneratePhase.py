import numpy as np
import cupy as cp
from numpy.random import RandomState
import scipy.special as spc
import cupyx.scipy.fft as gfft
from cupyx.scipy.fft import get_fft_plan


def GenerateFullPSD(PSD_AO, r0=0.1, L0=47.93, return_vK=False):
    xp = cp.get_array_module(PSD_AO)
       
    dk = 1.0/8.0 # PSD spatial frequency step [m-1]
    pitch = 0.2 # [m]
    kc = 1/(2*pitch)
    SPARTAunit2Microns = 35 # 35 microns per SPARTA unit
    Surface2WF = 2 # multiplicative factor to go from surface to WF
    A = SPARTAunit2Microns * Surface2WF

    def spectrum(r0, L0, N=240): 
        def freq_array(nX, L=1, offset=1e-10):
            k2D = xp.mgrid[0:nX, 0:nX].astype(float)
            k2D[0] -= nX//2
            k2D[1] -= nX//2
            k2D     = k2D*L + offset
            return k2D[0], k2D[1]
        
        kx,ky = freq_array(N, offset=1e-10, L=dk)
        k2 = kx**2 + ky**2
        cte = (24*spc.gamma(6/5)/5)**(5/6)*(spc.gamma(11/6)**2./(2.*xp.pi**(11/3)))
        PSD_out = r0**(-5/3)*cte*(k2 + 1/L0**2)**(-11/6)
        PSD_out[PSD_out.shape[0]//2, PSD_out.shape[1]//2] = 0.0
        return PSD_out, kx, ky

    rad2nm = 500.0/2/xp.pi #[nm]
    PSD_vK, kx, ky = spectrum(r0, L0, N=240) #r0 is @ 500 nm
    PSD_vK *= (dk*rad2nm)**2

    mask_AO  = 1.0*((kx**2+ky**2) <= kc**2)
    mask_atm = 1.0 - mask_AO

    xx,yy = xp.meshgrid(xp.arange(0,480//2),  xp.arange(0,480//2))
    PSD_small = xp.zeros([480//2]*2)
    PSD_small = PSD_AO[xx*2, yy*2] + PSD_AO[xx*2+1, yy*2] + PSD_AO[xx*2, yy*2+1] + PSD_AO[xx*2+1, yy*2+1] # Bin the PSD
    
    PSD_total = (PSD_small*(A*1e9)**2 / 2**6 * mask_AO) + (PSD_vK*1e14*mask_atm)
    
    if return_vK:
        return PSD_total, PSD_vK*1e14
    else:
        return PSD_total


def PSDrealizationBatchGPU(PSD_inp, batch_size=500, return_CPU=True, remove_piston=True):
    mempool = cp.get_default_memory_pool()
    pinned_mempool = cp.get_default_pinned_memory_pool()

    dimensions = (PSD_inp.shape[0], PSD_inp.shape[1], batch_size)
    phase_batch = cp.zeros(dimensions, dtype=np.float32)
    realization_buf = cp.zeros(dimensions, dtype=np.complex64)
    plan = get_fft_plan(realization_buf, axes=(0,1), value_type='C2C') # for batched, C2C, 2D transform
    
    rng = np.random.default_rng()
    realization = cp.asarray(rng.normal(size=dimensions) + 1j*rng.normal(size=dimensions), dtype=cp.complex64)
    realization_buf = cp.atleast_3d(cp.sqrt(cp.array(PSD_inp))) * cp.array(realization, dtype=cp.complex64)
    
    phase_batch = cp.real(
        gfft.fftshift
        (
            1.0 / PSD_inp.shape[0] * gfft.ifft2(gfft.fftshift(realization_buf.astype(cp.complex64)), axes = (0,1), plan = plan)
        )
    )
    
    if return_CPU:
        out = phase_batch.get()
        del phase_batch, realization_buf, plan
    else:
        out = phase_batch
        del realization_buf, plan
        
    mempool.free_all_blocks()
    pinned_mempool.free_all_blocks()
    
    if remove_piston:
        return out - out.mean(axis=(0,1))[None, None, :] # in [nm]
    else:
        return out # in [nm]
    

def remove_piston_and_TT(WFs, pupil):
    WFs -= WFs.mean(axis=(0,1))[None,None,:] # remove piston

    # Generate tip-tilt modes
    tip, tilt = cp.meshgrid(cp.arange(0, WFs.shape[0]),  cp.arange(0, WFs.shape[1]))

    def TT_mode(TT):
        TT = TT.astype(cp.float32) / cp.std(TT[pupil>0])
        return (TT - cp.mean(TT[pupil>0])) * pupil

    TT_basis = cp.dstack([TT_mode(tip), TT_mode(tilt)])
    TT_coefs = (WFs*pupil[...,None]).reshape(-1, WFs.shape[-1]).T @ TT_basis.reshape(-1, 2) / pupil.sum()
    
    return WFs*pupil[...,None] - TT_basis @ TT_coefs.T # remove tip-tilt


# Functions below map WFE to DIMM seiing, based on 'Test report GALACSI NFM October 2019 TTR-104.0016.docx', page 31
def HO_WFE_vs_seeing(seeing):
    a, b = (104.27, 58.6)  # [nm/arcsec], [nm]
    return a * seeing + b


def HO_WFE_vs_r0(r0):    
    rad2arc = 3600 * 180 / np.pi
    lmbd = 0.5e-6 # [m]
    seeing = rad2arc * 0.976 * lmbd/r0
    
    return HO_WFE_vs_seeing(seeing)    
    

def HO_PSD_vs_seeing(PSD, seeing, return_vK=False):
    # dk = 1.0/8.0
    # PSD_integral = np.trapz(np.trapz(PSD, dx=dk, axis=0), dx=dk, axis=0)
    # WFE_reference = np.sqrt(PSD_integral) * 1e-6 / np.sqrt(3) # this sqrt(3) is basically a kind of magic number to match the PSD to the WFE
    
    WFE_reference = 180.0
    WFE_expected  = HO_WFE_vs_seeing(seeing) # [nm]
    ratio = WFE_expected**2 / WFE_reference**2
    
    rad2arc = 3600 * 180 / np.pi
    r0 = rad2arc*0.976*0.5e-6 / seeing # [m]

    return GenerateFullPSD(PSD*ratio, r0, 47.93, return_vK)


def HO_PSD_vs_r0(PSD, r0, return_vK=False):
    # dk = 1.0/8.0
    # PSD_integral = np.trapz(np.trapz(PSD, dx=dk, axis=0), dx=dk, axis=0)
    # WFE_reference = np.sqrt(PSD_integral) * 1e-6 / np.sqrt(3) # this sqrt(3) is basically a kind of magic number to match the PSD to the WFE
    
    WFE_reference = 180.0
    WFE_expected  = HO_WFE_vs_r0(r0) # [nm]
    ratio = WFE_expected**2 / WFE_reference**2
    
    return GenerateFullPSD(PSD * ratio, r0, 47.93, return_vK)