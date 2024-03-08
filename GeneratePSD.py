import numpy as np
from astropy.io import fits

import os
from tqdm import tqdm

import cupy as cp
from cupyx.scipy.fft import get_fft_plan
import cupyx.scipy.fft as gfft


class GeneratePSD:
    def __init__(self, root_path=None, IF_path=None, eigen_path=None, CMLT_paths=None, GPU=True):
        self.__use_GPU = GPU
        
        self.allDM  = None
        self.__dat  = None
        self.KL_ids = None
        self.PSD    = None

        self.slopes = []
        self.CMLT   = []
        self.CM     = []

        self.rootpath   = root_path
        self.IF_path    = IF_path
        self.CMLT_paths = CMLT_paths
        self.eigen_path = eigen_path

        self.IF_entries = {
            'eigmod': 'RTC.KL.eigenmodes',
            'eigval': 'RTC.KL.eigenvalues',
            'usdact': 'RTC.USED_ACT_MAP'
        }

        self.exclude_modes = None

        if self.__use_GPU:
            self.mempool = cp.get_default_memory_pool()
            self.pinned_mempool = cp.get_default_pinned_memory_pool()

    def __del__(self):
        if self.__use_GPU:
            self.mempool.free_all_blocks()
            self.pinned_mempool.free_all_blocks() # clear GPU memory


    def ReadFITSdata(self, filename):
        with fits.open(filename) as HDUL:
            return HDUL[0].data


    def LoadInfluenceFuncs(self, path=None):
        self.IF_path = os.path.normpath(path)
        if self.IF_path is None:
            raise ValueError('Please specify the path to influence functions')

        self.allDM = np.zeros([240, 240, 1156]) # command 2 phase (3D cube)
        for cpt in tqdm(range(1156)):
            with fits.open(os.path.join(self.IF_path, 'IF','Mode_'+str(cpt+1)+'.fits')) as HDUL:
                self.allDM[:,:,cpt] = np.array(HDUL[0].data)

        self.allDM2 = self.allDM.reshape(240*240, 1156) # command 2 phase (2D matrix)
        
        
    def LoadData(self, rootpath=None):
        if rootpath is not None:
            self.rootpath = os.path.normpath(rootpath)

        if self.rootpath is None:
            raise ValueError('Error: specify the path to data!')

        filename = os.path.split(self.rootpath)[-1]
        with fits.open(os.path.join(self.rootpath, filename+'.fits')) as HDUL:
            self.__dat = np.array(HDUL[1].data)


    #'LTAO_CM'+str(i)+'_Ref_4Lay1300_nt100_np2p5.fits'
    def LoadCMandSlopes(self, rootpath=None, IF_path=None, eigen_path=None, CMLT_paths=None):
        
        if rootpath is not None:
            self.rootpath = os.path.normpath(rootpath)

        if IF_path is not None:
            self.IF_path = os.path.normpath(IF_path)
        
        if eigen_path is not None:
            self.eigen_path = os.path.normpath(eigen_path)

        if CMLT_paths is not None:
            self.CMLT_paths = CMLT_paths

        if self.__dat is None:
            self.LoadData(self.rootpath)

        if self.allDM is None:
            self.LoadInfluenceFuncs(self.IF_path)

        # Iterate over WFSs
        for i in range(1,5):
            self.slopes.append(self.__dat['WFS'+str(i)+'_Gradients']) # Read slopes per WFS
            self.CM.append( self.ReadFITSdata(os.path.join(self.rootpath, 'LGSRecn.REC'+str(i)+'.HOCM.fits')) ) # Read LTAO control matricies
            if self.CMLT_paths is not None:
                self.CMLT.append( self.ReadFITSdata(CMLT_paths[i-1]) ) # Read default LTAO control matricies

        if self.CMLT_paths is None:
            self.CMLT = self.CM # Don't use modified control matricies, use telemetry ones instead

        #C:\\Users\\akuznets\\Data\\AOF\\Modes\\NFM_CM_files_181121\\

        # Load the rest
        self.eigenmodes  = self.ReadFITSdata(os.path.join(self.eigen_path, self.IF_entries['eigmod']+'.fits'))
        self.eigenvalues = self.ReadFITSdata(os.path.join(self.eigen_path, self.IF_entries['eigval']+'.fits'))
        self.Act_valid   = self.ReadFITSdata(os.path.join(self.eigen_path, self.IF_entries['usdact']+'.fits'))

        self.Act_valid = np.array(self.Act_valid)-1
        #Act_broken = list(set(range(1156))-set(self.Act_valid.tolist()[0]))
        #Act_broken.sort()

        KLmod = np.zeros([1156, self.eigenmodes.shape[0]])
        KLmod[self.Act_valid,:] = self.eigenmodes
        self.Cmd2KL = np.linalg.pinv(KLmod)

        self.KL2phase = (self.allDM2 @ KLmod)
        self.pupil = np.array([self.KL2phase[:,0]!=0]).reshape([240,240])*1
        #self.KL_modes_cube = self.KL2phase.reshape([240, 240, KLmod.shape[1]])


    def ComputeKLids(self):
        if not self.CM or not self.CMLT or not self.slopes:
            self.LoadCMandSlopes(self.rootpath, self.IF_path, self.eigen_path, self.CMLT_paths)

        if self.__use_GPU:
            print('Computing KL modes coefficients from WFS slopes (on GPU)...')

            self.KL_ids = cp.zeros([10000, 1150, 4], dtype=cp.float32)
            slopes_buf  = cp.zeros([10000, 2480], dtype=cp.float32)
            CMLT_buf    = cp.zeros([1156,  2480], dtype=cp.float32)
            Cmd2KL_buf  = cp.array(self.Cmd2KL, dtype=cp.float32)

            for i in range(4):
                slopes_buf = cp.array(self.slopes[i], dtype=cp.float32)
                CMLT_buf   = cp.array(self.CMLT[i], dtype=cp.float32)
                self.KL_ids[:,:,i] = slopes_buf @ CMLT_buf.T @ Cmd2KL_buf.T
            self.KL_ids = self.KL_ids.mean(axis=2) # averaging KL_ids from each WFS
            
            del Cmd2KL_buf, CMLT_buf, slopes_buf
            self.mempool.free_all_blocks()
            self.pinned_mempool.free_all_blocks() # clear GPU memory

        else:
            print('Computing KL modes coefficients from WFS slopes...')
            self.KL_ids = []
            for i in range(4):
                self.KL_ids.append(self.slopes[i] @ self.CMLT[i].T @ self.Cmd2KL.T)
            self.KL_ids = np.dstack(self.KL_ids)
            self.KL_ids = self.KL_ids.mean(axis=2) # averaging KL_ids from each WFS
        print('Done!')


    def ComputePSD(self):
        if self.KL_ids is None:
            self.ComputeKLids()

        if self.exclude_modes is not None:
            self.KL_ids[:, self.exclude_modes] = 0.0

        #convert_factor = 2*np.pi/0.5e-6*1e-6  #[microns] -> [rad] at lambda=500 [nm]
        SPARTAunit2Microns = 35 # 35 microns per SPARTA unit
        Surface2WF = 2 # multiplicative factor to go from surface to WF
        A = SPARTAunit2Microns * Surface2WF#* convert_factor

        if not self.__use_GPU:
            print('Computing residual phase sreens from KL modes coefficients...')
            phase_screens = self.KL2phase @ self.KL_ids.T #* A 
            phase_screens_cube = phase_screens.reshape([240,240,phase_screens.shape[1]])
            print('Done!')

            print('Computing FFTs of each phase screen and spectra...')
            spectra = np.zeros([480, 480, phase_screens_cube.shape[2]], dtype=np.complex64)
            phase   = np.zeros([480, 480], dtype=np.complex64)
            # KL_pupil_pad = np.zeros([480, 480], dtype=cp.float32)
            # KL_pupil_pad[240//2:240//2+240, 240//2:240//2+240] = self.pupil

            for i in tqdm(range(phase_screens_cube.shape[2])):
                phase[240//2:240//2+240, 240//2:240//2+240] = phase_screens_cube[:,:,i] # zero padding
                # RMS = np.sqrt(np.sum(phase**2) / KL_pupil_pad.sum())
                # phase = phase / RMS * 100.0
                spectra[:,:,i] = np.abs(np.fft.fftshift( 1/480. * np.fft.fft2(np.fft.fftshift(phase)) ))
                
            print('Done!\nComputing PSD...')
            self.PSD = spectra.var(axis=2)
            del spectra, phase
            print('Done!')

        else: 
            KL2phase_buf = cp.array(self.KL2phase, dtype=cp.float32)
            batch_size = 500
            N = self.slopes[0].shape[0]

            print('Computing FFTs of each phase screen and spectra (on GPU)...')
            temp_mean          = cp.zeros([480, 480], dtype=cp.float32)
            FFT_batch          = cp.zeros([480, 480,  batch_size], dtype=cp.complex64)
            phase_padded_batch = cp.zeros([480, 480,  batch_size], dtype=cp.float32)
            variance_batches   = cp.zeros([480, 480,  N//batch_size], dtype=cp.float32)
            # KL_pupil_pad       = cp.zeros([480, 480], dtype=cp.float32)
            # KL_pupil_pad[240//2:240//2+240, 240//2:240//2+240] = cp.array(self.pupil)
            
            plan = get_fft_plan(phase_padded_batch, axes=(0,1), value_type='C2C') # for batched, C2C, 2D transform

            for i in tqdm(range(N//batch_size)):
                phase_padded_batch[240//2:240//2+240, 240//2:240//2+240] = \
                    (KL2phase_buf @ self.KL_ids[i*batch_size:(i+1)*batch_size,:].T).reshape([240,240,batch_size])
                
                # RMSs = cp.sqrt(cp.sum(phase_padded_batch**2, axis=(0,1)) / KL_pupil_pad.sum())
                # phase_padded_batch /= RMSs[None, None, ...]
                # phase_padded_batch *= 100.0 

                FFT_batch = cp.abs( gfft.fftshift(1/480.*gfft.fft2(gfft.fftshift(phase_padded_batch), axes=(0,1), plan=plan)) )
                temp_mean = FFT_batch.mean(axis=2, keepdims=True)
                variance_batches[:,:,i] = cp.sum( (FFT_batch-temp_mean)**2, axis=2 )

            self.PSD = cp.asnumpy( variance_batches.sum(axis=2) ) / N
            print('Done!')

            del phase_padded_batch, FFT_batch, variance_batches, temp_mean, plan, KL2phase_buf #, phase_batch
            self.mempool.free_all_blocks()
            self.pinned_mempool.free_all_blocks() # clear GPU memory
            
        return self.PSD
