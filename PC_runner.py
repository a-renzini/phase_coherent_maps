import numpy as np
import scipy as sc
import healpy as hp
import matplotlib.pyplot as plt
import PCMAPS_class as PCM

from config import Configuration
from numpy.linalg import norm
from math import log10, floor

import os
import sys
import re

from mpi4py import MPI
ISMPI = True

if ISMPI:
    comm  = MPI.COMM_WORLD
    nproc = comm.Get_size()
    my_id = comm.Get_rank()
    if my_id ==0: print('total number of processors:', nproc, flush = True)

else:
    comm  = None
    nproc = 1
    my_id = 0

'''Set of parameters in dictionnary format'''
nside_in  = 16
nside_out = 4
npix_in   = hp.nside2npix(nside_in)
npix_out  = hp.nside2npix(nside_out)


out_tag      = sys.argv[1]
namefile_out = 'map_' + out_tag + 'HL_ns_' + str(nside_out)
map_gen_flag = True

CONFIG_PARAMS = {'nside_in': nside_in,
                            'nside_out': nside_out,
                            'dect_labels': ['H1','L1'], # 'V1', 'K'],
                            'tag': out_tag,
                            'map_gen_flag': map_gen_flag}
                            
                            
''' Time array 1 day sampled every minute '''
TIME = np.arange(0, 24*3600, 60) #172800,

'''Instance of a configuration class'''
CONFIG = Configuration.type(CONFIG_PARAMS)

import healpy as hp

freqs = np.linspace(80,300,200)

mapmake = PCM.Mapper(CONFIG)


if my_id == 0:
    Dmap_buff = np.zeros(4*npix_out)
    Beam_buff = np.zeros((4*npix_out,4*npix_out))
    Dmap      = 0
    Beam      = 0

else:
    Dmap_buff = None
    Beam_buff = None
    Dmap      = None
    Beam      = None

#data_in   = []
#noise_mod = []
counter       = 0
counts        = int(len(TIME)/nproc)
t_stamp_sets  = np.split(TIME, counts)

if my_id == 0: print('Full run = ', counts, 'counts')

for stamps in t_stamp_sets:
    
    if my_id == 0: 
        print('counter:', counter, '/', counts) 
        counter +=1

    my_stamp   = stamps[my_id]
    my_data_in = mapmake.Faux_Data(my_stamp,freqs)
    # noise.append([mapmake.Faux_Noise(freqs),resp.Faux_Noise(freqs),resp.Faux_Noise(freqs)])
    # plt.plot(np.abs(noise[0][0]*noise[0][1]))
    # plt.savefig('fakenoise.png')
    # plt.close()
    # plt.plot(np.abs(c[0][0][0]))
    # plt.savefig('fakedata.png')
    # exit()
    if my_id == 0: print('constructing Dmap and Beam...', flush = True)

    my_Dmap, my_Beam = mapmake.Operators(freqs, my_stamp, my_data_in)
    
    if ISMPI:    
        comm.barrier()
        comm.Reduce(my_Dmap, Dmap_buff, root = 0, op = MPI.SUM) 
        comm.Reduce([np.ascontiguousarray(my_Beam),  MPI.DOUBLE], [np.ascontiguousarray(Beam_buff),  MPI.DOUBLE], root = 0, op = MPI.SUM) 

        if my_id == 0:
            print('accumulating Dmap, Beam...', flush = True)
            Dmap += Dmap_buff
            Beam += Beam_buff

    else:
        Dmap += my_Dmap
        Beam += my_Beam

if my_id == 0:
    h_plus, h_cross = mapmake.Map(Dmap, Beam)
    print('We have output map!', flush = True)


    np.savez(namefile_out,  h_plus = h_plus, h_cross = h_cross, Dmap = Dmap, Beam = Beam)
    print(namefile_out , 'saved', flush = True)

    
    import matplotlib as m
    m.rcParams.update({'font.size':26})
    from matplotlib import cm
    cs = cm.RdYlBu_r
    hp.mollview(h_plus*np.conj(h_plus) + h_cross*np.conj(h_cross), title = '', cmap = cs)
    plt.show()
