import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

file = np.load('GBmap_ns32.npz')
map  = file['map']
#map  = np.log(1.+map)

hp.mollview(map)
plt.show()

map = hp.ud_grade(map, nside_out = 16)

ph1 = np.pi*2*np.random.rand(len(map))
ph2 = np.pi*2*np.random.rand(len(map))
        
hpl  = map/np.max(map)
hcr  = map/np.max(map)

ph1 = 0.
ph2 = 0.

hpl = hpl*np.exp(1.j*ph1)
hcr = hcr*np.exp(1.j*ph2)

hp.mollview(hpl.real)
plt.show()

hp.mollview(hpl.imag)
plt.show()

np.savez('hp_hc_in_ns%s_%s.npz' % (16, 'GB_np'), hp_in = hpl, hc_in = hcr)

exit()

