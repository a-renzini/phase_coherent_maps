import numpy as np
import healpy as hp
import qpoint as qp
import quat_rotation as qr
import matplotlib.pyplot as plt
from numpy import cos,sin
import OverlapFunctsSrc as ofs
REARTH = 6378137.0
CLIGHT = 3.e8

def unit_vec(theta, phi):
	return np.array([sin(theta)*cos(phi),sin(theta)*sin(phi),cos(theta)] )


class Detector(object):
    
    def __init__(self, nside, dect_name):
        
        self._nside = nside
        lmax = int(nside/2)

        self.lmax = lmax
        self.name = dect_name
        
        self.Q = qp.QPoint(accuracy='low', fast_math=True, mean_aber=True)#, num_threads=1)
        
        # Configuration: radians and metres, Earth-centered frame
        if dect_name =='H1':
            
            self._lon = -2.08405676917
            self._lat = 0.81079526383
            self._elev = 142.554
            self._vec = np.array([-2.16141492636e+06, -3.83469517889e+06, 4.60035022664e+06])            
            self._alpha = (171.8)*np.pi/180.

        
        elif dect_name =='L1':
            self._lon = -1.58430937078
            self._lat = 0.53342313506           
            self._elev = -6.574
            self._vec = np.array([-7.42760447238e+04, -5.49628371971e+06, 3.22425701744e+06])
            
            self._alpha = (243.0)*np.pi/180.

      
        elif dect_name =='V1':
            self._lon = 0.18333805213
            self._lat = 0.76151183984
            self._elev = 51.884
            self._vec = np.array([4.54637409900e+06, 8.42989697626e+05, 4.37857696241e+06])
            
            self._alpha = 116.5*np.pi/180.         #np.radians()
        
        elif dect_name =='G':
            self._lon = 0.1710
            self._lat = 0.8326
            self._vec = np.array([4.2296e+6, 730394., 4.7178e+6])
            
            self._alpha = 68.8*np.pi/180.         #np.radians()
            
        elif dect_name =='K':
            self._lon = 2.39424267
            self._lat = 0.63268185
            self._vec = np.array([-3.7728e+6,3.4961e+6,3.77145e+6])
            
            self._alpha = 225.0*np.pi/180.         #np.radians()

        #######################
        
        else:
            dect_name = __import__(dect_name)
            #import name
            self._lon = dect_name.lon
            self._lat = dect_name.lat
            self._vec = dect_name.vec
            self._alpha = dect_name.alpha
        
        
        self._ph = self._lon + 2.*np.pi;
        self._th = self._lat + np.pi/2.
        
        self._alpha = np.pi/180.
        self._u = self.u_vec()
        self._v = self.v_vec()
        
        
        self.npix   = hp.nside2npix(self._nside)
        theta, phi  = hp.pix2ang(self._nside,np.arange(self.npix))
        self.Fplus  = self.Fplus(theta,phi)
        self.Fcross = self.Fcross(theta,phi)
        self.dott   = self.dott(self._vec)
        
    
    def u_(self):
        th = self._th
        ph = self._ph
        a = -cos(th)*cos(ph)
        b = -cos(th)*sin(ph)
        c = sin(th)
        norm = np.sqrt(a**2+b**2+c**2)
        return 1./norm * np.array([a,b,c])
        
    def v_(self):
        th = self._th
        ph = self._ph
        a = -sin(th)*sin(ph)
        b = sin(th)*cos(ph)
        c = 0.
        norm = np.sqrt(a**2+b**2+c**2)
        vec = np.array([a,b,c])
        if norm == 0.: 
            norm = 1.
        if self.name == 'E':
            vec = np.array([0.,-1.,0.])
        if self.name == 'F':
            vec = np.array([0.,1.,0.])
        return 1./norm * vec     
        

    def u_vec(self):
    
        a_p = self._alpha - np.pi/4.
        return self.u_()*cos(a_p) - self.v_()*sin(a_p)
        
    def v_vec(self):

        a_p = self._alpha - np.pi/4.
        return self.u_()*sin(a_p) + self.v_()*cos(a_p)
        

    def d_tens(self):
        return 0.5*(np.outer(self._u,self._u)-np.outer(self._v,self._v))   


    def Fplus(self, theta, phi):

        d_t = self.d_tens()
        res = 0
        i   = 0
        while i<3:
            j=0
            while j<3:
                res=res+d_t[i,j]*ofs.eplus(theta,phi,i,j)
                j=j+1
            i=i+1
            
        return res
    

    def Fcross(self, theta, phi): 
        
        d_t = self.d_tens()
        res = 0
        i   = 0
        while i<3:
            j=0
            while j<3:
                res=res+d_t[i,j]*ofs.ecross(theta,phi,i,j)
                j=j+1
            i=i+1
        
        return res


    def dott(self,x_vect):

        m      = hp.pix2ang(self._nside,np.arange(self.npix))
        m_vect = np.array(ofs.m(m[0], m[1])) #fits *my* convention: 0<th<pi, like for hp

        return np.dot(m_vect.T,x_vect)  #Rearth is in x_vect!


    def get_Fplus(self):
        return self.Fplus


    def get_Fcross(self):
        return self.Fcross    


    def get_dott(self):
        return self.dott




##################################


class Mapper(object):
    
    def __init__(self, config):
        
        ''' healpix numbers '''        
        self.nside_out = config.get_parameter('nside_out')
        self.npix_out  = hp.nside2npix(self.nside_out)
        self.nside_in  = config.get_parameter('nside_in')
        self.npix_in   = hp.nside2npix(self.nside_in)
        self.declabs   = config.get_parameter('dect_labels')
        self.tag       = config.get_parameter('tag')
        self.gen_flag  = config.get_parameter('map_gen_flag')
        self.input_dir = config.get_parameter('input_dir')
        self.N_freqs   = config.get_parameter('N_freqs')

        self.dects = np.array([])
        for d in self.declabs: 
            self.dects = np.append(self.dects,Detector(self.nside_in,d))
        self.ndet = len(self.dects)
        
        self.Qresp = self.Qresp_vec()
        self.Q     = qp.QPoint(accuracy='low', fast_math=True, mean_aber=True)#, num_threads=1)

        ''' if you don't already have input map file: '''
        if self.gen_flag ==True:
            self.Generate_Map_In()
            exit()
        ## while we are simulating:
        self.maps_in = self.Map_In()
            

    def Fshape(self, f, alpha = 3, f0 = 25.): #0.66666667
        return (f/f0)**((alpha-3.)/2.)
    

    def NoiseMatrices(self,noise):
        ''' noise should be a TDI chan XYZ complex vector on axis 0 '''
        nR = np.real(noise)
        nI = np.imag(noise)
        NR = np.einsum('i...,j...->ij...',nR,nR)
        NI = np.einsum('i...,j...->ij...',nI,nI)
        return NR, NI
   

    def MatrixProduct(self,n,d):
        ''' does the correct matrix product when -1 dim is freq '''
        return np.einsum('ik...,kj...->ij...',n,d)
   
    def Qresp_vec(self):
        ''' creates column of response functions '''
        
        Qpls = []
        Qcrs = []
        Qresp = []
        for i in range(self.ndet): 
            Qresp.append(np.ravel([self.dects[i].Fplus, self.dects[i].Fcross]))
        
        return np.array(Qresp)

    def geo_set(self, tstamp):
        '''returns the baseline pixel p and the boresight quaternion q_n'''

        nside = self.nside_in
        q_n = []
        n   = np.zeros(self.ndet, dtype = int)
        
        for i in range(self.ndet):
            q_n.append(self.Q.rotate_quat(self.Q.azel2bore(0., 90.0, None, None, np.degrees(self.dects[i]._lon), np.degrees(self.dects[i]._lat), tstamp)[0]))
            n[i] = self.Q.quat2pix(q_n[i], nside=nside, pol=True)[0]
        return q_n, n


    def rot_pix(self, m_array, n): 
        ''' Rotates string of pixels m around QUATERNION n '''

        nside                  = hp.npix2nside(len(m_array))
        dec_quatmap,ra_quatmap = hp.pix2ang(nside,m_array) #
        quatmap                = self.Q.radecpa2quat(np.rad2deg(ra_quatmap), np.rad2deg(dec_quatmap-np.pi*0.5), 0.*np.ones_like(ra_quatmap)) 
        quatmap_rotated        = np.ones_like(quatmap)
        
        i = 0
        while i < len(m_array): 
            quatmap_rotated[i] = qr.quat_mult(n,quatmap[i])
            i+=1
        quatmap_rot_pix = self.Q.quat2pix(quatmap_rotated,nside)[0] #rotated pixel list (polarizations are in [1])
        
        return quatmap_rot_pix
    

    def rot_Q(self, qresp, qn):
        #this can be translated with einsum
        
        rot_m_array = self.rot_pix(np.arange(self.npix_in), qn)
        
        #qresp shape = (+x, PIX)
        res = [qresp[0][rot_m_array], qresp[1][rot_m_array]]
        
        return res

    
    def Resp_Vec_ud(self, tstamp, freqs):
        ''' Full response function in correct format ''' 

        pix_ds = self.geo_set(tstamp)[1]
        q_ns   = self.geo_set(tstamp)[0]
        
        ''' exp factor '''
        theta, phi = hp.pix2ang(self.nside_in,np.arange(hp.nside2npix(self.nside_in)))
        pvecs  = REARTH * np.array(hp.pix2vec(self.nside_in, pix_ds)).T
        kl_val = freqs/CLIGHT
        k_unit = unit_vec(theta,phi).T
        
        '''shape(scalar) = (XYZ dect,PIXEL)'''
        scalar = np.einsum('Dx,px->Dp',pvecs,k_unit)
        arg    = np.einsum('Dp,f->Dpf' , scalar,kl_val)

        Fdep   = self.Fshape(freqs)
        expf   = np.einsum('Xpf,f->Xpf', np.exp(-2.j * np.pi * arg), Fdep)

        Qr_vec = np.zeros((2*self.ndet, len(freqs), 2*self.npix_out))

        for i in range(self.ndet):
            Qrexp = np.einsum('pf, Pp -> Ppf', expf[i], np.split(self.Qresp[i],2))

            res1  = []
            for idxf in range(len(freqs)):
                res1.append(np.array(hp.ud_grade(self.rot_Q(Qrexp[:,:,idxf], q_ns[i]), nside_out = self.nside_out)))
            res1  = np.swapaxes(np.array(res1), 0, 1)
            res2  = np.hstack((res1[0], res1[1])) # 0 == +, 1 == x
            Qr_vec[i] = np.real(res2)
            Qr_vec[i+self.ndet] = np.imag(res2)

        Qresp = Qr_vec
        return Qresp


    def Resp_Vec(self, tstamp, freqs):
        ''' Full response function in correct format ''' 

        pix_ds = self.geo_set(tstamp)[1]
        q_ns   = self.geo_set(tstamp)[0]
        
        ''' exp factor '''
        theta, phi = hp.pix2ang(self.nside_in,np.arange(hp.nside2npix(self.nside_in)))
        pvecs  = REARTH * np.array(hp.pix2vec(self.nside_in, pix_ds)).T
        kl_val = freqs/CLIGHT
        k_unit = unit_vec(theta,phi).T
        
        '''shape(scalar) = (XYZ dect,PIXEL)'''
        scalar = np.einsum('Dx,px->Dp',pvecs,k_unit)
        arg    = np.einsum('Dp,f->Dpf' , scalar,kl_val)

        Fdep   = self.Fshape(freqs)
        expf   = np.einsum('Xpf,f->Xpf', np.exp(-2.j * np.pi * arg), Fdep)

        Qr_vec = np.zeros((self.ndet, 2*self.npix_in, len(freqs)), dtype = complex)

        for i in range(self.ndet):
            Qrexp = np.einsum('pf, Pp -> Ppf', expf[i], np.split(self.Qresp[i],2))
            res1  = np.array(self.rot_Q(Qrexp, q_ns[i]))
            res2  = np.vstack((res1[0], res1[1]))
            Qr_vec[i]  = res2

        Qresp = np.swapaxes(Qr_vec, -1,-2)   #bring pix to last idx
        return Qresp


    def Faux_Data(self, tstamp, freqs):
        ''' Should return data for LHV... for a bkd '''
        
        delf = freqs[1] - freqs[0]

        hp_in, hc_in = self.maps_in
        h_maps       = np.hstack([hp_in, hc_in])

        ## no freq phase use:
        #h_maps       = np.ravel([hp_in, hc_in])
        #h_maps       = len(freqs)*[h_maps] 

        # add spectral shape use:
        # nonflat_h = []
        # for idx in range(len(freqs)):
        #     nonflat_h.append(Fdep[idx]*h_maps)

        Qresp = self.Resp_Vec(tstamp, freqs)
        res2  = np.einsum('Dfp,fp-> Df', Qresp, h_maps) #sums over pix
        res   = delf * 4. * np.pi / self.npix_in * res2
        return res
        

    def Operators(self, freqs, tstamp, data):
        ''' inherit beam from Response using tstamp '''

        Qresp_ud = self.Resp_Vec_ud(tstamp, freqs)
        # Qresp.shape = (2D, f, 2pix)
        # data.shape  = (2D, f)
        Noise_inv = np.array(len(freqs)*[list(np.diag(np.ones(2*self.ndet)))]).T
        #Noise_inv.shape = (2D, 2D, f)
        delf = freqs[1]-freqs[0]
        Data = np.vstack((data.real, data.imag))

        '''projector'''
        res1 = np.einsum('DE...,E...->D...',Noise_inv, Data)
        res2 = np.einsum('Df...,Df-> ...', Qresp_ud, res1)
        proj_1 = delf*4.*np.pi/self.npix_out*res2

        Data_ImRe = np.vstack((data.imag,-data.real))
        res12 = np.einsum('DE...,E...->D...',Noise_inv,Data_ImRe)
        res22 = np.einsum('Df...,Df-> ...', Qresp_ud, res12)

        proj_2 = delf*4.*np.pi/self.npix_out*res22
        
        proj = np.concatenate([proj_1,proj_2], axis = 0)

        #does matrix product AND sums over frequencies
        
        '''beam pattern'''
        res3 = np.einsum('DEf, Ef... -> Df...', Noise_inv,Qresp_ud)
        res4 = np.einsum('Dfp, Dfq -> pq', Qresp_ud,res3)
        
        beam_diag = (delf*4.*np.pi/self.npix_out)**2*res4    #+res4.T)
        Qresp_ImRe = np.concatenate([Qresp_ud[self.ndet:],-Qresp_ud[:self.ndet]],axis = 0)
        res32 = np.einsum('DEf, Ef... -> Df...', Noise_inv,Qresp_ImRe)
        res42 = np.einsum('Dfp, Dfq -> pq', Qresp_ud,res32)
        
        beam_od = (delf*4.*np.pi/self.npix_out)**2*res42
        #np.zeros_like(beam_diag) 
        beam = np.array([[beam_diag,-beam_od],[beam_od,beam_diag]])
        beam_out = []

        for i in range(len(beam[0][0])):
            beam_out.append(np.concatenate([beam[0][0][i],beam[0][1][i]]))
    
        for i in range(len(beam[0][0])):
            beam_out.append(np.concatenate([beam[1][0][i],beam[1][1][i]]))
        
        #beam = np.reshape(beam, (2*2*self.npix_out,2*2*self.npix_out))
        return proj, np.array(beam_out) 
    

    #def Map_Maker(self,tstamps,noise_stream,data_stream,freqs):
    #    ''' to parallelise! '''
    #    print('constructing operators...')
    #    Dirty_Map = 0.
    #    Beam = 0.
    #    i = 0
    #    for (tstamp,noise,data) in zip(tstamps,noise_stream,data_stream):
    #        print(tstamp)
    #        dmap, beam = self.Operators(tstamp,noise,data,freqs)
    #        Dirty_Map += dmap
    #        Beam += beam
    #        i+=1
    #    return Dirty_Map, Beam
        
    def Inverse_Beam(self, Beam, cond = 1.e-6):
        ''' _ '''
        print('Inverting Beam')
        #'Beam = np.swapaxes(Beam,1,2).reshape(3*self.npix_out,3*self.npix_out)'
        
        print('cond:', np.linalg.cond(Beam))
        Beam_inv = np.linalg.pinv(Beam, rcond = cond)
        #'Beam_inv = np.swapaxes(Beam_inv.reshape(3,self.npix_out,3,self.npix_out),1,2)'
        
        return Beam_inv
    
    def Map(self, Dmap, Beam):
        ''' Dirty_Map: (4pix) '''
        ''' Beam: (4pix x 4pix) '''

        Beam_inv = self.Inverse_Beam(Beam)
        res1 = np.einsum('pq,q->p', Beam_inv, Dmap)
        res2 = np.split(res1, 4)

        return res2[0]+1.j*res2[2], res2[1]+1.j*res2[3]


    def Generate_Map_In(self, lmax = 8):
        ''' Synfast random map '''

        const = 1.
        Cl = []
        for i in range(1,lmax):
            Cl.append(const/(i)**2)
        Cl = np.array(Cl)
        
        
        file = np.load('%s/hp_hc_in_ns16_GB_smooth.npz' % (self.input_dir))
        hp_in = file['hp_in']
        hc_in = file['hc_in']

        #hp_in = hp.sphtfunc.synfast(Cl/2.,self.nside_in, verbose = False)
        #hc_in = hp.sphtfunc.synfast(Cl/2.,self.nside_in, verbose = False)

        #alm = np.zeros(hp.Alm.getidx(lmax,lmax,lmax)+1,dtype=np.complex)
        #idx = hp.Alm.getidx(lmax,1,0)
        #alm[idx] = (1.+ 0.j)
        #hp_in = hp.alm2map(alm,nside = self.nside_in,verbose = False)
        #hp_in += -np.min(hp_in)
        #hc_in = np.copy(hp_in)

        ph1 = np.pi*2*np.random.rand(self.N_freqs, len(hp_in))
        ph2 = np.pi*2*np.random.rand(self.N_freqs, len(hp_in))        

        hp_in = hp_in*np.exp(1.j*ph1)
        hc_in = hc_in*np.exp(1.j*ph2)

        np.savez('hp_hc_in_ns%s_%s.npz' % (self.nside_in, self.tag), hp_in = hp_in, hc_in = hc_in)
        
        return 0


    def Map_In(self):
        try: 
            file = np.load('%s/hp_hc_in_ns%s_%s.npz' % (self.input_dir, self.nside_in, self.tag))
            hp_in = file['hp_in']
            hc_in = file['hc_in']
            return hp.ud_grade(hp_in, nside_out = self.nside_in), hp.ud_grade(hc_in, nside_out = self.nside_in)
        
        except FileNotFoundError:
            file = np.load('%s/hp_hc_in_ns%s_%s.npz' % (self.input_dir, 16, self.tag))
            hp_in = file['hp_in']
            hc_in = file['hc_in']
            return hp.ud_grade(hp_in, nside_out = self.nside_in), hp.ud_grade(hc_in, nside_out = self.nside_in)
        
               
               
    def Faux_Noise(self,freqs,Nsegs):
        
        varXX = Ln.compute_strain_sens_XX(freqs)
        varXY = Ln.compute_strain_sens_XY(freqs) 
        
        #rands = [np.random.normal(loc = 0., scale = 1. , size = len(freqs)),np.random.normal(loc = 0., scale = 1. , size = len(freqs))] 
        fakenoise = Ln.correlated_generation(varXX, varXY, Nsegs)                 #rands[0]+1.j*rands[1]

        return np.array(fakenoise)*1.e-3 

    def Noise_from_PSD(self,freqs):
        
        varXX = Ln.compute_strain_sens_XX(freqs)
        varXY = Ln.compute_strain_sens_XY(freqs) 
        
        #sign of the noise?
        
        fakenoise = np.array(Ln.corr_N_from_PSD(varXX, varXY))
        
        #input flat noise:
        #fakenoise = 1.e-46*np.array(len(freqs)*[[1.+1.j,1.+1.j,1.+1.j]])
        
        return fakenoise 
