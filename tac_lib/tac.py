# %%
from .common_imports import *
# %%
class TiledApertureBeamProp:

    def __init__(self,im_size,pix_size,n_channel,n_screens,Kvar,Z,trans_pn,atm,amp_v,g_amp,ra,d_):

        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        print(f"Using {self.device} device")

        self.im_size = im_size
        self.pix_size = pix_size
        self.n_channel = n_channel
        self.n_screens = n_screens
        self.Kvar = Kvar
        self.Z = Z
        self.trans_pn = trans_pn
        self.atm = atm
        self.amp_v = amp_v
        self.g_amp = g_amp
        self.ra = ra
        self.d_ = d_
        self.kernel = True


    def Transverse_ph_abbrZP(self,abr,th,rh,R):
        """
        Generate aberrations using Zernike coefficients.

        Parameters:
        - abr (list or numpy array): Zernike coefficients representing aberrations.
        - th (float or numpy array): Angular coordinate(s) in radians.
        - rh (float or numpy array): Radial coordinate(s) in the range [0, R].
        - R (float): Aperture radius.

        Returns:
        - AbPh (torch tensor): Transverse phase aberrations as complex numbers.

        This function computes transverse phase aberrations using Zernike polynomials
        based on the provided Zernike coefficients (abr). The aberrations are generated
        for given angular (th) and radial (rh) coordinates within the aperture defined
        by the radius (R). The output is returned as a torch tensor of complex numbers
        representing the phase aberrations.
        """
        rho = T.tensor(rh/R).to(self.device)
        th = T.tensor(th).to(self.device)
        abr = T.tensor(abr).to(self.device)
        ZP1 = 2*rho*T.sin(th)
        ZP2 = 2*rho*T.cos(th)

        ZP3 = np.sqrt(3)*(2*(rho**2)-1)

        ZP4 = np.sqrt(6)*(rho**2*T.sin(2*th))
        ZP5 = np.sqrt(6)*(rho**2*T.cos(2*th))

        ZP6 = np.sqrt(8)*((3*rho**3 - 2*rho)*T.sin(th))
        ZP7 = np.sqrt(8)*((3*rho**3 - 2*rho)*T.cos(th))

        ZP8 = np.sqrt(8)*(rho**3*T.sin(3*th))
        ZP9 = np.sqrt(8)*(rho**3*T.cos(3*th))

        ZP10 = np.sqrt(5)*(6*(rho**4)-6*(rho**2)+1)

        ZPt = abr[0]*ZP1 + abr[1]*ZP2 + abr[2]*ZP3 + abr[3]*ZP4 +\
            abr[4]*ZP5 + abr[5]*ZP6 + abr[6]*ZP7 + abr[7]*ZP8 +\
                    abr[8]*ZP9 + abr[9]*ZP10
        ZPt[rho>1] = 0
        AbPh = T.exp(1j*2*np.pi*ZPt)
        return AbPh
    
    def cart2pol(self,x, y):
        """
        Convert Cartesian coordinates to polar coordinates.

        Parameters:
        - x (float or numpy array): x-coordinate(s) in Cartesian space.
        - y (float or numpy array): y-coordinate(s) in Cartesian space.

        Returns:
        - phi (numpy array): Angular coordinate(s) in radians.
        - rho (numpy array): Radial coordinate(s).

        This function converts Cartesian coordinates (x, y) to polar coordinates
        (phi, rho). The angular coordinate phi represents the angle in radians
        measured from the positive x-axis, and the radial coordinate rho represents
        the distance from the origin to the point in Cartesian space.
        """
        rho = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        return phi, rho
    
    def GaussBeamNDefPsLw(self,lambda_,w0,Rc,X0,Y0,Dxy,NP,m,zmm):
        
        """
        Generate a Gaussian beam with a defined phase and waist.

        Parameters:
        - lambda_ (float): Wavelength of the laser beam.
        - w0 (float): Waist radius of the Gaussian beam.
        - Rc (float): Radius of curvature of the wavefront.
        - X0 (float): X-coordinate offset in the beam's transverse plane.
        - Y0 (float): Y-coordinate offset in the beam's transverse plane.
        - Dxy (float): Size of the transverse plane.
        - NP (int): Number of points for discretization.
        - m (int): Mode of the beam.
        - zmm (float): Position of the beam along the optical axis.

        Returns:
        - uout (torch tensor): Complex amplitude distribution of the Gaussian beam.

        This function generates a Gaussian beam with a defined phase and waist. It calculates
        the complex amplitude distribution of the beam based on the given parameters:
        wavelength (lambda_), waist radius (w0), radius of curvature (Rc), transverse offsets
        (X0, Y0), transverse plane size (Dxy), number of points for discretization (NP),
        beam mode (m), and position along the optical axis (zmm). The output is returned as
        a torch tensor representing the complex amplitude distribution of the Gaussian beam.
        """

        k = 2*np.pi/lambda_
        zR = np.pi*w0**2/lambda_
        z = zmm*1000

        M = NP
        dx = Dxy/M
        x = T.arange((-M/2 - X0)*dx, (M/2 - X0)*dx,dx).to(self.device)

        N=NP
        dy = Dxy/N
        y = T.arange((-N/2 - Y0)*dy, (N/2 - Y0)*dy,dy).to(self.device)

        [X,Y] = T.meshgrid(x,y,indexing='xy')


        R = T.sqrt(X**2+Y**2)

        P = 0.01
        A = (2*P)/(np.pi*(w0*1e-3)**2)
        C = np.sqrt(A)*T.exp(1j*k*(R**2)/(2*Rc)).to(self.device)

        Psi = C*T.exp(-(R**2)/(w0**2))#*T.exp(1j*m*phy)
        uout = Psi
        return uout
    
    def CirAperN(self,X1,Y1,Ra,Dx,NP):
        """
        Generate a circular aperture in a transverse plane.

        Parameters:
        - X1 (float): X-coordinate center of the circular aperture.
        - Y1 (float): Y-coordinate center of the circular aperture.
        - Ra (float): Radius of the circular aperture.
        - Dx (float): Size of the transverse plane.
        - NP (int): Number of points for discretization.

        Returns:
        - A (torch tensor): Binary array representing the circular aperture.

        This function generates a circular aperture in a transverse plane with specified
        parameters: center coordinates (X1, Y1), radius (Ra), transverse plane size (Dx),
        and number of points for discretization (NP). The output is returned as a torch tensor
        representing a binary array, where 1 indicates points inside the aperture and 0 indicates
        points outside the aperture.
        """
        M = NP
        dx = Dx/M
        x = T.arange((-M/2 - X1)*dx, (M/2 - X1)*dx,dx)[0:NP].to(self.device) #TODO:check later there was a size mismatch
        N=NP
        dy = Dx/N
        y = T.arange((-N/2 - Y1)*dy, (N/2 - Y1)*dy,dy)[0:NP].to(self.device)
        [X,Y] = T.meshgrid(x,y,indexing='xy')
        s = time.time()
        A = T.ones(X.shape,dtype=T.int16).to(self.device)
        e = time.time()-s
        r = Ra
        R = T.sqrt(X**2+Y**2)
        A[R>r] = 0
        return A
    
    def rectangularPulse(self,start,stop,x):
        """
        Generate a rectangular pulse waveform.

        Parameters:
        - start (float): Start time of the pulse.
        - stop (float): Stop time of the pulse.
        - x (numpy array): Time array at which to evaluate the pulse.

        Returns:
        - pulse (numpy array): Rectangular pulse waveform.

        This function generates a rectangular pulse waveform with specified start and stop
        times. The pulse waveform is evaluated at the provided time array (x), and the output
        is returned as a numpy array representing the rectangular pulse waveform.
        """
        pulse = []
        for i in x:
            if start<i and stop>i:
                val = 1
            elif i==stop or i==start:
                val = 0.5
            else:
                val = 0
            pulse.append(val)
        pulse = np.array(pulse)
        return pulse
    
    @staticmethod
    @njit(parallel=True,fastmath=True)
    def get_gamma(len_alpha,len_beta,alpha,beta):
        gamma = np.zeros((len_alpha,len_beta))
        for j in prange(len_beta):
            for i in prange(len_alpha):
                if alpha[i]**2 + beta[j]**2 > 1:
                    gamma[j,i] = 0
                else:
                    gamma[j,i] = np.sqrt(1-alpha[i]**2-beta[j]**2)
        return gamma

    @staticmethod
    @njit(parallel=True,fastmath=True)
    def getFs(FxBL,FxBLr,FyBL,FyBLc,c,r):
        for i in prange(c):
            FxBL[i,0:r] = FxBLr
        for j in prange(r):
            FyBL[0:c,j] = FyBLc
        return FxBL,FyBL
    
    def PropAngSpecBandLimF(self,uin,L,Dx,Dy,zmm):
        """
        Propagate a wavefront using the bandlimited angular spectrum method.

        Parameters:
        - uin (torch tensor): Input wavefront.
        - L (float): Wavelength of the wavefront in millimeters.
        - Dx (float): Width of the input wavefront in millimeters.
        - Dy (float): Height of the input wavefront in millimeters.
        - zmm (float): Propagation distance in millimeters.

        Returns:
        - uout (torch tensor): Output wavefront after propagation.

        This function propagates a wavefront using the bandlimited angular spectrum method,
        which calculates the wavefront's Fourier transform and applies a transfer function
        to account for diffraction effects during propagation. The input parameters include
        the input wavefront (uin), wavelength (L), width (Dx) and height (Dy) of the input
        wavefront, and the propagation distance (zmm). The output is the wavefront after
        propagation, returned as a torch tensor.
        """
        layer = uin
        if self.kernel : ## This will run once and save the kernel
            lambda_ = L*1e-3
            k = 2*np.pi/lambda_
            z = zmm*1e-3
            phy_x = Dx*1e-3
            phy_y = Dy*1e-3

            obj_size = layer.shape
            r,c = layer.shape[0], layer.shape[1]
            Fs_x = obj_size[1]/phy_x
            Fs_y = obj_size[0]/phy_y

            dFx = Fs_x/obj_size[1]
            dFy = Fs_y/obj_size[0]

            Fx = np.arange(-Fs_x/2,Fs_x/2,dFx)
            Fy = np.arange(-Fs_y/2,Fs_y/2,dFy)

            alpha = lambda_*Fx
            beta = lambda_*Fy
            len_alpha = len(alpha)
            len_beta = len(beta)

            gamma = self.get_gamma(len_alpha,len_beta,alpha,beta)


            ival = T.tensor(k*gamma*z).to(self.device)
            H0 = T.exp(1j*ival)

            Fxlim = 1/(np.sqrt(1+(2*dFx*z)**2)*lambda_)
            Fylim = 1/(np.sqrt(1+(2*dFy*z)**2)*lambda_)

            FxBL = np.zeros((len(beta),len(alpha)))
            FyBL = np.zeros((len(beta), len(beta)))
            FxBLr = self.rectangularPulse(-Fxlim,Fxlim,Fx)
            FyBLc = self.rectangularPulse(-Fxlim,Fylim,Fx)

            FxBL,FyBL = self.getFs(FxBL,FxBLr,FyBL,FyBLc,c,r)

            FxBL = T.tensor(FxBL).to(self.device)
            FyBL = T.tensor(FyBL).to(self.device)
            H1 = H0*FxBL*FyBL
            self.H1 = H1
            self.kernel = False
        uout = ifft2(ifftshift((fftshift(fft2(layer)))*self.H1))

        return uout
    
    def SphLens(self,uin,L1,L2,NP,lambda_,zf,n=1.55,th=0):
        #Units: mm
        """
        Simulate the propagation of a wavefront through a spherical lens.

        Parameters:
        - uin (torch tensor): Input wavefront.
        - L1 (float): Length of the input plane in millimeters.
        - L2 (float): Length of the output plane in millimeters.
        - NP (int): Number of points for discretization.
        - lambda_ (float): Wavelength of the wavefront in millimeters.
        - zf (float): Focal length of the lens in millimeters.
        - n (float): Refractive index of the lens. Default is 1.55.
        - th (float) : Thickness of the lens in mm. Default is 0. Thin Lens approximation.

        Returns:
        - uout (torch tensor): Output wavefront after passing through the lens.

        This function simulates the propagation of a wavefront through a spherical lens.
        It applies a phase shift to the input wavefront based on the lens properties
        (focal length and aperture size) to simulate the lens effect. The input parameters
        include the input wavefront (uin), length of the input and output planes (L1 and L2),
        number of points for discretization (NP), wavelength of the wavefront (lambda_), and
        focal length of the lens (zf). The output is the wavefront after passing through
        the lens, returned as a torch tensor.
        #TODO: Take the refractive index of the lens as input too (for small angle approximations for now)
        """
        k=2*np.pi/lambda_

        dy=L1/NP
        y=T.arange(-L1/2,L1/2,dy).to(self.device)

        dx=L2/NP
        x=T.arange(-L2/2,L2/2,dx).to(self.device)

        X,Y=T.meshgrid(x,y,indexing='xy')
        uout=T.multiply(uin,T.exp(-1j*(k/(2*zf))*(X**2+Y**2)))#,T.exp(1j*k*n*th))
        return uout

    def sourceTAC_final(self,lambda_,w0,Ra,a,NL,Dx,NP,mDr,zm,m,Rc,phNs,Kvar,trans_pn,amp_v,g_amp, n_chan):

        """
        Simulate a source array with transverse angular diversity for coherent beam combining.

        Parameters:
        - lambda_ (float): Wavelength of the laser beam in millimeters.
        - w0 (float): Waist radius of the Gaussian beam.
        - Ra (float): Radius of the circular aperture.
        - a (float): Distance between adjacent beams in the source array in millimeters.
        - NL (int): Number of levels in the source array.
        - Dx (float): Size of the transverse plane in millimeters.
        - NP (int): Number of points for discretization in the transverse plane.
        - mDr (float): Maximum transverse angle diversity in degrees.
        - zm (float): Position of the beam along the optical axis in millimeters.
        - m (int): Mode of the beam.
        - Rc (float): Radius of curvature of the wavefront.
        - phNs (list): List of phase noise coefficients.
        - Kvar (float): Variance of the phase noise.
        - trans_pn (bool): Flag indicating whether to apply transverse phase noise.
        - amp_v (bool): Flag indicating whether to apply amplitude variation.
        - g_amp (float): Amplitude variation factor.
        - n_chan (int): Number of channels.

        Returns:
        - uc0 (numpy array): Initial complex amplitude distribution of the source array.
        - coord (numpy array): Coordinates of the source array elements.

        This function simulates a source array with transverse angular diversity for
        coherent beam combining. It generates an initial complex amplitude distribution
        (uc0) representing the source array and calculates the coordinates (coord) of
        the source array elements. The input parameters include the wavelength of the
        laser beam (lambda_), waist radius of the Gaussian beam (w0), radius of the
        circular aperture (Ra), distance between adjacent beams in the source array (a),
        number of levels in the source array (NL), size of the transverse plane (Dx),
        number of points for discretization in the transverse plane (NP), maximum transverse
        angle diversity (mDr), position of the beam along the optical axis (zm), mode of
        the beam (m), radius of curvature of the wavefront (Rc), list of phase noise
        coefficients (phNs), variance of the phase noise (Kvar), flag indicating whether
        to apply transverse phase noise (trans_pn), flag indicating whether to apply amplitude
        variation (amp_v), amplitude variation factor (g_amp), and number of channels (n_chan).
        """
        k = 2*np.pi/lambda_
        xf = a/2
        yf = round(np.sqrt(3)*a/2)

        theta = mDr
        RN1 = np.random.randint(low=0,high=100,size=(36,))
        RN2 = np.random.randint(low=0,high=100,size=(36,))
        theta_rx = np.radians(theta*RN1/100)
        theta_ry = np.radians(theta*RN2/100)
        kx = k*np.sin(theta_rx)
        ky = k*np.sin(theta_ry)

        mnE = 2*NL+1

        zmm = zm*1000

        U = T.zeros(NP,NP, dtype=T.cfloat).to(self.device)
        X0 = np.arange(-2*NL,2*NL+2,2)*xf
        Y0 = 0*yf

        phNs1 = T.tensor(phNs)
        n_cn = 0

        for r in range(mnE):
            u0 = self.GaussBeamNDefPsLw(lambda_,w0,Rc,X0[r],Y0,Dx,NP,m,zmm)
            c0 = self.CirAperN(X0[r],Y0,Ra,Dx,NP)

            UC0 = np.sqrt(g_amp)*u0
            Uel_1 = T.multiply(T.exp(1j*phNs1[n_cn]), UC0).to(self.device)
            E_1i = T.real(Uel_1) + T.tensor(amp_v*T.randn(1)).to(self.device)
            E_1q = T.imag(Uel_1) + T.tensor(amp_v*T.randn(1)).to(self.device)
            E_1 = T.complex(E_1i,E_1q).to(self.device)
            Uel = E_1*c0
            U += Uel
            n_cn += 1

        p=1
        while p <= NL:
            X1 = np.arange(-2*NL+p,2*NL-p+2,2)*xf
            Y1 = p*yf
            for q in range(mnE-p):

                u1p = self.GaussBeamNDefPsLw(lambda_,w0,Rc,X1[q],Y1,Dx,NP,m,zmm)
                c1p = self.CirAperN(X1[q],Y1,Ra,Dx,NP)

                uc1p = np.sqrt(g_amp)*u1p
                Uel_1 = T.multiply(T.exp(1j*phNs1[n_cn]), uc1p).to(self.device)
                E_1i = T.real(Uel_1) + T.tensor(amp_v*np.random.randn(1)).to(self.device)
                E_1q = T.imag(Uel_1) + T.tensor(amp_v*np.random.randn(1)).to(self.device)
                E_1 = T.complex(E_1i,E_1q).to(self.device)
                Uel = E_1*c1p

                U += Uel
                n_cn += 1

                u1m = self.GaussBeamNDefPsLw(lambda_,w0,Rc,X1[q],-Y1,Dx,NP,m,zmm)
                c1m = self.CirAperN(X1[q],-Y1,Ra,Dx,NP)

                uc1m = np.sqrt(g_amp)*u1m
                Uel_1 = T.multiply(T.exp(1j*phNs1[n_cn]),uc1m).to(self.device)
                E_1i = T.real(Uel_1) + T.tensor(amp_v*np.random.randn(1)).to(self.device)
                E_1q = T.imag(Uel_1) + T.tensor(amp_v*np.random.randn(1)).to(self.device)
                E_1 = T.complex(E_1i,E_1q).to(self.device)
                Uel = E_1*c1m

                U += Uel
                n_cn += 1
            p += 1
        return U
    
    def TiledAperture_2(self, p_n,atm=None):
        """
        Simulate coherent beam combining with a tiled aperture.

        Returns:
        - U_ (numpy array): Initial complex amplitude distribution of the source array.
        - Up (torch tensor): Final complex amplitude distribution after propagation.
        - coord (numpy array): Coordinates of the source array elements.

        This function simulates coherent beam combining with a tiled aperture. It generates
        an initial complex amplitude distribution (U_) representing the source array, propagates
        it through atmospheric screens if provided, and computes the final complex amplitude
        distribution (Up) after propagation. The input parameters include the size of the image
        plane (im_size), pixel size (pix_size), number of channels (n_channel), phase noise coefficients
        (p_n), number of atmospheric screens (n_screens), variance of the phase noise (Kvar), propagation
        distance (Z), flag indicating whether to apply transverse phase noise (trans_pn), atmospheric
        phase screens (atm), flag indicating whether to apply amplitude variation (amp_v), amplitude
        variation factor (g_amp), radius of the aperture (ra), and distance between adjacent apertures (d_).
        The function returns the initial complex amplitude distribution (U_), final complex amplitude
        distribution after propagation (Up), and coordinates of the source array elements (coord).
        """
        lambda_ = 1.064*1e-3
        Rc=1e15
        NP=self.im_size
        Dx=self.pix_size*NP
        m=0
        Ra= (self.ra/2)*1e3
        D=self.d_*1e3
        a=D*NP/Dx
        mDr=0
        w=0.85*Ra

        if self.n_channel == 7:
            NL=1
        elif self.n_channel == 19:
            NL = 2
        elif self.n_channel == 37:
            NL = 3
        elif self.n_channel == 61:
            NL = 4
        elif self.n_channel == 91:
            NL = 5
        elif self.n_channel == 127:
            NL = 6
        elif self.n_channel==217:
            NL=8
        else:
            ValueError('Please provide correct channel number..')

        zmm = self.Z*1e3
        zm = zmm*1e-3
        phNs = p_n
        n_step = self.n_screens
        zsmm = zmm/(n_step+1)
        w0 = w

        U = self.sourceTAC_final(lambda_,w0,Ra,a,NL,Dx,NP,mDr,zm,m,Rc,phNs,self.Kvar,self.trans_pn,\
                                       self.amp_v,self.g_amp,self.n_channel)
        U_ = U.cpu().numpy()
        # pib_ = PIB(np.abs(U_)**2,1024,1024,1023,pix_size)
        # print('Input Power: ', np.real(pib_))

        phy_x = Dx
        phy_y = Dx

        if self.atm is None:
            z_prop = zmm
        else:
            z_prop = zsmm
        Up = self.PropAngSpecBandLimF(U,lambda_,phy_x,phy_y,z_prop)  #final field z_prop

        self.atm = atm if atm is not None else self.atm
        if self.atm is not None:
            for ii in range(n_step):
                Uat = Up*T.exp(1j*self.atm[:,:,ii])
                Up = self.PropAngSpecBandLimF(Uat,lambda_,phy_x,phy_y,z_prop)

        # Up_f = Up.cpu().numpy()

        return U_,Up.cpu().numpy()
    
    def PIB(self,uin, Xc, Yc, Roc, pix_size):
        """
        Calculate the power within a circular region of interest (ROI) on an image.

        Parameters:
        - uin (numpy array): Input image.
        - Xc (int): X-coordinate of the center of the circular ROI.
        - Yc (int): Y-coordinate of the center of the circular ROI.
        - Roc (float): Radius of the circular ROI.
        - pix_size (float): Pixel size in millimeters.

        Returns:
        - P (float): Power within the circular ROI.

        This function calculates the power within a circular region of interest (ROI) on an input image.
        The circular ROI is defined by its center coordinates (Xc, Yc) and radius (Roc). The input image
        (uin) is multiplied by a circular mask to extract the intensity within the ROI. The total power
        within the circular ROI is computed by summing the intensities and scaling by the pixel area.
        The function returns the total power (P) within the circular ROI as a float value.
        """
        sx,sy = uin.shape
        x = np.arange(sx)
        y = np.arange(sy)

        X,Y = np.meshgrid(x,y)
        Circ = np.ones((sx,sy))
        R = np.sqrt((X-Xc)**2 + (Y-Yc)**2)
        Circ[R>Roc] = 0
        IntfCir = uin*Circ
        IntfCir1 = IntfCir*(pix_size*1e-3)**2
        P = np.sum(IntfCir1)
        uout = P
        return np.real(uout)
# %%
class TiledApertureBeamPropFast(TiledApertureBeamProp):

    def __init__(self,im_size,pix_size,n_channel,p_n,Kvar,Z,trans_pn,amp_v,g_amp,ra,d_,iteration=False):
        super().__init__(im_size,pix_size,n_channel,0,Kvar,Z,trans_pn,0,amp_v,g_amp,ra,d_) ## The zeros are for no atm and no n_screens
        self.p_n = p_n
        if iteration:
            self.iterations_setup()
  
    def iterations_setup(self): #TODO: Yet to implement this
        phy_x = self.pix_size*self.im_size
        obj_size = (self.im_size,self.im_size)
        lambda_ = 1.064*1e-3
        self.H1 = self.PropAngSpecBandLimF_kernel(obj_size,lambda_,phy_x,phy_x,self.Z)
        # self.CircROI = self.CircMask(obj_size,)
    
    def PropAngSpecBandLimF_kernel(self,obj_size,L,Dx,Dy,zm): #TODO: To modify this function , saving H kernel
        """
        Propagate a wavefront using the bandlimited angular spectrum method.

        Parameters:
        - L (float): Wavelength of the wavefront in millimeters.
        - Dx (float): Width of the input wavefront in millimeters.
        - Dy (float): Height of the input wavefront in millimeters.
        - zm (float): Propagation distance in meters.

        Returns:
        - H (torch tensor): Propagation Kernel.

        This function propagates the Propagation Kerenl of the wavefront using the bandlimited angular spectrum method,
        which essentially calculates the wavefront's Fourier transform and saves the point spread function 
        to account for diffraction effects during propagation. The input parameters include
        the input wavefront (uin), wavelength (L), width (Dx) and height (Dy) of the input
        wavefront, and the propagation distance (zmm). The output is Propagation Kernel which is a torch tensor. Note that this might have more memory usage.
        """
        # layer = uin
        lambda_ = L*1e-3
        k = 2*np.pi/lambda_
        z = zm #*1e-3
        phy_x = Dx*1e-3
        phy_y = Dy*1e-3

        # obj_size = (self.im_size,self.im_size)
        r,c = obj_size[0], obj_size[1]
        Fs_x = obj_size[1]/phy_x
        Fs_y = obj_size[0]/phy_y

        dFx = Fs_x/obj_size[1]
        dFy = Fs_y/obj_size[0]

        Fx = np.arange(-Fs_x/2,Fs_x/2,dFx)
        Fy = np.arange(-Fs_y/2,Fs_y/2,dFy)

        alpha = lambda_*Fx
        beta = lambda_*Fy
        len_alpha = len(alpha)
        len_beta = len(beta)

        gamma = self.get_gamma(len_alpha,len_beta,alpha,beta)


        ival = T.tensor(k*gamma*z).to(self.device)
        H0 = T.exp(1j*ival)

        Fxlim = 1/(np.sqrt(1+(2*dFx*z)**2)*lambda_)
        Fylim = 1/(np.sqrt(1+(2*dFy*z)**2)*lambda_)

        FxBL = np.zeros((len(beta),len(alpha)))
        FyBL = np.zeros((len(beta), len(beta)))
        FxBLr = self.rectangularPulse(-Fxlim,Fxlim,Fx)
        FyBLc = self.rectangularPulse(-Fxlim,Fylim,Fx)

        FxBL,FyBL = self.getFs(FxBL,FxBLr,FyBL,FyBLc,c,r)

        FxBL = T.tensor(FxBL).to(self.device)
        FyBL = T.tensor(FyBL).to(self.device)
        H1 = H0*FxBL*FyBL
        return H1
        # self.H1 = H1 #FIXME might change ? 

    def PropAngSpecBandLimF_loop(self,uin,H1=None):
        """
        Uses the wavefront's Fourier transform and applies a transfer function
        to account for diffraction effects during propagation.

        """
        if H1 is None:
            H1 = self.H1
        return ifft2(ifftshift((fftshift(fft2(uin)))*H1))
    #TODO: Optimize
    def sourceTAC_final(self,lambda_,w0,Ra,a,NL,Dx,NP,mDr,zm,m,Rc,phNs,Kvar,trans_pn,amp_v,g_amp, n_chan):

        """
        Simulate a source array with transverse angular diversity for coherent beam combining.

        Parameters:
        - lambda_ (float): Wavelength of the laser beam in millimeters.
        - w0 (float): Waist radius of the Gaussian beam.
        - Ra (float): Radius of the circular aperture.
        - a (float): Distance between adjacent beams in the source array in millimeters.
        - NL (int): Number of levels in the source array.
        - Dx (float): Size of the transverse plane in millimeters.
        - NP (int): Number of points for discretization in the transverse plane.
        - mDr (float): Maximum transverse angle diversity in degrees.
        - zm (float): Position of the beam along the optical axis in millimeters.
        - m (int): Mode of the beam.
        - Rc (float): Radius of curvature of the wavefront.
        - phNs (list): List of phase noise coefficients.
        - Kvar (float): Variance of the phase noise.
        - trans_pn (bool): Flag indicating whether to apply transverse phase noise.
        - amp_v (bool): Flag indicating whether to apply amplitude variation.
        - g_amp (float): Amplitude variation factor.
        - n_chan (int): Number of channels.

        Returns:
        - uc0 (numpy array): Initial complex amplitude distribution of the source array.
        - coord (numpy array): Coordinates of the source array elements.

        This function simulates a source array with transverse angular diversity for
        coherent beam combining. It generates an initial complex amplitude distribution
        (uc0) representing the source array and calculates the coordinates (coord) of
        the source array elements. The input parameters include the wavelength of the
        laser beam (lambda_), waist radius of the Gaussian beam (w0), radius of the
        circular aperture (Ra), distance between adjacent beams in the source array (a),
        number of levels in the source array (NL), size of the transverse plane (Dx),
        number of points for discretization in the transverse plane (NP), maximum transverse
        angle diversity (mDr), position of the beam along the optical axis (zm), mode of
        the beam (m), radius of curvature of the wavefront (Rc), list of phase noise
        coefficients (phNs), variance of the phase noise (Kvar), flag indicating whether
        to apply transverse phase noise (trans_pn), flag indicating whether to apply amplitude
        variation (amp_v), amplitude variation factor (g_amp), and number of channels (n_chan).
        """
        k = 2*np.pi/lambda_
        xf = a/2
        yf = round(np.sqrt(3)*a/2)

        theta = mDr
        RN1 = np.random.randint(low=0,high=100,size=(36,))
        RN2 = np.random.randint(low=0,high=100,size=(36,))
        theta_rx = np.radians(theta*RN1/100)
        theta_ry = np.radians(theta*RN2/100)
        kx = k*np.sin(theta_rx)
        ky = k*np.sin(theta_ry)

        mnE = 2*NL+1

        zmm = zm*1000


        start = time.time()

        U = T.zeros(NP,NP)

        X0 = np.arange(-2*NL,2*NL+2,2, dtype=int)*int(xf)
        Y0 = int(0*yf)

        end = time.time()-start

        phNs1 = T.tensor(phNs)
        n_cn = 0

        uc0 = np.sqrt(g_amp)*self.GaussBeamNDefPsLw(lambda_,w0,Rc,0,0,Dx,NP,m,zmm)*self.CirAperN(0,0,Ra,Dx,NP)
        coord = np.array([])
        for r in range(mnE):
            coord = np.append(coord,[X0[r],Y0])

        p=1
        X1 = np.array([])
        Y1 = np.array([])
        while p <= NL:
            X1 = np.arange(-2*NL+p,2*NL-p+2,2, dtype=int)*int(xf)
            Y1 = int(p*yf)
            for q in range(mnE-p):
                coord = np.append(coord,[X1[q],Y1])
                coord = np.append(coord,[X1[q],-Y1])

            p += 1

        return uc0,coord
    
    def get_ff(self,U, coord, noise):
        """
        Compute the far-field intensity pattern from the given wavefront and source array.

        Parameters:
        - U (torch tensor): Wavefront representation.
        - coord (numpy array): Coordinates of the source array elements.
        - noise (list or numpy array): Phase noise coefficients.

        Returns:
        - I (numpy array): Far-field intensity pattern.

        This function computes the far-field intensity pattern from the given wavefront
        (U) and source array coordinates (coord) with optional phase noise. The phase noise
        coefficients (noise) are applied to each source element, and the resulting wavefront
        is summed to obtain the far-field intensity pattern (I), which is returned as a
        numpy array.
        """
        U_ = T.zeros_like(U).to(self.device)
        phNs = T.tensor(noise).to(self.device)
        for idx, c in enumerate(coord):
            U_ += T.roll(U,shifts=(c[0],c[1]),dims=(0,1))*T.exp(1j*phNs[idx])
        U_1 = U_.cpu().numpy()
        I=np.abs(U_1)**2
        return I
    
    def TiledAperture_2(self):
        """
        Simulate coherent beam combining with a tiled aperture.

        Returns:
        - U_ (numpy array): Initial complex amplitude distribution of the source array.
        - Up (torch tensor): Final complex amplitude distribution after propagation.
        - coord (numpy array): Coordinates of the source array elements.

        This function simulates coherent beam combining with a tiled aperture. It generates
        an initial complex amplitude distribution (U_) representing the source array, propagates
        it through atmospheric screens if provided, and computes the final complex amplitude
        distribution (Up) after propagation. The input parameters include the size of the image
        plane (im_size), pixel size (pix_size), number of channels (n_channel), phase noise coefficients
        (p_n), number of atmospheric screens (n_screens), variance of the phase noise (Kvar), propagation
        distance (Z), flag indicating whether to apply transverse phase noise (trans_pn), atmospheric
        phase screens (atm), flag indicating whether to apply amplitude variation (amp_v), amplitude
        variation factor (g_amp), radius of the aperture (ra), and distance between adjacent apertures (d_).
        The function returns the initial complex amplitude distribution (U_), final complex amplitude
        distribution after propagation (Up), and coordinates of the source array elements (coord).
        """
        lambda_ = 1.064*1e-3
        Rc=1e15
        NP=self.im_size
        Dx=self.pix_size*NP
        m=0
        Ra= (self.ra/2)*1e3
        D=self.d_*1e3
        a=D*NP/Dx
        mDr=0
        w=0.85*Ra

        if self.n_channel == 7:
            NL=1
        elif self.n_channel == 19:
            NL = 2
        elif self.n_channel == 37:
            NL = 3
        elif self.n_channel == 61:
            NL = 4
        elif self.n_channel == 91:
            NL = 5
        elif self.n_channel == 127:
            NL = 6
        elif self.n_channel==217:
            NL=8
        else:
            ValueError('Please provide correct channel number..')

        zmm = self.Z*1e3
        zm = zmm*1e-3
        phNs = self.p_n

        w0 = w
        
        U,coord = super().sourceTAC_final(lambda_,w0,Ra,a,NL,Dx,NP,mDr,zm,m,Rc,phNs,self.Kvar,self.trans_pn,\
                                       self.amp_v,self.g_amp,self.n_channel)
        U_ = U.cpu().numpy()
        # pib_ = PIB(np.abs(U_)**2,1024,1024,1023,pix_size)
        # print('Input Power: ', np.real(pib_))

        phy_x = Dx
        phy_y = Dx

        Up = self.PropAngSpecBandLimF(U,lambda_,phy_x,phy_y,zmm)  #final field z_prop

        # Up_f = Up.cpu().numpy()

        return U_,Up,coord.reshape(self.n_channel,2).astype(int)
    def TiledAperture_mod(self,H1=None): #TODO: Update the Description
        """
        Simulate coherent beam combining with a tiled aperture.

        Returns:
        - U_ (numpy array): Initial complex amplitude distribution of the source array.
        - Up (torch tensor): Final complex amplitude distribution after propagation.
        - coord (numpy array): Coordinates of the source array elements.

        This function simulates coherent beam combining with a tiled aperture. It generates
        an initial complex amplitude distribution (U_) representing the source array, propagates
        it through atmospheric screens if provided, and computes the final complex amplitude
        distribution (Up) after propagation. The input parameters include the size of the image
        plane (im_size), pixel size (pix_size), number of channels (n_channel), phase noise coefficients
        (p_n), number of atmospheric screens (n_screens), variance of the phase noise (Kvar), propagation
        distance (Z), flag indicating whether to apply transverse phase noise (trans_pn), atmospheric
        phase screens (atm), flag indicating whether to apply amplitude variation (amp_v), amplitude
        variation factor (g_amp), radius of the aperture (ra), and distance between adjacent apertures (d_).
        The function returns the initial complex amplitude distribution (U_), final complex amplitude
        distribution after propagation (Up), and coordinates of the source array elements (coord).
        """
        lambda_ = 1.064*1e-3
        Rc=1e15
        NP=self.im_size
        Dx=self.pix_size*NP
        m=0
        Ra= (self.ra/2)*1e3
        D=self.d_*1e3
        a=D*NP/Dx
        mDr=0
        w=0.85*Ra

        if self.n_channel == 7:
            NL=1
        elif self.n_channel == 19:
            NL = 2
        elif self.n_channel == 37:
            NL = 3
        elif self.n_channel == 61:
            NL = 4
        elif self.n_channel == 91:
            NL = 5
        elif self.n_channel == 127:
            NL = 6
        elif self.n_channel==217:
            NL=8
        else:
            ValueError('Please provide correct channel number..')

        zmm = self.Z*1e3
        zm = zmm*1e-3
        phNs = self.p_n

        w0 = w

        U,coord = self.sourceTAC_final(lambda_,w0,Ra,a,NL,Dx,NP,mDr,zm,m,Rc,phNs,self.Kvar,self.trans_pn,\
                                       self.amp_v,self.g_amp,self.n_channel)
        U_ = U.cpu().numpy()
        # pib_ = PIB(np.abs(U_)**2,1024,1024,1023,pix_size)
        # print('Input Power: ', np.real(pib_))

        phy_x = Dx
        phy_y = Dx

        Up = self.PropAngSpecBandLimF_loop(U,H1)  #final field z_prop

        # Up_f = Up.cpu().numpy()

        return U_,Up,coord.reshape(self.n_channel,2).astype(int)
    
    def CircMask(self,shape,Xc,Yc,Roc):
        """
        Calculate a circular region of interest (ROI) mask on an image which will be used to calculate the Power within the ROI.

        Parameters:
        - shape (tuple): Input image shape 
        - Xc (int): X-coordinate of the center of the circular ROI.
        - Yc (int): Y-coordinate of the center of the circular ROI.
        - Roc (float): Radius of the circular ROI.

        Returns:
        - Circ (numpy array): circular ROI Mask.

        The circular ROI is defined by its center coordinates (Xc, Yc) and radius (Roc). The input image
        (uin) is multiplied by a circular mask to extract the intensity within the ROI. The total power
        within the circular ROI is computed by summing the intensities and scaling by the pixel area.
        The function returns the total power (P) within the circular ROI as a float value.
        """
        sx,sy = shape[0],shape[1]
        x = np.arange(sx)
        y = np.arange(sy)

        X,Y = np.meshgrid(x,y)
        Circ = np.ones((sx,sy))
        R = np.sqrt((X-Xc)**2 + (Y-Yc)**2)
        Circ[R>Roc] = 0
        self.CircROI = Circ
        return Circ
    
    def PIB_loop(self,uin,Circ=None):
        """
        Calculate the power within a circular region of interest (ROI) on an image.

        Parameters:
        - uin (numpy array): Input image.
        - Circ (numpy array): ROI mask.
        - pix_size (float): Pixel size in millimeters.

        Returns:
        - P (float): Power within the circular ROI (Circ).

        This function calculates the power within a circular region of interest (ROI) on an input image.
        The circular ROI is reused and is passed as parameter. This avoid recalulating the mask everytime an iteration is run. 
        The input image (uin) is multiplied by a circular mask to extract the intensity within the ROI. The total power
        within the circular ROI is computed by summing the intensities and scaling by the pixel area.
        The function returns the total power (P) within the circular ROI as a float value.
        """
        Circ = self.CircROI if Circ is None else Circ
        IntfCir = uin*Circ
        IntfCir1 = IntfCir*(self.pix_size*1e-3)**2
        P = np.sum(IntfCir1)
        uout = P
        return np.real(uout)
# %%
class TiledApertureBeamPropNested(TiledApertureBeamPropFast):
    def __init__(self, im_size, pix_size, n_channel, p_n, Kvar, Z, trans_pn, amp_v, g_amp, ra, d_, iteration=False, nested=True):
        super().__init__(im_size, pix_size, n_channel, p_n, Kvar, Z, trans_pn, amp_v, g_amp, ra, d_,iteration) ## Calls the Tilled Aperture Nested Function 
        self.shfx = np.array([-4, -2, 0, 2, 4, -3, -3, -1, -1, 1, 1, 3, 3, -2, -2, 0, 0, 2, 2]) #### Modify the array here 
        self.shfy = np.array([0, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 2, -2, 2, -2, 2, -2])
        self.shfx = np.array([0,2,-2,1,-1,1,-1]) ## For 7 units 
        self.shfy = np.array([0,0,0,1,1,-1,-1])
    def TiledAperture_2(self, p_n, tilt, TiltFact=0.25, f_lens_mm=2e3, fLensFull_mm=25e3):
        """
        Modified version of the TiledAperture_2 function to include the tilt parameter.
        """
        start = time.time()
        lambda_ = 1.064 * 1e-3
        Rc = 1e15
        NP = self.im_size
        Dx = self.pix_size * NP
        m = 0
        Ra = (self.ra / 2) * 1e3
        D = self.d_ * 1e3
        a = D * NP / Dx
        mDr = 0
        w = 0.85 * Ra

        if self.n_channel == 7:
            NL = 1
        elif self.n_channel == 19:
            NL = 2
        elif self.n_channel == 37:
            NL = 3
        elif self.n_channel == 61:
            NL = 4
        elif self.n_channel == 91:
            NL = 5
        elif self.n_channel == 127:
            NL = 6
        elif self.n_channel == 217:
            NL = 8
        else:
            raise ValueError('Please provide correct channel number.')

        zmm = self.Z * 1e3
        zm = zmm * 1e-3
        phNs = p_n
        n_step = self.n_screens
        zsmm = zmm / (n_step + 1)
        w0 = w

        Du = (2 * NL + 1) * D - 1
        ShMag = round(Du / self.pix_size)
        xSf = round(ShMag / 2)
        ySf = round(np.sqrt(3) * ShMag / 2)

        ShFx = T.tensor(self.shfx, dtype=T.int16).to(self.device)
        ShFy = T.tensor(self.shfy, dtype=T.int16).to(self.device)

        Ln = 2 * (NL * D + Ra)
        Lnp = Ln / self.pix_size
        LnF = 2 * (Du + NL * D + Ra)
        LnFp = LnF / self.pix_size

        rPIB = 1.22 * lambda_ * zmm / LnF
        rPIBp = rPIB / self.pix_size
        k = 2 * np.pi / lambda_
        rp_lens = 4 * (lambda_ * f_lens_mm) / (np.pi * Ln * self.pix_size)
        xs = T.arange(NP).to(self.device)
        xl = (Dx / NP) * xs
        xlc = xl - 0.5 * xl[NP - 1]
        X, Y = T.meshgrid(xlc, xlc, indexing='xy')
        U = T.zeros(NP, NP, dtype=T.cfloat).to(self.device)
        Lens = T.zeros(NP, NP, dtype=T.cfloat).to(self.device)
        UL = T.zeros(NP, NP, dtype=T.cfloat).to(self.device)

        phy_x = Dx
        phy_y = Dx
        array = [100]
        for sU in range(len(ShFx)):
            Ys = xSf * ShFx[sU]
            Xs = ySf * ShFy[sU]
            kx = k * T.sin(T.arctan(TiltFact * Xs * self.pix_size / zmm)) * tilt
            ky = k * T.sin(T.arctan(TiltFact * Ys * self.pix_size / zmm)) * tilt
            if sU in array:
                Uab = self.sourceTAC_final(lambda_, w0, Ra, a, NL, Dx, NP, mDr, zm, m, Rc, phNs[sU], self.Kvar, self.trans_pn, self.amp_v, self.g_amp, self.n_channel)
            else:
                Uab = self.sourceTAC_final(lambda_, w0, Ra, a, NL, Dx, NP, mDr, zm, m, Rc, phNs[sU], self.Kvar, self.trans_pn, self.amp_v, self.g_amp, self.n_channel)
            U += Uab * T.exp(1j * ky * (Y + Ys * self.pix_size)) * T.exp(1j * kx * (X + Xs * self.pix_size))
            Lens += self.SphLens(Uab, phy_x, phy_y, -Xs * self.pix_size, -Ys * self.pix_size, NP, lambda_, f_lens_mm)

        U_ = U.cpu().numpy()

        if self.atm is None:
            z_prop = zmm
        else:
            z_prop = zsmm

        lens = True
        if lens:
            ULf = self.SphLens(U, phy_x, phy_y, 0, 0, NP, lambda_, fLensFull_mm)
            UL = self.PropAngSpecBandLimF(Lens, lambda_, phy_x, phy_y, f_lens_mm)
        else:
            Up = self.PropAngSpecBandLimF(U, lambda_, phy_x, phy_y, zmm)
        Up = self.PropAngSpecBandLimF(U, lambda_, phy_x, phy_y, zmm)
        UL_ = UL.cpu().numpy()

        if self.atm is not None:
            for ii in range(n_step):
                Uat = Up * T.exp(1j * self.atm[:, :, ii])
                Up = self.PropAngSpecBandLimF(Uat, lambda_, phy_x, phy_y, z_prop)

        Up_f = Up.cpu().numpy()
        Intf = np.abs(Up_f) ** 2

        end = time.time() - start
        return U_, Up_f, Intf, UL_, (xSf * ShFx).cpu().numpy(), (ySf * ShFy).cpu().numpy(), rPIBp, rp_lens
    ## Nested PIB #TODO , redefine sourceplane tac 
    def find_sub_array_pib(self,UL,Xs,Ys,rp_lens,pib_n,units):
        """
        Old Implementation , where the mask is created everytime , this is not recommended if there is multiple iterations are to run 
        """
        inner_pib = np.zeros((units,1))
        # xs_idx = {'0':0,'1':2,'2':1,'3':6,'4':5,'5':4,'6':3}
        for idx in range(units):
          I_1  = self.PIB_loop(np.power(np.abs(UL),2),self.im_size/2+Xs[idx],self.im_size/2+Ys[idx],rp_lens,self.pix_size)
          inner_pib[idx] = I_1/pib_n
        return inner_pib
    def find_sub_array_pib_mod(self,U_,masks,pin_total): ## Much faster
        """ 
        U_ (Torch tensor)
        masks (Torch tensor)
        pix_size (float)
        pin_total (float) (for normalization)
        """
        return (np.real(np.sum((np.abs(U_)**2 * masks) *(self.pix_size * 1e-3) ** 2,axis=(1,2)))/pin_total)#.cpu().numpy() ## dim(1,2) is dimentions that we want to sum over
    ##FIXME this 
    def sourceTAC_final(self,lambda_,w0,Ra,a,NL,Dx,NP,mDr,zm,m,Rc,phNs,Kvar,trans_pn,amp_v,g_amp,n_channel,XsU,YsU,rotate=False):
        """
        Modifed sourceTAC_final function to include the XsU and YsU parameters.
        """
        start = time.time()
        k = 2*np.pi/lambda_
        xf = a/2
        yf = round(np.sqrt(3)*a/2)
        
        theta = mDr
        RN1 = np.random.randint(low=0,high=100,size=(36,))
        RN2 = np.random.randint(low=0,high=100,size=(36,))
        theta_rx = np.radians(theta*RN1/100)
        theta_ry = np.radians(theta*RN2/100)
        kx = k*np.sin(theta_rx)
        ky = k*np.sin(theta_ry)
        
        mnE = 2*NL+1
        
        zmm = zm*1000
        
        # xs = T.arange(NP).to(device)
        # xl = (Dx/NP)*xs
        # xlc = xl - 0.5*xl[NP-1]-XsU
        # ylc = xl - 0.5*xl[NP-1]-YsU
        # X,Y = T.meshgrid(xlc,ylc,indexing='xy')
        
        U = T.zeros(NP,NP,dtype=T.cfloat).to(self.device)
        
        Y0 = np.arange(-2*NL,2*NL+2,2)*xf
        X0 = 0*yf
        
        
        
        phNs1 = T.tensor(phNs)
        n_cn = 0
        
        for r in range(mnE):
          if rotate:
            u0 = self.GaussBeamNDefPsLw(lambda_,w0,Rc,Y0[r]+YsU,X0+XsU,Dx,NP,m,zmm)[0:NP,0:NP]
            c0 = self.CirAperN(Y0[r]+YsU,X0+XsU,Ra,Dx,NP)[0:NP,0:NP]
          else:
            u0 = self.GaussBeamNDefPsLw(lambda_,w0,Rc,X0+XsU,Y0[r]+YsU,Dx,NP,m,zmm)[0:NP,0:NP]
            c0 = self.CirAperN(X0+XsU,Y0[r]+YsU,Ra,Dx,NP)[0:NP,0:NP]
        
          UC0 = np.sqrt(g_amp)*u0
          Uel_1 = T.multiply(T.exp(1j*phNs1[n_cn]), UC0).to(self.device)
          E_1i = T.real(Uel_1) + T.tensor(amp_v*T.randn(1)).to(self.device)
          E_1q = T.imag(Uel_1) + T.tensor(amp_v*T.randn(1)).to(self.device)
          E_1 = T.complex(E_1i,E_1q).to(self.device)
          Uel = E_1*c0
          U += Uel
          n_cn += 1
        
        p=1
        
        
        while p <= NL:
          Y1 = np.arange(-2*NL+p,2*NL-p+2,2)*xf
          X1 = p*yf
          for q in range(mnE-p):
            if rotate:
              u1p = self.GaussBeamNDefPsLw(lambda_,w0,Rc,Y1[q]+YsU,X1+XsU,Dx,NP,m,zmm)[0:NP,0:NP]
              c1p = self.CirAperN(Y1[q]+YsU,X1+XsU,Ra,Dx,NP)[0:NP,0:NP]
            else:
              u1p = self.GaussBeamNDefPsLw(lambda_,w0,Rc,X1+XsU,Y1[q]+YsU,Dx,NP,m,zmm)[0:NP,0:NP]
              c1p = self.CirAperN(X1+XsU,Y1[q]+YsU,Ra,Dx,NP)[0:NP,0:NP]
        
            uc1p = np.sqrt(g_amp)*u1p
            Uel_1 = T.multiply(T.exp(1j*phNs1[n_cn]), uc1p).to(self.device)
            E_1i = T.real(Uel_1) + T.tensor(amp_v*np.random.randn(1)).to(self.device)
            E_1q = T.imag(Uel_1) + T.tensor(amp_v*np.random.randn(1)).to(self.device)
            E_1 = T.complex(E_1i,E_1q).to(self.device)
            Uel = E_1*c1p
        
            U += Uel
            n_cn += 1
        
            # u1m = GaussBeamNDefPsLw(lambda_,w0,Rc,-X1+XsU,Y1[q]+YsU,Dx,NP,m,zmm)[0:NP,0:NP]
            # c1m = CirAperN(-X1+XsU,Y1[q]+YsU,Ra,Dx,NP)[0:NP,0:NP]
            if rotate:
              u1m = self.GaussBeamNDefPsLw(lambda_,w0,Rc,Y1[q]+YsU,-X1+XsU,Dx,NP,m,zmm)[0:NP,0:NP]
              c1m = self.CirAperN(Y1[q]+YsU,-X1+XsU,Ra,Dx,NP)[0:NP,0:NP]
            else:
              u1m = self.GaussBeamNDefPsLw(lambda_,w0,Rc,-X1+XsU,Y1[q]+YsU,Dx,NP,m,zmm)[0:NP,0:NP]
              c1m = self.CirAperN(-X1+XsU,Y1[q]+YsU,Ra,Dx,NP)[0:NP,0:NP]
        
            uc1m = np.sqrt(g_amp)*u1m
            Uel_1 = T.multiply(T.exp(1j*phNs1[n_cn]),uc1m).to(self.device)
            E_1i = T.real(Uel_1) + T.tensor(amp_v*np.random.randn(1)).to(self.device)
            E_1q = T.imag(Uel_1) + T.tensor(amp_v*np.random.randn(1)).to(self.device)
            E_1 = T.complex(E_1i,E_1q).to(self.device)
            Uel = E_1*c1m
        
            U += Uel
            n_cn += 1
          p += 1
        
        end = time.time()-start
        # print('Source TAC:',str(end),'s')
        uout = U
        
        return uout
    def debug_pib(self,U_):
        try:
            plt.figure(figsize=(10, 8))
            plt.imshow((T.abs(U_)**2).cpu().numpy(), cmap='viridis')
            plt.xlim(1024-100,1024+100) 
            plt.ylim(1024-100,1024+100)
            plt.title('Outer Loop Field ')
            plt.colorbar()
            plt.show()
        except:
            print('KeyError')
            print('OuterLoop')
            pass
    def debug_pib_inner(self,U_,rplens,pib,pin_total):
        plt.figure(figsize=(10, 8))
        # plt.imshow((T.abs(U_)**2).cpu().numpy(), cmap='viridis')
        udebug = (T.abs(U_)**2).cpu().numpy()
        # udebug = cv2.bitwise_xor(udebug,pib2.cpu().numpy())
        # contours, _ = cv2.findContours((mask.clone().cpu().numpy()).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # contour_mask = np.zeros_like(mask.clone().cpu().numpy())
        # cv2.drawContours(udebug, contours, -1, (0, 255, 0), 2)
        plt.imshow(udebug,cmap='viridis')
        # pch1 =  plt.Circle((im_size/2,im_size/2),rplens,fill=False,color='r')
        # plt.gca().add_patch(pch1)
        plt.gca().add_patch(plt.Circle((self.im_size/2,self.im_size/2),rplens,fill=False,color='r'))
        plt.gca().add_patch(plt.Circle((self.im_size/2,self.im_size/2),rplens*4,fill=False,color='g'))
        # plt.xlim(im_size//2-600,im_size//2+600) 
        # plt.ylim(im_size//2-600,im_size//2+600)
        # plt.text(im_size//2-600,im_size//2+600,f'PIB Value: {pib}',color='r',fontsize=12)
        plt.annotate(f"PIB Value: {str(pib)[:5]},\nPIN {str(pin_total)[:5]} \n absPIB : {str(pin_total*pib)[:5]}",xy=(self.im_size//2,self.im_size//2),xytext=(self.im_size//2,self.im_size//2+self.im_size//6),arrowprops=dict(facecolor='black',arrowstyle='->'))
        plt.colorbar()
        plt.show()
        # plt.figure(figsize=(10, 8))
        # # plt.imshow((T.abs(U_)**2).cpu().numpy(), cmap='viridis')
        # udebug = (T.abs(U_)**2).cpu().numpy(
        # # udebug = cv2.bitwise_xor(udebug,pib.cpu().numpy())
        # plt.imshow(udebug,cmap='viridis'
        # plt.gca().add_patch(plt.Circle((self.im_size/2,self.im_size/2),rplens,fill=False,color='r'))
        # plt.gca().add_patch(plt.Circle((self.im_size/2,self.im_size/2),rplens*4,fill=False,color='g'))
        # plt.text(self.im_size//2,0,f'PIB Value: {pib}',color='r',fontsize=12)
        # # plt.xlim(1024-100,1024+100) 
        # # plt.ylim(1024-100,1024+100)
        # plt.colorbar()
        # plt.show()
    def debug_pib_inner_lens(self,U_,rplens,pib,pin_total,seq,coord):
        plt.figure(figsize=(10, 8))
        # plt.imshow((T.abs(U_)**2).cpu().numpy(), cmap='viridis')
        udebug = (T.abs(U_)**2).cpu().numpy() 
        # udebug = cv2.bitwise_xor(udebug,pn.cpu().numpy())
        plt.imshow(udebug,cmap='viridis')
        if seq:
            pch1 =  plt.Circle((self.im_size/2,self.im_size/2),rplens,fill=False,color='r')
            plt.gca().add_patch(pch1)
        else:
            # plt.imshow(T.sum((T.abs(U_)**2 * mask) *(pix_size * 1e-3) ** 2,dim=(0)).cpu().numpy())
            for i,c in enumerate(np.array(coord)[:,1,:]):
                # plt.scatter(c[0],c[1],c='r')
                plt.gca().add_patch(plt.Circle((self.im_size//2+c[1],self.im_size//2+c[0]),rplens,fill=False,color='r'))
                plt.gca().add_patch(plt.Circle((self.im_size//2+c[1],self.im_size//2+c[0]),rplens*4,fill=False,color='g'))
                plt.annotate(f"{str(pib[i])[:5]},\nPIN {str(pin_total)[:5]},\n absPIB {str(pin_total*pib)[:5]}",xy=(self.im_size//2+c[1],self.im_size//2+c[0]),xytext=(self.im_size//2+c[1],self.im_size//2+c[0]+rplens*4),arrowprops=dict(facecolor='black',arrowstyle='->'))
        # plt.xlim(1024-200,1024+200) 
        # plt.ylim(1024-200,1024+200)
        plt.title('Inner Loop Field ')
        plt.colorbar()
        plt.show()
    def get_ff_pib(self,U:np.ndarray,coord:list,noise:np.ndarray,mask,pin_total,k=None,fllens = None,inner=False,use_lens=True,rplens=None,plot=False,seq=True):
        '''
        Modification of the get_ff function for Nested Loop Structures #TODO
        '''
        phsm = noise
        mask = T.tensor(mask).to(self.device)
        U_ = T.zeros_like(U).to(self.device)

        phNs = T.tensor(noise).to(self.device)
        Xs,Ys = U.shape
        X, Y = T.meshgrid(T.arange(Xs).to(self.device) - Xs // 2, T.arange(Ys).to(self.device) - Ys // 2) ## Centered at zero
        ################################################ Outer Loop ########################################################
        if use_lens is False: ### For the Outer Loop Case with no lens
            
            for idx_m in range(len(coord)): # 7
                for idx, c in enumerate(coord[idx_m]): # 7
                    U_ += T.roll(U,shifts=(c[0],c[1]),dims=(0,1))*T.exp(1j*phNs[idx_m,idx])
            pib2 = (T.abs(U_)**2 * mask) *(self.pix_size * 1e-3) ** 2#.cpu().numpy()
            pib = (T.real(T.sum(pib2))/pin_total).cpu().numpy() 
            if plot:
                self.debug_pib_inner(U_,rplens,pib,pin_total) ### Debugging the Outer Loop Field
            else:
                pass

        ############################################ Outer Loop with Lens ##################################################      
        elif use_lens is True and inner is False: ### For the Outer Loop Lens Case
            
            for idx_m in range(len(coord)):
                for idx, c in enumerate(coord[idx_m]):
                    kx = k * T.sin(T.tensor(-c[0] * self.pix_size * 1e-3 / fllens))
                    ky = k * T.sin(T.tensor(-c[1] * self.pix_size * 1e-3 / fllens))
                    ubb = T.exp(1j * kx * X * self.pix_size) * T.exp(1j * ky * Y * self.pix_size)
                    U_ += U * ubb * T.exp(1j * phNs[idx_m, idx])

            self.debug_pib(U_) ### Debugging the Outer Loop Field   
            pib2 = (T.abs(U_)**2 * mask)# *(pix_size * 1e-3) ** 2#.cpu().numpy()
            pib = (T.real(T.sum(pib2 *(self.pix_size *1e-3) **2 ))).cpu().numpy()
            print(f"Outer PIB : {pib}")
            print(f"Input Power :{pin_total}\n")
            pib = pib / pin_total   
            if plot:
                self.debug_pib_inner(U_) ### Debugging the Inner Loop Field
            else:
                pass
            # pib = (T.real(T.sum(pib))/pin_total).cpu().numpy()

        ################################################ Inner Loop ########################################################
        elif use_lens is True and inner is True :  ### For the Inner Loop with lens 

            U_2 = T.zeros_like(U).to(self.device)
            Xs,Ys = U.shape
            Kx = k * T.sin(T.tensor(-(np.array(coord)[0,:,0]) * self.pix_size * 1e-3 / fllens)) ## Calcualte the kx and ky for the Center inner array (Once)
            Ky = k * T.sin(T.tensor(-(np.array(coord)[0,:,1]) * self.pix_size * 1e-3 / fllens))
            # print(Kx,Ky)
            pib = []
            timepib2 = time.time()
            for idx_m in range(7): # 7  
                U_2 = T.zeros_like(U).to(self.device)
                for idx, c in enumerate(coord[idx_m]):
                    ## translate to inner array center
                    U_2 += U*T.exp(1j * Kx[idx] * (X) * self.pix_size) * T.exp(1j * Ky[idx] * (Y) * self.pix_size)*T.exp(1j * phNs[idx_m, idx])
                #TODO: PIB mask here
                if seq: ## Less memory usage, single mask calculation
                    timepib = time.time()
                    pn = (T.abs(U_2)**2 * mask) *(self.pix_size * 1e-3) ** 2
                    # print(f" Inner Input Power: {pin_total}\n")
                    pib.append((T.real(T.sum(pn))/pin_total).cpu().numpy())
                #   pib.append((T.real(T.sum(pn))).cpu().numpy())
                    # print(f"Inner array PIB : {pib[-1]}")
                    print(f"Time taken for single PIB : {time.time()-timepib}")
                U_ += T.roll(U_2,shifts=(np.array(coord)[idx_m,1,0],np.array(coord)[idx_m,1,1]),dims=(0,1)) ## Roll the array to the center of the outer arrays   
            #TODO : PIB on U_
            if seq != True: ## uses ufunc to calculate PIB 
                timepib = time.time()
                print(f"Inner Input Power: {pin_total.shape}\n")
                pib = (T.real(T.sum((T.abs(U_)**2 * mask) *(self.pix_size * 1e-3) ** 2,dim=(1,2)))/pin_total).cpu().numpy() ## dim(1,2) is dimentions that we want to sum over
                print(f"Time taken for PIB array : {time.time()-timepib}") 
            else:
                pib = np.array(pib)
            print(f"Time Taken for seq PIb : {time.time()-timepib2}")
            if plot:
                self.debug_pib_inner_lens(U_,rplens,pib,pin_total,seq,coord) ### Debugging the Inner Loop Field
                
        return pib

    
    def get_outer_coordinates(self,NL,
                            D,#seperation of the mirror in mm
                            ra, #in m
                            Z,#in m
                            pix_size,
                            f_lens_mm, #in mm
                            Du=None,
                            inner=False):
        lambda_ = 1.064 * 1e-3 
        # Du = (2 * NL + 1) * D + 1  ### Unit mm Seperation of the Mirror 
        print('Using Du',Du)
        ShMag = round(Du / pix_size)
        xSf = round(ShMag / 2)  ### Unit Seperation in X
        ySf = round(np.sqrt(3) * ShMag / 2)     
        ShFx = self.shfx#np.array([0, 2, -2, 1, -1, 1, -1])
  #      Encodes the Structure
        ShFy = self.shfy#np.array([0, 0, 0, 1, 1, -1, -1])
        Xs = xSf * ShFx  ## Shift in X
        Ys = ySf * ShFy
        coords = [(y,x) for x,y in zip(Xs,Ys)]
        # coords = []
        # for sU in range(7):
        #     Xs = xSf * ShFx[sU]  ## Shift in X
        #     Ys = ySf * ShFy[sU]
        #     coords.append([Xs, Ys])       
        zmm = Z * 1e3 ## to mm
        Ra = (ra / 2) * 1e3 ## mm
        Ln = 2 * (NL * D + Ra) #mm
        LnF = 2 * (Du + NL * D + Ra) ## mm
        # print(LnF)
        rPIB = 1.22 * lambda_ * zmm / LnF
        rPIBp = rPIB / pix_size
        # TiltFact = 0.25
        if inner:
            rp_lens = 4 * (lambda_ * f_lens_mm) / (np.pi * Ln * pix_size)
        else:
            rp_lens = 4 * (lambda_ * f_lens_mm) / (np.pi * LnF * pix_size)
        return coords,rp_lens,rPIBp#/pix_size,rPIBp
# %%    
class TiledAperture_test(TiledApertureBeamPropFast): ## Currently in Use 
    def __init__(self, im_size, pix_size, n_channel, p_n, Kvar, Z, trans_pn, amp_v, g_amp, ra, d_,wx,wy,use_lens,fflens,Pxx,iteration=False):
        super().__init__(im_size, pix_size, n_channel, p_n, Kvar, Z, trans_pn, amp_v, g_amp, ra, d_,iteration)
        self.wx = wx
        self.wy = wy
        self.use_lens = use_lens
        self.fflens = fflens
        self.Pxx = Pxx


    def GaussBeamNDefPsLw(self,lambda_, wx, wy, Rc, X0, Y0, Dxy, NP, m, zmm, P):  ## Here is Polarization is applied
        s = time.time()
        k = 2 * np.pi / lambda_
        w0 = 0.5 * (wx + wy)

        zR = np.pi * w0**2 / lambda_
        z = zmm * 1000

        M = NP
        dx = Dxy / M
        x = T.arange((-M / 2 - X0) * dx, (M / 2 - X0) * dx, dx)[0:NP]

        N = NP
        dy = Dxy / N
        y = T.arange((-N / 2 - Y0) * dy, (N / 2 - Y0) * dy, dy)[0:NP]

        [X, Y] = T.meshgrid(x, y, indexing="xy")
        X = X.to(self.device)
        Y = Y.to(self.device)

        R = T.sqrt(X**2 + Y**2).to(self.device)

        # P = 0.01
        A = (2 * P) / (np.pi * (w0 * 1e-3) ** 2)

        C = np.sqrt(A) * T.exp(1j * k * (R**2) / (2 * Rc)).to(self.device)

        Psi = C * T.exp(-(X**2 / wx**2) - (Y**2 / wy**2))
        uout = Psi
        e = time.time() - s
        # print('Gaussian_beam: ',str(e),'s')
        return uout
    
    def sourceTAC_final(self,lambda_,wx,wy,Ra,a,NL,Dx,NP,zm,m,Rc,phNs,trans_pn,amp_v,g_amp,n_chan,thetar,th2d,P):
        start = time.time()
        device = self.device
        k = 2 * np.pi / lambda_
        xf = a / 2
        yf = round(np.sqrt(3) * a / 2)

        # theta = mDr
        thetar = np.radians(thetar)
        th2d = np.radians(th2d)
        # print(thetar)
        kx = k * np.sin(thetar) * np.sin(th2d)[0]
        ky = k * np.sin(thetar) * np.cos(th2d)[0]
        # print(kx.shape)

        mnE = 2 * NL + 1

        zmm = zm * 1000

        xs = np.arange(0, NP)
        ys = np.arange(0, NP)

        x1 = (Dx / NP) * xs
        y1 = (Dx / NP) * ys
        x1c = x1 - 0.5 * x1[NP - 1]
        y1c = y1 - 0.5 * y1[NP - 1]
        X, Y = np.meshgrid(x1c, y1c)

        U = T.zeros(NP, NP, dtype=T.cfloat).to(device)
       
        X0 = np.arange(-2 * NL, 2 * NL + 2, 2) * xf
        Y0 = 0 * yf

        phNs1 = T.tensor(phNs)
        n_cn = 0
        # bsf=np.array([1.0462,  1.0308, 1.0308])

        for r in range(mnE):
            # P=0.01
            # if (X0[r]+Y0)==0:
            #   P=0.
            shx = 0
            if (X0[r] + Y0) == 0:
                shx = 125

            u0 = self.GaussBeamNDefPsLw(lambda_, wx[n_cn], wy[n_cn], Rc, X0[r] - shx, Y0, Dx, NP, m, zmm, P[0, n_cn])
            c0 = self.CirAperN(X0[r], Y0, Ra, Dx, NP)
            # th0,rh0 = cart2pol(X-(Dx/2)-X0[r]*(Dx/NP), Y-(Dx/2)-Y0*(Dx/NP))   #Uncomment to include Transverse Abberation
            # Rabr = np.sqrt(Kvar)*trans_pn[r,:]
            # abrZpc = Rabr[0:10]
            # Phabr[:,:,r] = Transverse_ph_abbrZP(abrZpc,th0,rh0,Ra)
            # UC0[:,:,r] = np.sqrt(g_amp)*T.multiply(u0[:,:,r],Phabr[:,:,r]).to(device)
            th0, rh0 = self.cart2pol(X - X0[r] * (Dx / NP), Y - Y0 * (Dx / NP))
            Rabr = trans_pn[n_cn, :]
            signRand = 2 * np.random.randint(0, 2, size=10) - 1
            abrZpc = Rabr * signRand
            Phabr = self.Transverse_ph_abbrZP(lambda_, abrZpc, th0, rh0, Ra)
            UC0 = np.sqrt(g_amp) * T.multiply(u0, Phabr).to(device)

            UC0 *= T.multiply(
                T.exp(T.tensor(-1j * ky[n_cn] * (Y - (Dx / NP) * Y0))).to(device),
                T.exp(T.tensor(1j * kx[n_cn] * (X - (Dx / NP) * X0[r]))).to(device),
            )
            Uel_1 = T.multiply(T.exp(1j * phNs1[n_cn]), UC0).to(device)
            E_1i = T.real(Uel_1) + T.tensor(amp_v * T.randn(1)).to(device)
            E_1q = T.imag(Uel_1) + T.tensor(amp_v * T.randn(1)).to(device)
            E_1 = T.complex(E_1i, E_1q).to(device)
            Uel = E_1 * c0
            U += Uel #* 0  ## Making it zero to allow only one beam
            n_cn += 1

        p = 1

        while p <= NL:
            X1 = np.arange(-2 * NL + p, 2 * NL - p + 2, 2) * xf
            Y1 = p * yf
            for q in range(mnE - p):
                # P=0.01
                # if q==1:
                #   P=0
                shy = 0.0
                if q == 1:
                    shy = 150

                u1p = self.GaussBeamNDefPsLw(
                    lambda_, wx[n_cn], wy[n_cn], Rc, X1[q], Y1, Dx, NP, m, zmm, P[0, n_cn]
                )
                c1p = self.CirAperN(X1[q], Y1, Ra, Dx, NP)
                # th1p,rh1p = cart2pol(X-(Dx/2)-X1[q]*(Dx/NP), Y-(Dx/2)-Y1*(Dx/NP))   #Uncomment to include Transverse Abberation
                # Rabr = np.sqrt(Kvar)*trans_pn[mnE+2*q-1,:]
                # abrZpc = Rabr[0:10]
                # Phabrp[:,:,q] = Transverse_ph_abbrZP(abrZpc,th1p,rh1p,Ra)
                # uc1p[:,:,q] = np.sqrt(g_amp)*T.multiply(u1p[:,:,q],Phabrp[:,:,q]).to(device)
                th1p, rh1p = self.cart2pol(
                    X - X1[q] * (Dx / NP), Y - Y1 * (Dx / NP)
                )  # Uncomment to include Transverse Abberation
                Rabr = trans_pn[n_cn, :]
                signRand = 2 * np.random.randint(0, 2, size=10) - 1
                abrZpc = Rabr * signRand
                Phabrp = self.Transverse_ph_abbrZP(lambda_, abrZpc, th1p, rh1p, Ra)
                uc1p = np.sqrt(g_amp) * T.multiply(u1p, Phabrp).to(device)
                # uc1p = np.sqrt(g_amp)*u1p
                uc1p *= T.multiply(
                    T.exp(T.tensor(-1j * ky[n_cn] * (Y - (Dx / NP) * Y1))).to(device),
                    T.exp(T.tensor(1j * kx[n_cn] * (X - (Dx / NP) * X1[q]))).to(device),
                )
                Uel_1 = T.multiply(T.exp(1j * phNs1[n_cn]), uc1p).to(device)
                E_1i = T.real(Uel_1) + T.tensor(amp_v * np.random.randn(1)).to(device)
                E_1q = T.imag(Uel_1) + T.tensor(amp_v * np.random.randn(1)).to(device)
                E_1 = T.complex(E_1i, E_1q).to(device)
                Uel = E_1 * c1p
                # if p == 1 and q == 1:  # Only this beam is allowed
                U += Uel #* 0  # to avoid double beams
                n_cn += 1

                # P=0.01
                u1m = self.GaussBeamNDefPsLw(
                    lambda_,
                    wx[n_cn],
                    wy[n_cn],
                    Rc,
                    X1[q],
                    -Y1 + shy,
                    Dx,
                    NP,
                    m,
                    zmm,
                    P[0, n_cn],
                )
                c1m = self.CirAperN(X1[q], -Y1, Ra, Dx, NP)
                # th1m,rh1m = cart2pol(X-(Dx/2)-X1[q]*(Dx/NP),Y-(Dx/2)+Y1*(Dx/NP))    #Uncomment to include Transverse Abberation
                # Rabr=np.sqrt(Kvar)*trans_pn[mnE+2*q,:]
                # abrZpc = Rabr[0:10]
                # Phabrm[:,:,q] = Transverse_ph_abbrZP(abrZpc,th1m,rh1m,Ra)
                # uc1m[:,:,q] = np.sqrt(g_amp)*T.multiply(u1m[:,:,q],Phabrm[:,:,q]).to(device)
                th1m, rh1m = self.cart2pol(
                    X - X1[q] * (Dx / NP), Y + Y1 * (Dx / NP)
                )  # Uncomment to include Transverse Abberation
                Rabr = trans_pn[n_cn, :]
                signRand = 2 * np.random.randint(0, 2, size=10) - 1
                abrZpc = Rabr * signRand
                Phabrm = self.Transverse_ph_abbrZP(lambda_, abrZpc, th1m, rh1m, Ra)
                uc1m = np.sqrt(g_amp) * T.multiply(u1m, Phabrm).to(device)
                # uc1m = np.sqrt(g_amp)*u1m
                uc1m *= T.multiply(
                    T.exp(T.tensor(-1j * ky[n_cn] * (Y + (Dx / NP) * Y1))).to(device),
                    T.exp(T.tensor(1j * kx[n_cn] * (X - (Dx / NP) * X1[q]))).to(device),
                )
                Uel_1 = T.multiply(T.exp(1j * phNs1[n_cn]), uc1m).to(device)
                E_1i = T.real(Uel_1) + T.tensor(amp_v * np.random.randn(1)).to(device)
                E_1q = T.imag(Uel_1) + T.tensor(amp_v * np.random.randn(1)).to(device)
                E_1 = T.complex(E_1i, E_1q).to(device)
                Uel = E_1 * c1m
                # if p == 1 and q == 1:  ### Take only one beam
                U += Uel
                n_cn += 1
            p += 1

        end = time.time() - start
        # print('Source TAC:',str(end),'s')
        uout = U
        return uout
    
    def sourceTAC_mod(self,lambda_,wx,wy,Ra,a,NL,Dx,NP,mDr,Rc,phNs,Kvar,trans_pn,amp_v,g_amp,n_chan,P,):  ## This function is for single beam generation
        k = 2 * np.pi / lambda_
        xf = a / 2
        yf = round(np.sqrt(3) * a / 2)

        theta = mDr
        RN1 = np.random.randint(low=0, high=100, size=(36,))
        RN2 = np.random.randint(low=0, high=100, size=(36,))
        theta_rx = np.radians(theta * RN1 / 100)
        theta_ry = np.radians(theta * RN2 / 100)
        #   kx = k*np.sin(theta_rx)
        #   ky = k*np.sin(theta_ry)
        xs = np.arange(0, NP)
        ys = np.arange(0, NP)

        x1 = (Dx / NP) * xs
        y1 = (Dx / NP) * ys
        x1c = x1 - 0.5 * x1[NP - 1]
        y1c = y1 - 0.5 * y1[NP - 1]
        X, Y = np.meshgrid(x1c, y1c)
        mnE = 2 * NL + 1

        X0 = np.arange(-2 * NL, 2 * NL + 2, 2, dtype=int) * int(xf)
        Y0 = int(0 * yf)

        n_cn = 0
        uc0 = (
            np.sqrt(g_amp)
            * self.GaussBeamNDefPsLw(lambda_,wx=wx,wy=wy,Rc=Rc,X0=0,Y0=0,Dxy=Dx,NP=NP,m=None,zmm=7e6,P=P[0, n_cn])
            #* self.CirAperN(0, 0, Ra, Dx, NP)
        )
        coord = np.array([])
        for r in range(mnE):
            coord = np.append(coord, [X0[r], Y0])
        p = 1
        X1 = np.array([])
        Y1 = np.array([])
        shy = 150
        while p <= NL:
            X1 = np.arange(-2 * NL + p, 2 * NL - p + 2, 2, dtype=int) * int(xf)
            Y1 = int(p * yf)
            for q in range(mnE - p):
                coord = np.append(coord, [X1[q], Y1])
                coord = np.append(coord, [X1[q], -Y1])
                # if p == 1 and q == 1: ### Take only one beam
                # uc0 = np.sqrt(g_amp)*GaussBeamNDefPsLw(lambda_,wx=wx,wy=wy,Rc=Rc,X0=X1[q],Y0=-Y1+shy,Dxy=Dx,NP=NP,m=None,zmm=7e6,P=P[0,n_cn])*CirAperN(X1[q],-Y1+shy,Ra,Dx,NP)
            p += 1
        return uc0, coord  # *Dx/NP

    def TiledAperture_beam(self,p_n,Z=None):
        # start = time.time()
        lambda_ = 1 * 1.064 * 1e-3  # in mm
        Rc = 1e25
        NP = self.im_size
        Dx = self.pix_size * NP
        m = 0
        Ra = (self.ra / 2) * 1e3
        D = self.d_ * 1e3
        a = D * NP / Dx  ## image coordinates in pixels
        mDr = 0
        w = 0.85 * Ra

        if self.n_channel == 7:
            NL = 1
        elif self.n_channel == 19:
            NL = 2
        elif self.n_channel == 37:
            NL = 3
        elif self.n_channel == 61:
            NL = 4
        elif self.n_channel == 91:
            NL = 5
        elif self.n_channel == 127:
            NL = 6
        elif self.n_channel == 217:
            NL = 8
        else:
            ValueError("Please provide correct channel number..")
        if Z is None:
            Z = self.Z
        zmm = Z* 1e3
        # fllens = 2e5
        zm = zmm * 1e-3
        phNs = p_n
        n_step = self.n_screens
        zsmm = zmm / (n_step + 1)
        w0 = w
        wx = w
        wy = w
        U, coord = self.sourceTAC_mod(
            lambda_,
            wx,
            wy,
            Ra,
            a,
            NL,
            Dx,
            NP,
            mDr,
            Rc,
            phNs,
            self.Kvar,
            self.trans_pn,
            self.amp_v,
            self.g_amp,
            self.n_channel,
            self.Pxx,
        )

        U_ = U.cpu().numpy()
        # pib_ = PIB(np.abs(U_)**2,1024,1024,1023,pix_size)
        # print('Input Power: ', np.real(pib_))
        # use_lens = False #True
        phy_x = Dx
        phy_y = Dx
        if self.use_lens:
            # U0 = PropAngSpecBandLimF(U,lambda_,phy_x,phy_y,100) #field just before lens (after 100 mm propagation)
            U1 = self.SphLens(
                U, phy_x, phy_y, NP, lambda_, self.fflens
            )  # field just after lens ##TODO: This might not be correct
            Up = self.PropAngSpecBandLimF(
                U1, lambda_, phy_x, phy_y, self.fflens
            )  # final field at focal plane
            # Up = PropAngSpecBandLimF_mod(U1,H2)#final field at focal plane
        else:
            Up = self.PropAngSpecBandLimF(
                U, lambda_, phy_x, phy_y, zmm
            )  # final field at far-field distance
            # Up = PropAngSpecBandLimF_mod(U,H1)  #final field at far-field distance

        # if atm is None:
        #   z_prop = zmm
        # else:
        #   z_prop = zsmm
        #   Up = PropAngSpecBandLimF(U,lambda_,phy_x,phy_y,z_prop)  #final field z_prop

        # if atm is not None:
        #   for ii in range(n_step):
        #     Uat = Up*T.exp(1j*atm[:,:,ii])
        #     Up = PropAngSpecBandLimF(Uat,lambda_,phy_x,phy_y,z_prop)

        Up_f = Up.cpu().numpy()
        Intf = np.abs(Up_f) ** 2

        # end = time.time() - start
        # print("TAC: ", str(end))
        return Up_f, coord.reshape(self.n_channel, 2).astype(int), U_

    def TiledAperture_2(self,p_n,thetar,thetad,Z=None):
        start = time.time()
        lambda_ = 1.064 * 1e-3
        Rc = 1e15
        NP = self.im_size
        Dx = self.pix_size * NP

        Dmm = self.d_ * 1e3
        RApmm = (self.ra / 2) * 1e3
        # w0=RApmm*0.85
        # wmm=w*1e3
        a = Dmm * NP / Dx
        m = 0
        thetar = thetar
        thetad = thetad

        if self.n_channel == 7:
            NL = 1
        elif self.n_channel == 19:
            NL = 2
        elif self.n_channel == 37:
            NL = 3
        elif self.n_channel == 61:
            NL = 4
        elif self.n_channel == 91:
            NL = 5
        elif self.n_channel == 127:
            NL = 6
        else:
            print("Please provide correct channel number..")

         # if n_channel == 7:
        #     pib_n = 387.52
        #     NL = 1
        # elif n_channel == 19:
        #     pib_n = 1042.077
        #     NL = 2
        # elif n_channel == 37:
        #     pib_n = 2015.09
        #     NL = 3
        # elif n_channel == 61:
        #     pib_n = 3297.03
        #     NL = 4
        # elif n_channel == 91:
        #     pib_n = 4875.546
        #     NL = 5
        # elif n_channel == 127:
        #     pib_n = 6710.44
        #     NL = 6
        # else:
        #     raise Exception("Enter Correct Number of channels")
        Z =  self.Z if Z is None else Z
        zmm = Z * 1e3
        zm = zmm * 1e-3
        phNs = p_n

        # U_ = U.cpu().numpy()
        # U = self.sourceTAC_final(lambda_,w0,Ra,a,NL,Dx,NP,mDr,zm,m,Rc,phNs,self.Kvar,self.trans_pn,\
        #                                self.amp_v,self.g_amp,self.n_channel)
        U = self.sourceTAC_final(
            lambda_,
            self.wx,
            self.wy,
            RApmm,
            a,
            NL,
            Dx,
            NP,
            zm,
            m,
            Rc,
            phNs,
            self.trans_pn,
            self.amp_v,
            self.g_amp,
            self.n_channel,
            thetar,
            thetad,
            self.Pxx,
        )
        U_ = U.cpu().clone().detach().numpy()
        # pib_ = PIB(np.abs(U_)**2,1024,1024,1023,pix_size)
        # print('Input Power: ', np.real(pib_))

        phy_x = Dx
        phy_y = Dx
        if self.use_lens:
            # U0 = PropAngSpecBandLimF(U,lambda_,phy_x,phy_y,100) #field just before lens (after 100 mm propagation)
            U1 = self.SphLens(U, phy_x, phy_y, NP, lambda_,self.fflens)  # field just after lens
            Up = self.PropAngSpecBandLimF(
                U1, lambda_, phy_x, phy_y, self.fflens
            )  # final field at focal plane
            # Up = PropAngSpecBandLimF_mod(U1,H2)#final field at focal plane
        else:
            Up = self.PropAngSpecBandLimF(
                U, lambda_, phy_x, phy_y, zmm
            )  # final field at far-field distance
            # Up = PropAngSpecBandLimF_mod(U,H1)  #final field at far-field distance

        Up1 = Up.cpu().clone().detach().numpy()
        Intf = Up1 * np.conj(Up1)
        phase = np.angle(Up1)

        end = time.time() - start
        # print("TAC: ", str(end))

        # return Up1, Intf
        return U_, np.abs(U_) ** 2, Intf, phase
    def Transverse_ph_abbrZP(self,lambda_,abr,th,rh,R):
        s = time.time()
        device = self.device
        rho = T.tensor(rh/R).to(device)
        th = T.tensor(th).to(device)
        abr = T.tensor(abr).to(device)
        ZP1 = 2*rho*T.sin(th)
        ZP2 = 2*rho*T.cos(th)

        ZP3 = np.sqrt(3)*(2*(rho**2)-1)

        ZP4 = np.sqrt(6)*(rho**2*T.sin(2*th))
        ZP5 = np.sqrt(6)*(rho**2*T.cos(2*th))

        ZP6 = np.sqrt(8)*((3*rho**3 - 2*rho)*T.sin(th))
        ZP7 = np.sqrt(8)*((3*rho**3 - 2*rho)*T.cos(th))

        ZP8 = np.sqrt(8)*(rho**3*T.sin(3*th))
        ZP9 = np.sqrt(8)*(rho**3*T.cos(3*th))

        ZP10 = np.sqrt(5)*(6*(rho**4)-6*(rho**2)+1)

        ZPt = abr[0]*ZP1 + abr[1]*ZP2 + abr[2]*ZP3 + abr[3]*ZP4 +\
            abr[4]*ZP5 + abr[5]*ZP6 + abr[6]*ZP7 + abr[7]*ZP8 +\
                 abr[8]*ZP9 + abr[9]*ZP10
        ZPt[rho>1] = 0
        AbPh = T.exp(1j*(2*np.pi/lambda_)*ZPt)
        e = time.time()-s
        # print('Transverse_ph_abbrZP:',str(e),'s')
        return AbPh


    def get_ff_pib(self,U,coord: list,noise: np.ndarray,mask,pin_total,thetar=None,th2d=None,fllens=None,use_lens=True,rp=None,plot=False,fig=None):
        phsm = noise
        device = self.device
        Dx = self.pix_size * self.im_size ## Physical Parameters
        NP = self.im_size ## Number of Points/Pixels in the image plane
        Z = self.Z 
        k = 2 * np.pi / 1.064e-3  ## Wavelength in mm ## Physical Parameters
        print(f"Z : {Z} , pix_size : {(Dx/NP)} , k : {k}\n")
        mask = T.tensor(mask).to(device)
        # Cicrc = T.tensor(Circ3).to(device)
        U_ = T.zeros_like(U).to(self.device)
        thetar = np.radians(thetar)
        th2d = np.radians(th2d)
        phNs = T.tensor(noise).to(self.device)
        Xs, Ys = U.shape
        # X, Y = T.meshgrid(T.arange(Xs).to(device) - Xs // 2, T.arange(Ys).to(device) - Ys // 2) ## Centered at zero , to add the tilt phase
        X, Y = np.meshgrid(np.arange(Xs) - Xs // 2, np.arange(Ys) - Ys // 2)  ## Centered at zero , to add the tilt phase
        X, Y = X * (Dx / NP), Y * (Dx / NP)
        fllens = self.fflens if fllens is not None else fllens
        d2 = fllens ## At the Focal Plane 
        d1 = 0
        points = []
        ubb_list = []
        udebug = None
        if (self.use_lens is False):  ### With no lens , shifting the beam in the far field for pointing error , by thetar, th2d
            cx = (np.tan(thetar) * Z * 1e3 * np.cos(th2d) * (NP / Dx)).T
            # cx = (np.tan(thetar)*Z* np.cos(th2d)).T ## Physical Shift in x and y
            cy = (np.tan(thetar) * Z * 1e3 * np.sin(th2d) * (NP / Dx)).T
            # cy = (np.tan(thetar)*Z* np.sin(th2d)).T
            kx = k * np.sin(thetar) * np.sin(th2d)[0]  ## tilt phase in mm
            ky = k * np.sin(thetar) * np.cos(th2d)[0]  ## tilt phase in mm
            # corrected_coord = [(1,0),(1,1),(1,2),(0,0),(0,2),(2,0),(2,2)]
            # ubb_list = []
            for idx, c in enumerate(coord):  # 7
                # TODO : Correct for the Propagation Delays in the other beams
                ubb = T.exp(
                    T.tensor(1j * kx[idx] * (X + (c[0] + cx[idx]) * Dx / NP)).to(
                        device
                    )
                ) * T.exp(
                    T.tensor(1j * ky[idx] * (Y + (c[1] + cy[idx]) * Dx / NP)).to(
                        device
                    )
                )
                # ubb = T.exp(T.tensor(1j * kx[idx] * (X)).to(device)) * T.exp(T.tensor(1j * ky[idx] * (Y)).to(device))
                # ax = fig2.add_subplot(int(f"33{corrected_coord[idx]}"))
                # ax2[corrected_coord[idx]].imshow(np.angle((ubb).cpu().numpy()),cmap='viridis')
                # ax2[corrected_coord[idx]].set_title(f'Beam {corrected_coord[idx]}')
                ubb_list.append((ubb).cpu().numpy())
                ### Shift and Tilt Phase
                U_ += (
                    T.roll(
                        U, shifts=(c[0] + int(cx[idx]), (c[1] + int(cy[idx]))), dims=(1, 0)
                    )
                    * ubb 
                    * T.exp(1j * phNs[idx])
                )
                ### only Tilt Phase
                # U_ += U*ubb*T.exp(1j*phNs[idx])
                ## only shift
                # U_ += T.roll(U,shifts=(c[1]+int(cy[idx]),c[0]+int(cx[idx])),dims=(0,1)) * T.exp(1j * phNs[idx])
            pib = (
                (
                    T.real(T.sum((T.abs(U_) ** 2 * mask) * ((Dx / NP) * 1e-3) ** 2))
                    / pin_total
                )
                .cpu()
                .numpy()
            )
            # plot = False
            udebug = None
            if plot:
                # fig = plt.figure(figsize=(10, 8))
                ax1 = fig.add_subplot(121)
                udebug = (T.abs(U_) ** 2).cpu().numpy()
                # ax1.imshow(udebug,cmap='viridis')
                ax1.add_patch(
                    plt.Circle((NP / 2, NP / 2), rp, fill=False, color="r")
                )
                ax1.add_patch(
                    plt.Circle((NP / 2, NP / 2), rp * 4, fill=False, color="g")
                )
                ax1.text(NP // 2, 0, f"PIB Value: {pib}", color="r", fontsize=12)
                # ax1.colorbar()
                for i, c in enumerate(coord):
                    ax1.scatter(
                        c[0] + NP // 2 + int(cy[i]),
                        c[1] + NP // 2 + int(cx[i]),
                        color="r",
                    )
                    points.append(
                        (c[0] + NP // 2 + int(cx[i]), c[1] + NP // 2 + int(cy[i]))
                    )
                    # plt.xlim(NP//2-100,NP//2+100)
                    # plt.ylim(NP//2-100,NP//2+100)
                ax1.imshow(udebug, cmap="viridis")
                # plt.show()
                # plt.cla()
            return pib, udebug, points, ubb_list
        elif self.use_lens is True:  ### Lens Case
            zmm = Z * 1e3  ## Z in mm
            Cx = (
                np.tan(thetar) * zmm * np.cos(th2d) * (NP / Dx)
            )  ## Shift intented in Far field in pixels
            Cy = np.tan(thetar) * zmm * np.sin(th2d) * (NP / Dx)
            ## cx , cy will be calculated using the lens formula
            thetax = np.arctan(Cx / (zmm * (NP / Dx))).T
            thetay = np.arctan(Cy / (zmm * (NP / Dx))).T
            thetax[1] = 0 ## No Pointing Error for the first beam
            thetay[1] = 0
            for idx, c in enumerate(coord):
                print(thetax.shape)
                cx = (1 - (d2 / fllens)) * c[0] + (fllens * NP / Dx) * (
                    thetax
                )  ## R shift in x and y pixels
                cy = (1 - (d2 / fllens)) * c[1] + (fllens * NP / Dx) * (thetay) #There was an error here , corrected
                kx = k * np.sin(
                    ((-(c[0] * Dx / NP) / fllens) + (1 - (d1 / fllens)) * thetax)
                )  ## Needs to cal in physical units of mm
                ky = k * np.sin(
                    ((-(c[1] * Dx / NP) / fllens) + (1 - (d1 / fllens)) * thetay)
                )
                ## Debug Print
                print(
                    f"cx: {cx[idx]},cy:{cy[idx]} , kx: {kx.shape}"
                )  # , c : {c[0]*Dx/NP}, {c[1]*Dx/NP}')
                # print(f'x:{(c[0]+cx[idx])*Dx/NP}, y:{(c[1]+cy[idx])*Dx/NP}')
                # ubb = T.exp(T.tensor(1j * kx[idx] * (X)).to(device)) * T.exp(T.tensor(1j * ky[idx] * (Y)).to(device)) ## Tilt Phase in x and y , shifted to c[0],c[1] #FIXME : Why only X and Y works , and not the shifted one , which has a higher magnitude
                ubb = T.exp(
                    T.tensor(-1j * kx[idx] * (X - int(cx[idx]) * Dx / NP)).to(device) #FIXME: Findout why there is a negative sign , refer source TAC final
                ) * T.exp(
                    T.tensor(1j * ky[idx] * (Y - int(cy[idx]) * Dx / NP)).to(device)
                )  ## Tilt Phase in x and y , shifted to c[0],c[1]
                ## Tilt and Shifted
         
                U_ += (T.roll(U, shifts=(int(cx[idx]), int(cy[idx])), dims=(1, 0)) * ubb * T.exp(1j * phNs[idx]))  # Tilt and Shifted 
                
                # U_ += T.roll(U*0,shifts=(int(cx[idx]),int(cy[idx])),dims=(1,0))#*ubb* T.exp(1j * phNs[idx])#Tilt and Shifted #FIXME: why is the shift in physical units of mm
                #  cond = 0
                # U_ += T.roll(U,shifts=(0,0),dims=(1,0)) * ubb * T.exp(1j * phNs[idx]) #Tilt  #FIXME: why is the shift in physical units of mm
                # ubb_list.append((ubb).cpu().numpy())
                
            pib = ((T.real(T.sum((T.abs(U_) ** 2 * mask) * (Dx / NP * 1e-3) ** 2))/ pin_total).cpu().numpy())
            
            if plot:
                ax1 = fig.add_subplot(221)
                udebug = (T.abs(U_) ** 2).cpu().numpy()
                ax1.imshow(udebug, cmap="viridis")
                # ax1.add_patch(plt.Circle((NP/2,NP/2),rp,fill=False,color='r'))
                # ax1.add_patch(plt.Circle((NP/2,NP/2),rp*4,fill=False,color='g'))
                # ax1.annotate(f"PIB Value: {str(pib)[:5]},\nPIN {str(pin_total)[:5]} \n absPIB : {str(pin_total*pib)[:5]}",xy=(NP//2,NP//2),xytext=(NP//2,NP//2+NP//22),arrowprops=dict(facecolor='black',arrowstyle='->'))
                ax1.set_title("FF addition Intensity")
                # for i,c in enumerate(coord):
                #   ax1.scatter(c[0]+NP//2+int(cy[i]),c[1]+NP//2+int(cx[i]),color='r')
                #   points.append((c[0]+NP//2+int(cx[i]),c[1]+NP//2+int(cy[i])))
                ax1.set_xlim(NP // 2 - 100, NP // 2 + 100)
                ax1.set_ylim(NP // 2 - 100, NP // 2 + 100)
                ax2 = fig.add_subplot(222)
                ax2.imshow(np.angle((U_).cpu().numpy()), cmap="viridis")
                # ax2.set_title('Phase Not Zoomed')
                ax2.set_title("FF Phase Profile")
                # ax4 = fig.add_subplot(224)
                # ax4.imshow(np.angle(U_.cpu().numpy()),cmap='viridis')
                # ax4.set_title('Phase Zoomed')
                ax2.set_xlim(NP // 2 - 100, NP // 2 + 100)
                ax2.set_ylim(NP // 2 - 100, NP // 2 + 100)
                # ax1.colorbar()
                # plt.show()
                # plt.cla()
            return pib, U_.cpu().numpy(), points, ubb_list
    
    def get_lens_coordinates(self,x, y, fllens, im_size, Dx, d2, thetax, thetay):

        cx = (1 - (d2 / fllens)) * x + (fllens * im_size / Dx) * (thetax)  ## R shift in x and y pixels
        cy = (1 - (d2 / fllens)) * y + (fllens * im_size / Dx) * (thetay)
        return cx, cy
    
    def abberations_tolerance_setup(self,):
        pass