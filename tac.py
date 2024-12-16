# %%
from common_imports import *
# %%

class TiledApertureBeamPropFast:

    def __init__(self,im_size,pix_size,n_channel,p_n,Kvar,Z,trans_pn,amp_v,g_amp,ra,d_):

        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        print(f"Using {self.device} device")

        self.im_size = im_size
        self.pix_size = pix_size
        self.n_channel = n_channel
        self.p_n = p_n
        self.Kvar = Kvar
        self.Z = Z
        self.trans_pn = trans_pn
        self.amp_v = amp_v
        self.g_amp = g_amp
        self.ra = ra
        self.d_ = d_


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
        x = T.arange((-M/2 - X1)*dx, (M/2 - X1)*dx,dx).to(self.device)
        N=NP
        dy = Dx/N
        y = T.arange((-N/2 - Y1)*dy, (N/2 - Y1)*dy,dy).to(self.device)
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
    
    def PropAngSpecBandLimF(self,uin,L,Dx,Dy,zmm): #TODO: To modify this function , saving H kernel
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

        uout = ifft2(ifftshift((fftshift(fft2(layer)))*H1))

        return uout
    
    def SphLens(self,uin,L1,L2,NP,lambda_,zf):
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

        Returns:
        - uout (torch tensor): Output wavefront after passing through the lens.

        This function simulates the propagation of a wavefront through a spherical lens.
        It applies a phase shift to the input wavefront based on the lens properties
        (focal length and aperture size) to simulate the lens effect. The input parameters
        include the input wavefront (uin), length of the input and output planes (L1 and L2),
        number of points for discretization (NP), wavelength of the wavefront (lambda_), and
        focal length of the lens (zf). The output is the wavefront after passing through
        the lens, returned as a torch tensor.
        """
        k=2*np.pi/lambda_

        dy=L1/NP
        y=T.arange(-L1/2,L1/2,dy).to(self.device)

        dx=L2/NP
        x=T.arange(-L2/2,L2/2,dx).to(self.device)

        X,Y=T.meshgrid(x,y,indexing='xy')
        uout=T.multiply(uin,T.exp(-1j*(k/(2*zf))*(X**2+Y**2)))
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

        U,coord = self.sourceTAC_final(lambda_,w0,Ra,a,NL,Dx,NP,mDr,zm,m,Rc,phNs,self.Kvar,self.trans_pn,\
                                       self.amp_v,self.g_amp,self.n_channel)
        U_ = U.cpu().numpy()
        # pib_ = PIB(np.abs(U_)**2,1024,1024,1023,pix_size)
        # print('Input Power: ', np.real(pib_))

        phy_x = Dx
        phy_y = Dx

        Up = self.PropAngSpecBandLimF(U,lambda_,phy_x,phy_y,zmm)  #final field z_prop

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
        return Circ


    def PIB_loop(self,uin,Circ,pix_size):
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
        IntfCir = uin*Circ
        IntfCir1 = IntfCir*(pix_size*1e-3)**2
        P = np.sum(IntfCir1)
        uout = P
        return np.real(uout)

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
class TiledApertureBeamProp

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
        x = T.arange((-M/2 - X1)*dx, (M/2 - X1)*dx,dx).to(self.device)
        N=NP
        dy = Dx/N
        y = T.arange((-N/2 - Y1)*dy, (N/2 - Y1)*dy,dy).to(self.device)
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

        uout = ifft2(ifftshift((fftshift(fft2(layer)))*H1))

        return uout
    
    def SphLens(self,uin,L1,L2,NP,lambda_,zf):
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

        Returns:
        - uout (torch tensor): Output wavefront after passing through the lens.

        This function simulates the propagation of a wavefront through a spherical lens.
        It applies a phase shift to the input wavefront based on the lens properties
        (focal length and aperture size) to simulate the lens effect. The input parameters
        include the input wavefront (uin), length of the input and output planes (L1 and L2),
        number of points for discretization (NP), wavelength of the wavefront (lambda_), and
        focal length of the lens (zf). The output is the wavefront after passing through
        the lens, returned as a torch tensor.
        """
        k=2*np.pi/lambda_

        dy=L1/NP
        y=T.arange(-L1/2,L1/2,dy).to(self.device)

        dx=L2/NP
        x=T.arange(-L2/2,L2/2,dx).to(self.device)

        X,Y=T.meshgrid(x,y,indexing='xy')
        uout=T.multiply(uin,T.exp(-1j*(k/(2*zf))*(X**2+Y**2)))
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
    
    def TiledAperture_2(self, p_n):
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
