# %%
from .common_imports import *
from scipy.io import loadmat
import importlib.resources as pkg_resources
# %%
class Utilities:
    def __init__(self):
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
    
    def calc_ampv(self):
        df = 2e6
        w = 1064e-9
        vpi = 1.8
        c = 2.99e8

        u = c/w
        nesp = 1
        h = 6.626e-34
        P0 = 100
        g_amp = 1e4*P0/100
        del_lamb = 10e-9
        o_bw = c*del_lamb/(w**2)

        amp_pn = (2*nesp*h*u*(g_amp-1)*o_bw)
        amp_v = np.sqrt(amp_pn)
        return amp_v, g_amp
    
    def get_ff_distance_and_bucket_size(self, n_channel, a,d,lens=False,fflens=None):
        """
        Calculate the far-field distance and bucket size for a given number of channels. The far-filed distance 
        is 10 times the Fronhauffer distance, and the bucket size is calculated based on the number of channels.
        
        Inputs:
        - n_channel (int): Number of channels.
        - a (float): Aperture size.
        - d (float): center-to-center distance between beams.

        """
        w = 1064e-9
        if n_channel==7:
            NL=1
        elif n_channel==19:
            NL=2
        elif n_channel==37:
            NL=3
        elif n_channel==61:
            NL=4
        elif n_channel==91:
            NL=5
        elif n_channel==127:
            NL=6
        elif n_channel==217:
            NL=8
        else:
            raise ValueError("Enter Correct Number of channels..")
        print('chosen NL:',NL)
        Z = np.round(10*(a+2*NL*d)**2/w)
        print('propogating ',Z,'m')
        r = 1.22*w*Z/(a+2*NL*d)
        if lens:
            r = (4 / np.pi) * w * fflens*1e-3 / (a + 2 * NL * d)
            return fflens,r
        return Z,r

    def env_pn(self, n_channel, cyc, delt,streams=None,filename=None):
        """
        Generate phase noise patterns based on environmental parameters.

        Parameters:
        - n_channel (int): Number of channels.
        - cyc (int): Number of cycles in the phase noise pattern.
        - delt (float): Time interval between samples.
        - streams (int): To make sure the same phase noise pattern in produced , random number generator

        Returns:
        - P1 (numpy array): Phase noise patterns for each channel.

        This function generates phase noise patterns based on environmental parameters.
        It loads environmental data from a file, interpolates the data using cubic spline
        interpolation, and generates phase noise patterns for each channel. The input parameters
        include the number of channels (n_channel), number of cycles in the phase noise pattern (cyc),
        and time interval between samples (delt). The function returns the phase noise patterns (P1)
        as a numpy array.
        """
        if filename is None:
            with pkg_resources.open_binary('tac_lib.data', 'new_spline_fit.mat') as file: 
                data = loadmat(file)
        else:
            try :
                data = loadmat(filename)
            except FileNotFoundError:
                print('File not found')
                return None
        
        env_t = data['env_t']
        env_1 = data['env_1']
        f = interp1d(env_t[:,0], env_1[:,0], kind='cubic')
        t_samp = np.arange(-10,-10+cyc,delt)
        env_n = f(t_samp)
        fft_env1 = fft(env_n)

        P1a = np.ones((n_channel,len(env_n)), dtype=np.cdouble)
        for ii in range(n_channel):
            
            if streams is not None:
                rng = streams[ii]
                P1a[ii,:] = ifft(fft_env1*np.exp(1j*2*np.pi*rng.uniform(1,len(env_n))))
            else:
                P1a[ii,:] = ifft(fft_env1*np.exp(1j*2*np.pi*np.random.rand(1,len(env_n))))

        P1b = np.zeros((n_channel,len(env_n)))
        for ii in range(n_channel):
            P1b[ii,:] = (np.abs(P1a[ii,:])/np.mean(np.abs(P1a[ii,:])))-1

        P1b_ = np.abs(P1b)
        max_array = P1b_.max(axis=1)
        P1 = np.arccos(np.transpose(P1b)/max_array)

        return P1

    def ss_turb(self,im_size, N_ss, l_0, L_0, r_c,rng=None):
        """
        Simulate atmospheric turbulence using the von Karman spectrum.

        Parameters:
        - im_size (int): Size of the turbulence image.
        - N_ss (int): Number of samples. (500 - 1000) beyond 1k no change
        - l_0 (float): Outer scale of turbulence. Minimum vortex Size 
        - L_0 (float): Integral length scale of turbulence. Largest Vortex Size 
        - r_c (float): Correlation radius.
        - rng (generator): Default is None , For Reproducibility 

        Returns:
        - si_n (numpy array): Simulated turbulence phase screen.

        This function simulates atmospheric turbulence using the von Karman spectrum.
        It generates turbulence phase screens by modeling the turbulence spectrum and
        calculating the phase screen based on random phase shifts. The input parameters
        include the size of the turbulence image (im_size), number of samples (N_ss),
        outer scale of turbulence (l_0), integral length scale of turbulence (L_0),
        and correlation radius (r_c). The function returns the simulated turbulence
        phase screen (si_n) as a numpy array.
        """
        n = np.arange(0,N_ss)

        kapp_0 = 2*np.pi/L_0
        kapp_m = 2*np.pi/l_0

        k_max = 2*kapp_m
        k_min = kapp_0

        K_n = k_min*np.exp((n/N_ss)*np.log(k_max/k_min))
        zeta = rng.uniform(1, len(list(K_n))) if rng is not None else np.random.rand(1, len(list(K_n)))
        k_n = np.zeros((1,len(list(K_n))))
        k_n[0,0] = rng.uniform() if rng is not None else np.random.rand(1)

        for ii in range(1,len(list(K_n))):
            k_n[0,ii] = np.sqrt(K_n[ii-1]**2 + zeta[0,ii]*(K_n[ii]**2 - K_n[ii-1]**2))
        alp = 5/3
        c = (alp*2**(alp-2)*math.gamma(1+alp/2))/(np.pi*math.gamma(1-alp/2))
        func = lambda p: (p*c*r_c**(-alp)*np.exp(-p**2/kapp_m**2))/((p**2 + kapp_0**2)**(1+alp/2))
        phi = np.zeros((1,len(list(K_n))))

        phi[0,0],_ = integral.quad(func,K_n[0],K_n[1])
        for jj in range(1,len(list(K_n))):
            phi[0,jj],_ = integral.quad(func,K_n[jj-1],K_n[jj],epsrel=1.0e-3)

        alpha = rng.standard_normal(1,len(list(n))) if rng is not None else np.random.randn(1,len(list(n)))#FIXME Same Random Seed ? 
        beta = rng.standard_normal(1,len(list(n))) if rng is not None else np.random.randn(1,len(list(n)))
        s_n = 2*np.pi*phi
        a_n = np.sqrt(s_n)*(alpha + 1j*beta)
        a = T.tensor(np.diag(np.transpose(a_n)[:,0])).to(self.device)
        theta = 2*np.pi*(rng.uniform(1,len(list(n))) if rng is not None else np.random.rand(1,len(list(n))))
        X = np.arange(0,1,1/im_size)
        Y = np.arange(0,1,1/im_size)
        X = X[:,np.newaxis]
        Y = Y[:,np.newaxis]
        x_n = T.tensor(np.exp(1j*X*(k_n*np.cos(theta)))).to(self.device)
        y_n = T.tensor(np.exp(1j*np.transpose(k_n*np.sin(theta))*np.transpose(Y))).to(self.device)

        si_n1 = T.matmul(x_n,a).to(self.device)
        si_n1 = T.matmul(si_n1,y_n).to(self.device)
        si_n = T.real(si_n1).to(self.device)

        return si_n.cpu().clone().detach().numpy()
    
    def ZernCoeff_TransvPhs(self,RWEH, RWEL, NumZP,n_channel):
        KvarH = RWEH**2
        KvarL = RWEL**2
        abrZPC = np.zeros((n_channel, NumZP))
        for nn in range(n_channel):
            ZH = np.random.rand(1, 8)
            WvarH = np.sum(ZH**2)
            ZHn = np.sqrt(KvarH / WvarH) * ZH
            # WvarH = np.sum(ZHn ** 2)
            ZL = np.random.rand(1, 2)
            WvarL = np.sum(ZL**2)
            ZLn = np.sqrt(KvarL / WvarL) * ZL
            # WvarL = np.sum(ZLn ** 2)
            Rabr = np.zeros(NumZP)
            Rabr[0:2] = ZLn.flatten()
            Rabr[2:NumZP] = ZHn.flatten()
            abrZPC[nn, :] = Rabr
        return abrZPC
    def find_centroid_coord(self,matrix):
        rows, cols = np.indices(matrix.shape)
        centroid_x = int(np.average(cols, weights=matrix))  ### Rasises Zero Division Error
        centroid_y = int(np.average(rows, weights=matrix))
        return centroid_x, centroid_y
    def find_max_coordinates(self,matrix):
        flattened_index = np.argmax(matrix)
        rows, cols = np.unravel_index(flattened_index, matrix.shape)
        return rows, cols
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
        # print('shape:',shape)
        sx,sy = shape#[0],shape[1]
        x = np.arange(sx)
        y = np.arange(sy)

        X,Y = np.meshgrid(x,y)
        Circ = np.ones((sx,sy))
        R = np.sqrt((X-Xc)**2 + (Y-Yc)**2)
        Circ[R>Roc] = 0
        return Circ
    
    def plot_pib(self,pib_val, pib_n, title=None, name=None, t=None):
        """
        Plot the normalized Point Intensity Behavior (PIB) values over time or steps.

        Parameters:
        - pib_val (list or numpy array): List or array containing the PIB values.
        - pib_n (float): Normalization factor for PIB values.
        - title (str): Title of the plot (optional).
        - name (str): File name to save the plot as an image (optional).
        - t (list or numpy array): Time or step values corresponding to PIB values (optional).

        Returns:
        - None

        This function generates a plot of the normalized PIB values over time or steps.
        It plots the PIB values against time or steps and includes horizontal lines
        representing the mean, mean plus one standard deviation, and mean minus one
        standard deviation of the PIB values. The plot is styled with a ggplot style
        and saved as an image if a file name is provided.
        """

        pib_val=np.array(pib_val)/pib_n
        plt.figure(figsize=(10,7))
        plt.style.use('ggplot')
        if t is None:
            t = np.arange(len(pib_val))
        plt.plot(t,pib_val,linewidth=1.5,color='b')
        # Get the current axis
        ax = plt.gca()

        # Instantiate the ScalarFormatter
        formatter = ScalarFormatter()

        # Set size thresholds for scientific notation
        formatter.set_powerlimits((0, 0))

        # Set the formatter for the major ticker
        ax.xaxis.set_major_formatter(formatter)

        plt.axhline(np.mean(pib_val),linewidth=1.5, linestyle='dashed',color='m',\
                    label='$\mu:${_i}'.format(_i=np.round(np.mean(pib_val),2)))
        plt.axhline(np.mean(pib_val)+np.std(pib_val),linewidth=1.5, linestyle='dashed',\
                    color='k',label='$\mu+\sigma:${_i}'.format(_i=np.round(np.mean(pib_val)+np.std(pib_val),2)))
        plt.axhline(np.mean(pib_val)-np.std(pib_val),linewidth=1.5, linestyle='dashed',\
                    color='r',label='$\mu-\sigma:${_i}'.format(_i=np.round(np.mean(pib_val)-np.std(pib_val),2)))
        if t is None:
            plt.xlabel('Steps',fontsize=14,fontweight='bold',color='k')
        else:
            plt.xlabel('Time (s)',fontsize=14,fontweight='bold',color='k')

        plt.ylabel('Norm. PIB',fontsize=14,fontweight='bold',color='k')
        # plt.grid()
        plt.xticks(fontsize=14,fontweight='bold',color='k')
        plt.yticks(fontsize=14,fontweight='bold',color='k')
        if title is not None:
            plt.title(title,fontsize=16,fontweight='bold')
        plt.legend(fontsize=14)
        plt.ylim(0,1.2)
        if name is not None:
            plt.savefig(name+'.png')
        plt.style.use('default')

    
    def plot_phase_screens(self,atm, n_screens, cn2=None):
        """
        Plot phase screens generated from atmospheric turbulence.

        Parameters:
        - atm (numpy array): Array containing phase screens.
        - n_screens (int): Total number of phase screens.
        - cn2 (float): Logarithmic value of the atmospheric turbulence strength Cn2 (default=-15).

        Returns:
        - None

        This function generates a grid of subplots displaying phase screens obtained from atmospheric turbulence.
        Each subplot represents a phase screen, and the colormap 'jet' is used to visualize the phase values.
        The function creates a color bar for each subplot to indicate the corresponding phase values.
        """
        plt.style.use('default')
        n_samples = int(n_screens/2)
        fig, ax = plt.subplots(n_samples, 2, figsize=(10,5*n_samples))
        fig.tight_layout()
        plt.axis('off')
        c=0
        if cn2 is not None:
            fig.suptitle('Phase screens for $C_{n^2} = 10^{Cn2}$'.format(Cn2=cn2), fontsize=16,fontweight='bold')
        for i in range(n_samples):
            for j in range(2):
                im = ax[i,j].imshow(atm[:,:,c],cmap='jet')
                fig.colorbar(im, ax=ax[i,j])
                ax[i,j].set_title('Phase screen {_i}'.format(_i=c),fontweight='bold')
            c+=1 
        plt.show()
    
    def num_of_screen(self,k, Cn2, L, del_z):
        """
        Calculate the number of atmospheric screens required for wave propagation.

        Parameters:
        - k (float): Wavenumber.
        - Cn2 (float): Atmospheric turbulence strength.
        - L (float): Outer scale of turbulence.
        - del_z (float): Distance between screens.

        Returns:
        - num (int): Number of atmospheric screens.

        This function calculates the number of atmospheric screens required for wave propagation
        based on the specified parameters. It iteratively adjusts the distance between screens
        until the total Rytov variance is within 10% of the required value. Then, it calculates
        the number of screens needed to cover the specified outer scale of turbulence (L) using
        the adjusted distance between screens. The function returns the calculated number of screens (num).
        """

        rytov_total = 1.23 * Cn2 * k ** (7 / 6) * L ** (11 / 6)
        rytov_total = 0.1 * rytov_total
        donet = 0

        while (not donet):
            rytov_delz = 1.23 * Cn2 * k ** (7 / 6) * del_z ** (11 / 6)
            if rytov_delz < 0.1:
                donet = 1
            else:
                del_z = del_z - 10

        donet = 0
        while (not donet):
            if rytov_delz < rytov_total:
                donet = 1
            else:
                del_z = del_z - 10
                rytov_delz = 1.23 * Cn2 * (k ** (7 / 6)) * (del_z ** (11 / 6))        #intensity variance in del_z distance <0.1total rytov variance

        num = 2*math.ceil(L / del_z)
        return num
# %%
