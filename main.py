# %%
from common_imports import *
from utilities import Utilities
from tac import TiledApertureBeamProp
# %%
device = T.device('cuda' if T.cuda.is_available() else 'cpu')
# %%
if __name__ == '__main__':

    n_channel = 7#int(input('Enter Number of channels: '))
    n_img = 25e3#float(input('Enter n_iter: '))
    n_img = int(n_img)
    im_size = int(4096)
    pix_size = 1
    plt.style.use('default')

    utilities = Utilities() 
    #   U = T.zeros(im_size,im_size,dtype=T.cfloat).to(device)

    fs = 5e6/20
    delt = 1/fs
    cyc = n_img/fs
    t = np.arange(0,cyc-delt,delt)
    L = 5e3

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
    # amp_v = 0 #Uncomment for Ideal case

    noise_env = utilities.env_pn(n_channel, cyc, delt)
    noise_env = np.transpose(noise_env)
    p_n = 2*noise_env-np.pi
    p_n[2,:] = 0
    # Random DC phase offset
    for i in range(n_channel):
        p_n[i,:] += np.pi*np.random.rand()

    # p_n = np.zeros((n_img,n_channel)) #Uncomment for ideal case


    V = np.random.rand(n_channel) #Control Voltage initialization
    # cc=1

    d = 10e-3
    a = 9.41e-3

    # d = 3e-3
    # a = 2.94e-3

    Kvar = 0.0
    trans_pn = np.random.rand(n_channel,10)

    L_0 = 10
    l_0 = 10e-3
    N_ss = 500
    w_l = 1064e-9
    c_n2 = 1e-14
    k = 2*np.pi/w_l
    n_screens = utilities.num_of_screen(k, c_n2, L, L) #int(10)
    # Z = 25e3
    if n_channel==7:
        pib_n = 396.76
        NL=1
    elif n_channel==19:
        pib_n = 1059.827
        NL=2
    elif n_channel==37:
        pib_n = 2023.937
        NL=3
    elif n_channel==61:
        pib_n = 3297.03
        NL=4
    elif n_channel==91:
        pib_n = 4875.546
        NL=5
    elif n_channel==127:
        pib_n = 6710.44
        NL=6
    elif n_channel==217:
        NL=8
        pib_n=1
    else:
        raise ValueError("Enter Correct Number of channels..")
    print('chosen NL:',NL)
    Z = np.round(7*(a+2*NL*d)**2/w)
    print('propogating ',Z,'m')
    r = 1.22*w*Z/(a+2*NL*d)

    rp = round(r/(pix_size*1e-3))
    theta = np.arange(0,2*np.pi+np.pi/36,np.pi/36)
    x = (im_size)/2 + rp*np.cos(theta)
    y = (im_size)/2 + rp*np.sin(theta)

    r_c = 0.184*((c_n2*Z/n_screens)/w_l**2)**(-3/5)
    turb = True
    if turb == True:
        print('Number of screens:',n_screens)
        atm = np.zeros((im_size,im_size,n_screens))
        for jj in range(n_screens):
            atm[:,:,jj] = utilities.ss_turb(im_size,N_ss,l_0,L_0,r_c)
        atm = T.tensor(atm).to(device)
    else:
        atm = None

    run = 'cl'#input('Run Mode: ')
    # if run == 'cl':
    #   G = float(input('Enter gain: '))
    pib_val = []
    Tac = TiledApertureBeamProp(im_size,pix_size,n_channel,np.zeros(n_channel),n_screens,Kvar,Z,trans_pn,atm,amp_v,g_amp,a,d)
    U_in,Up,coord = Tac.TiledAperture_2()
    I = Tac.get_ff(Up,coord,0*np.random.randn(n_channel))
    pib_n = np.ceil(Tac.PIB(I,im_size/2, im_size/2,rp,pix_size))
    # masked_pib_n = np.ceil(masked_PIB(I,im_size/2, im_size/2,0.5*rp,pix_size))
    print('PIB ideal: ',pib_n)
    # print('Masked PIB ideal: ', masked_pib_n)

    f_loop = fs#5e4
    f_ctrl = round(fs/f_loop)
    print('Control rate: ', f_ctrl)
# %%
plt.imshow(I,cmap='jet')
plt.show()

# %%
