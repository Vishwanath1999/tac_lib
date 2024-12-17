# %% Code to simulate the TAC with only longitudinal phase noise
from common_imports import *
from utilities import Utilities
from tac import TiledApertureBeamPropFast
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

    d = 10e-3 # center to center distance between beams
    a = 9.41e-3 #aperture size

    utilities = Utilities() 
    #   U = T.zeros(im_size,im_size,dtype=T.cfloat).to(device)

    fs = 5e6/20
    delt = 1/fs
    cyc = n_img/fs
    t = np.arange(0,cyc-delt,delt)
    L = 5e3

    amp_v,g_amp = utilities.calc_ampv()
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

    Kvar = 0.0
    trans_pn = np.random.rand(n_channel,10)

    w_l = 1064e-9
    k = 2*np.pi/w_l
    # Z = 25e3
    
    Z,r = utilities.get_ff_distance_and_bucket_size(n_channel, a, d)

    rp = round(r/(pix_size*1e-3))
    theta = np.arange(0,2*np.pi+np.pi/36,np.pi/36)
    x = (im_size)/2 + rp*np.cos(theta)
    y = (im_size)/2 + rp*np.sin(theta)

    lambda_ = 1.68 * 1e-3 
    phyx = pix_size*im_size
    run = 'cl'#input('Run Mode: ')
    # if run == 'cl':
    #   G = float(input('Enter gain: '))
    pib_val = []
    Tac = TiledApertureBeamPropFast(im_size,pix_size,n_channel,np.zeros(n_channel),Kvar,Z,trans_pn,amp_v,g_amp,a,d)
    H1 = Tac.PropAngSpecBandLimF_kernel((im_size,im_size),lambda_,phyx,Z) ## For Loop 
    U_in,Up,coord = Tac.TiledAperture_2() ## Normal 
    U_in,Up,coord = Tac.TiledAperture_mod(H1) ## For Loop 
    ## Alternatively 
    Tac.iterations_setup() ### Run this at the start of the loop 
    Uin,Up,coord = Tac.TiledAperture_mod()
    I = Tac.get_ff(Up,coord,0*np.random.randn(n_channel))
    pib_n = np.ceil(Tac.PIB(I,im_size/2, im_size/2,rp,pix_size)) ## Recommended for single execution 
    roiMask = Tac.CircMask(I.shape,im_size/2,im_size/2,rp) ## For many iterations , define the mask 
    pib_n = Tac.PIB_loop(I,roiMask) ## and loop only this (when using different masks)
    ### Alternatively 
    pib_n = Tac.PIB_loop(I) ## Uses the mask from the interation setup function

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
