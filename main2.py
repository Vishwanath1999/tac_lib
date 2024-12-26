# %% Code to simulate the TAC with longitudinal phase noise and turbulence
from tac_lib.common_imports import *
from tac_lib.utilities import Utilities
from tac_lib.tac import TiledApertureBeamProp
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

    L_0 = 10
    l_0 = 10e-3
    N_ss = 500
    w_l = 1064e-9
    c_n2 = 1e-13
    k = 2*np.pi/w_l
    n_screens = utilities.num_of_screen(k, c_n2, L, L)

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
    Tac = TiledApertureBeamProp(im_size,pix_size,n_channel,n_screens,Kvar,Z,trans_pn,atm,amp_v,g_amp,a,d,) ## TODO: saving the kernel here can save a lot of time 
    U_in, U_out = Tac.TiledAperture_2(np.zeros(n_channel)) 

    I = np.abs(U_out)**2

    pib_n = np.ceil(Tac.PIB(I,im_size/2, im_size/2,rp,pix_size)) ## Recommended for single execution 
    roiMask = Tac.CircMask(I.shape,im_size/2,im_size/2,rp) ## For many iterations , define the mask 
    pib_n = Tac.PIB_loop(I,roiMask,pix_size) ## and loop only this 
    # print('Masked PIB ideal: ', masked_pib_n)

    f_loop = fs#5e4
    f_ctrl = round(fs/f_loop)
    print('Control rate: ', f_ctrl)
# %%
# plt.imshow(I,cmap='jet')
# plt.show()
# create subplot of two images of U_in and I with jet colormap
fig, ax = plt.subplots(1, 2)
xp = 100
ax[0].imshow(np.abs(U_in[im_size//2-xp:im_size//2+xp,im_size//2-xp:im_size//2+xp])**2, cmap='jet')
ax[0].set_title('I_in')
ax[1].imshow(I, cmap='jet')
ax[1].set_title('I_out')
plt.show()
# %%
