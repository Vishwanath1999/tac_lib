from .common_imports import *
class Optimizer:
    def __init__(self,algotype='adam',gain=0.1,f_ctrl=1,upperV=9.5,lowerV=0.01,resetV=5):
       self.gain = gain
       self.beta1 = 0.9
       self.beta2 = 0.999
       self.eps = 1e-8
       self.m = np.zeros((self.n_channel,))
       self.v = np.zeros((self.n_channel,))
       self.algotype = algotype
       self.del_V = 1*np.random.rand(self.n_channel,) ## this has to be reset everytime
       self.f_ctrl = f_ctrl
    def reset(self):
        self.m = np.zeros((self.n_channel,))
        self.v = np.zeros((self.n_channel,))
        self.del_V = 1*np.random.rand(self.n_channel,)
        self.V = np.zeros(self.n_channel)
    
    def adam(self,I_plus,I_minus,ii):
        grad = I_plus - I_minus
        self.m = self.beta1*self.m + (1-self.beta1)*grad
        self.v = self.beta2*self.v + (1-self.beta2)*(grad**2)
        m_hat = self.m/(1-self.beta1**(ii+1)) # bias correction
        v_hat = self.v/(1-self.beta2**(ii+1)) # bias correction
        self.V += self.gain*m_hat/(np.sqrt(v_hat)+self.eps)
    def spgd(self,I_plus,I_minus,ii):
        grad = I_plus - I_minus
        self.V += self.gain*grad*self.del_V
def optimizer(self,I_plus,I_minus,ii,resetV=5,upperV=9.5,lowerV=0.01):
        """Optimizer Module 

        Args:
            I_plus (ndarray): Positive Perturbation 
            I_minus (ndarray): Negative Perturbation
            ii (int): _description_
            resetV (int, optional): _description_. Defaults to 5.
            upperV (float, optional): _description_. Defaults to 9.5.
            lowerV (float, optional): _description_. Defaults to 0.01.

        Returns:
            np array: returns the updated control vector. 
        """
        V = np.zeros(self.n_channel)
        grad = I_plus - I_minus 
        if self.runtype=='cl' and ii%self.f_ctrl==0:
            del_I = np.real(I_plus) - np.real(I_minus)
            grad = del_I* self.del_V
            if self.algotype == 'adam':
                m = self.beta1*m + (1-self.beta1)*grad
                v = self.beta2*v + (1-self.beta2)*(grad**2)
                m_hat = m/(1-self.beta1**(ii+1)) # bias correction
                v_hat = v/(1-self.beta2**(ii+1)) # bias correction
                V += self.gain*m_hat/(np.sqrt(v_hat)+self.eps)

            elif self.algotype == 'spgd':
                V += self.gain*del_I*self.del_V
            
            #TODO: implement the other algorithms 
            ## Voltage Correction for the controllers
            for k in V:
                if k>9.5 or k>0.01:
                    k=5
            return V
