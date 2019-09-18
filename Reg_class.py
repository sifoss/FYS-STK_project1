
import numpy as np
from numba import jit
from sklearn.linear_model import Lasso

def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

@jit
def DesignMatrix(x, y, px, py):

    '''
    Takes in predictor varaiables and max order for each variable for 2d 
    function case and constructs design matrix. 
    Assumes x, y are one dimensional N**2 vectors.
    '''
    DM = np.zeros_like(x)
    for i in range(px+1):
        for j in range(py+1):
            DM = np.c_[DM, (x**i)*(y**j)]
    return DM[:, 1:]

def kfold_split(X, z, k):
    
    n = z.shape[0]
    index = np.arange(n)
    np.random.shuffle(index)
    z_split = np.array_split(z[index], k)
    X_split = np.array_split(X[index], k)
    return X_split, z_split

def train_test_split(X, z, split_frac = 0.75):
    n = z.shape[0]
    index = np.arange(n)
    np.random.shuffle(index)
    z_shuff = z[index]
    X_shuff = X[index]
    k_split = round(z.shape[0]*split_frac)
    return X_shuff[:k_split], z_shuff[:k_split], X_shuff[k_split:], z_shuff[k_split:]    
    

def plotter(x, y, z, save= False):
    '''
    plots the surface z = f(x, y), and saves the figure if save != False
    '''
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    # Customize the z axis.
    ax.set_zlim(np.min(z) - 0.2, np.max(z) + 0.2)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    if save != False:
        plt.savefig(str(save) + '.pdf', format='pdf')
    plt.show()
'''
def MSE(z, z_pred):
    return np.sum((z - z_pred)**2)/z.shape[0]
'''

def bias(z, z_pred):
    return np.sum(z_pred - z)/z.shape[0]

def variance(z):
    z_mean = np.sum(z)/np.sum(z.shape)
    return np.sum((z - z_mean)**2)/z.shape[0]  

@jit    
def OLS(X, z):
    '''
    Performs an OLS  2d polynomial fit of order (px, py) in p(x, y) 
    using the SVD decomposition
    and stores the regression coefficients and fitted function.
    '''
    p = X.shape[1]
    U, Sigma, VT = np.linalg.svd(X)
    UT = U[:, :p].T
    C = np.zeros_like(UT)
    S_inv = 1/Sigma
    for i in range(UT.shape[0]):
        C[i, :] = S_inv[i]*UT[i, :]
        
    OLS_beta = VT.T.dot(C).dot(z)
    OLS_fit = X.dot(OLS_beta)
    
    return OLS_beta, OLS_fit

@jit    
def ridge(X, z, lmbda=0):
    
    n, p = X.shape
    
    C = X.T.dot(X) + lmbda*np.eye(p)
    ridge_beta = np.linalg.inv(C).dot(X.T).dot(z)
    ridge_fit = X.dot(ridge_beta)
    
    return ridge_beta, ridge_fit

def lasso(X, z, lmbda=0):
    lasso_reg = Lasso(alpha=lmbda)
    lasso_reg.fit(X, z)
    lasso_fit = lasso_reg.predict(X)
    lasso_beta = lasso_reg.coef_
    
    return lasso_beta, lasso_fit
  

class Polyfit:
    
    def __init__(self):
        pass
        
    def fit(self,X, z,  model, lm=0):
        
        self.X, self.z = X, z
        
        if model == 'OLS':
            self.reg =  OLS(X, z)
        elif model == 'ridge':
            self.reg = ridge(X, z, lmbda = lm)
        elif model == 'lasso':
            self.reg = lasso(X, z, lmbda = lm)
        else:
            return
        
        return self.reg
     
    def beta_variance(self):
        X, z = self.X, self.z
        sigma2 = variance(z)
        '''
        Sigma, VT = np.linalg.svd(X)[1:]
        D2 = 1/(Sigma*Sigma)
        C = np.zeros_like(VT)
        for i in range(VT.shape[0]):
            C[i, :] = D2[i]*VT[i, :]
            
        return sigma2*np.sqrt(np.diag(np.matmul(VT.T, C)))
        '''
        inv = np.diag(np.linalg.inv(np.matmul(X.T, X)))
        return sigma2*np.sqrt(inv)
    
    def MSE(self):
        z = self.z
        z_pred = self.reg[1]
        return np.sum((z - z_pred)**2)/z.shape[0]
    
    def R2(self):
        z = self.z
        z_pred = self.reg[1]
        S1 = np.sum((z - z_pred)**2)
        z_mean = np.sum(z)/z.shape[0]
        S2 = np.sum((z - z_mean)**2)
        return 1 - S1/S2
    
    
        
                    
            
        
        
        
        
        
        
        
        
        
        
        
        
