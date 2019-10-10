
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
def DesignMatrix(x, y, p):

    '''
    Takes in predictor varaiables and max order for each variable for 2d 
    function case and constructs design matrix. 
    Assumes x, y are one dimensional N**2 vectors.
    '''
    DM = np.zeros_like(x)
    for i in range(p+1):
        for j in range(p+1):
            DM = np.c_[DM, (x**i)*(y**j)]
    return DM[:, 1:]

def kfold_split(N, n_folds=3):
    '''
    Returns arrays of indices for kfold splitting.
    '''
    index = np.arange(N)
    np.random.shuffle(index)
    k_test = np.array_split(index, n_folds)
    k_train = []
    for fold in range(n_folds):
        k_train.append(np.concatenate(np.delete(k_test, fold, axis=0)))
    return k_train, k_test

def train_test_split(*args, split_frac = 0.75):
    '''
    Returns [X_train, z_train, X_test, z_test]
    '''
    n = args[0].shape[0]
    index = np.arange(n)
    np.random.shuffle(index)
    k_split = round(n*split_frac)
    train, test = np.empty((len(args), k_split, 1)), np.empty((len(args), n-k_split, 1))
    
    for i in range(len(args)):
        train[i,:, :] = args[i][index][:k_split].reshape(int(n*split_frac), 1)
        test[i,:, :] = args[i][index][k_split:].reshape(int(n) - int(n*split_frac), 1)
    return train, test
    
    
    
    return train, test   
    

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

def MSE(z, z_pred):
    return np.mean((z - z_pred)**2)

def R2(z, z_pred):
    S1 = np.sum((z - z_pred)**2)
    S2 = np.sum((z - np.mean(z))**2)
    return 1 - S1/S2



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
    lasso_reg = Lasso(alpha=lmbda, fit_intercept=False)
    lasso_reg.fit(X, z)
    lasso_fit = lasso_reg.predict(X)
    lasso_beta = lasso_reg.coef_ 
    
    return lasso_beta, lasso_fit
  

class Polyfit:
    
    def __init__(self):
        pass
        
    def fit(self,X, z,  model, lm=0):
        
        self.X, self.z = X, z
        self.p = X.shape[1]
        
        if model == 'OLS':
            self.reg =  OLS(X, z)
        elif model == 'ridge':
            self.reg = ridge(X, z, lmbda = lm)
        elif model == 'lasso':
            self.reg = lasso(X, z, lmbda = lm)
        else:
            return
        
        return self.reg

    @jit
    def predict(self, X):
        return np.matmul(X, self.reg[0]) 
    

                    
            
        
        
        
        
        
        
        
        
        
        
        
        
