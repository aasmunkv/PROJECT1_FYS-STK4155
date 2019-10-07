import numpy as np
import scipy
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed
from sklearn.model_selection import train_test_split
import sklearn.linear_model as linear_model
import sklearn as skl
import sklearn.metrics as metrics
import pylab
import scipy.stats as stats
import seaborn as sns
from seaborn import heatmap
from imageio import imread
import sys


def CreateDesignMatrix(x, y, n = 5):
	"""
	Function for creating a design X-matrix with rows [1, x, y, x^2, xy, xy^2 , etc.]
	Input is x and y mesh or raveled mesh, keyword agruments n is the degree of the polynomial you want to fit.
	"""
	if len(x.shape) > 1:
		x = np.ravel(x)
		y = np.ravel(y)
	N = len(x)
	l = int((n+1)*(n+2)/2)		# Number of elements in beta
	X = np.ones((N,l))
	for i in range(1,n+1):
		q = int((i)*(i+1)/2)
		for k in range(i+1):
			X[:,q+k] = x**(i-k) * y**k
	return X
def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4


def MSE(z, z_hat):
	#Methods for calculating the mean squared error and the r2-score
	# Test that the input is in vector shape

	if (len(z.shape) > 1):
		z = np.ravel(z)
	if (len(z_hat.shape)) > 1:
		z_hat= np.ravel(z_hat)
	SYY = (z-z_hat)@(z-z_hat)
	return SYY/len(z)



def r2_score(z, z_hat):
	# Method for calculating the r2-score
	# Test that the input is in vector shape
	if (len(z.shape) > 1):
			z = np.ravel(z)
	if (len(z_hat.shape)) > 1:
		z_hat = np.ravel(z_hat)
	n = len(z)
	en = (z-z_hat)@(z-z_hat)
	mean = np.sum(z)/n
	sst = (z - mean)@(z - mean)
	return 1-(en/sst)
#Generate x- and y-values (uniformly randomly distributed between 0 and 1) for testing the model

def createMesh(num_points):
    xp = np.linspace(0, 1, num_points)
    yp = np.linspace(0, 1, num_points)
    x, y = np.meshgrid(xp, yp)
    return x, y



def exploreOLS(num_points, polydegree, verbose = False, noise_degree = 0):
	#Create a meshgrid of x- and y-values linearly spaced between 0 and 1
	#Create the target values, generated from the function we want to approximate.
	#Caltulate the coefficient vector beta by OLS (We need to use np.ravel to get the z-values on vector form.
	#Predicted values:
	n = num_points**2
	x = np.linspace(0, 1, num_points)
	y = np.linspace(0, 1, num_points)
	x, y = np.meshgrid(x, y)
	x, y = createMesh(num_points)
	X = CreateDesignMatrix(x, y, polydegree)
	#TODO: Test if singular matrix, use SVD.
	z = FrankeFunction(x, y)
	#add gaussian noise
	z_noisy = np.ravel(FrankeFunction(x, y)) + noise_degree*np.random.randn(n)
	z_noisy = np.reshape(z_noisy, (num_points, num_points))

	H = scipy.linalg.pinv(X.T@X)
	beta_ols = H@(X.T@np.ravel(z))
	beta_ols_noisy = H@(X.T@np.ravel(z_noisy))
	z_estimated = np.reshape(X@beta_ols, (num_points, num_points))
	z_estimated_noisy = np.reshape(X@beta_ols_noisy, (num_points, num_points))
	#We want to calculate the variance:
	avg = np.mean(z)
	avg_noisy = np.mean(z_noisy)
	variance = np.var(z)
	variance_noisy = np.var(z_noisy)
	V = np.diag(H*variance)
	V_noisy = np.diag(H*variance_noisy)
	residuals = z_estimated_noisy-z_noisy
	if verbose:
	    fig = plt.figure(figsize=(9, 3))
	    ax1 = fig.add_subplot(1,3 , 1, projection = '3d')
	    ax1.set_title("1")
	    ax1.plot_surface(x, y, z_noisy, cmap=cm.coolwarm, antialiased=False)
	    # Plot the approximated surface from the OLS-regression:
	    ax2 = fig.add_subplot(1, 3, 2, projection = '3d')
	    ax2.set_title("2" )
	    ax2.plot_surface(x, y, z, cmap=cm.coolwarm, antialiased=False)
	    fig2 = plt.figure()
	    ax3 = fig.add_subplot(1, 3, 3, projection = '3d')
	    ax3.set_title("3" )
	    ax3.plot_surface(x, y, z_estimated_noisy, cmap=cm.coolwarm, antialiased=False)
	    # Quantile-quantile plot for residuals between estimated and true (noisy) data.
	    stats.probplot(np.ravel(residuals), dist="norm", plot=pylab)

	#Print statistical results:
	    print("Performed OLS-regression with a polynomial of degree %i.\nR2-score:%f" % (polydegree, r2_score(z_estimated,z)))
	    print("Mean squared error:", MSE(z_estimated,z))
	    print("\n\n")
	    print("Performed OLS-regression with a polynomial of degree %i with N(0, 1) noise added.\nR2-score:%f " % (polydegree, r2_score(z_estimated_noisy,z_noisy)))
	    print("Mean squared error witn N(0, 1)-noise:", MSE(z_estimated_noisy,z_noisy))
	    printConfidenceIntervals(beta_ols_noisy, V)
	    print("Mean squared error (estimated values from model trained on noisy data vs values without noise):", MSE(z_estimated_noisy, z))
	    print("r2-score (estimated values from model trained on noisy data vs values without noise):", r2_score(z_estimated_noisy, z))
	return (r2_score(z_estimated_noisy, z_noisy), MSE(z_estimated_noisy, z_noisy))

def exb(num_points, polydegree, verbose = False):
        #K-fold resampling:
        # 1) Make a set of indices, shuffle it randomly and use this as a reference for the x- and y vectors
        # 2) split the data into 10 folds, shuffled by the indices array
        #

        x = np.linspace(0, 1, num_points)
        y = np.linspace(0, 1, num_points)
        x, y = np.meshgrid(x, y)
        x = np.ravel(x)
        y = np.ravel(y)
        ind =  np.array(range(num_points**2))
        np.random.shuffle(ind)
        k = 10
        fold_size = int((num_points**2)/k)
        x_train_fold = np.zeros( (k-1)*fold_size)
        y_train_fold = np.zeros( (k-1)*fold_size)
        x_test_fold = np.zeros(fold_size)
        y_test_fold = np.zeros(fold_size)
        beta = np.empty((k, int((polydegree+1)*(polydegree+2)/2)))
        mse_vec = np.zeros(k)
        r2_vec = np.zeros(k)
        bias_vec = np.zeros(k)
        s_vec = np.zeros(k)
        for i in range(k):

            ind1 = ind[:fold_size*i]
            ind2 = ind[fold_size*(i+1):]
            train_ind = np.array(list(ind1)+list(ind2))
            test_ind = ind[fold_size*i: fold_size*(i+1)]
            for j in range(len(train_ind)):
                x_train_fold[j] = x[train_ind[j]]
                y_train_fold[j] = y[train_ind[j]]
            for l in range(fold_size):
                x_test_fold[l] = x[test_ind[l]]
                y_test_fold[l] = y[test_ind[l]]

            X_train_fold = CreateDesignMatrix(x_train_fold, y_train_fold, polydegree)
            X_test_fold = CreateDesignMatrix(x_test_fold, y_test_fold, polydegree)
            z_train_fold = FrankeFunction(x_train_fold, y_train_fold) + .2*np.random.randn(len(x_train_fold))
            z_test_fold = FrankeFunction(x_test_fold, y_test_fold)+0.2*np.random.randn(len(x_test_fold))
            H_fold = np.linalg.pinv(X_train_fold.T@X_train_fold)
            beta_ols_fold = H_fold@(X_train_fold.T@z_train_fold)
            estimated = X_test_fold@beta_ols_fold
            mse_vec[i] = MSE(estimated, z_test_fold)
            r2_vec[i] = r2_score(estimated, z_test_fold)
            beta[i] = beta_ols_fold
            s_vec[i] = np.mean((estimated - np.mean(estimated))**2)

            bias_vec[i] = np.mean(z_test_fold-np.mean(estimated))

            #Calculate
        beta = sum(beta)/k
        r2 = sum(r2_vec)/k
        mse = sum(mse_vec)/k
        bias = sum(bias_vec)/k
        s = sum(s_vec)/k
        return (r2, mse, s, bias)
        # return (r2_score(z_test_estimated, z_test), MSE(z_test_estimated, z_test))
        # return 0, 0


def k_fold(model, x, y, num, polydegree, lamb, noise_level = 0):
	k = 10 #Nuber of folds
	ind =  np.array(range(num**2))
	np.random.shuffle(ind)
	fold_size = int((num**2)/k)
	x_train_fold = np.zeros( (k-1)*fold_size)
	y_train_fold = np.zeros( (k-1)*fold_size)
	x_test_fold = np.zeros(fold_size)
	y_test_fold = np.zeros(fold_size)
	betas = np.empty((k, int((polydegree+1)*(polydegree+2)/2)))
	mse_vec = np.zeros(k)
	r2_vec = np.zeros(k)
	bias_vec = np.zeros(k)
	s_vec = np.zeros(k)
	for i in range(k):

		ind1 = ind[:fold_size*i]
		ind2 = ind[fold_size*(i+1):]
		train_ind = np.array(list(ind1)+list(ind2))
		test_ind = ind[fold_size*i: fold_size*(i+1)]
		for j in range(len(train_ind)):
			x_train_fold[j] = x[train_ind[j]]
			y_train_fold[j] = y[train_ind[j]]
		for l in range(fold_size):
			x_test_fold[l] = x[test_ind[l]]
			y_test_fold[l] = y[test_ind[l]]
		X_train_fold = CreateDesignMatrix(x_train_fold, y_train_fold, polydegree)
		X_test_fold = CreateDesignMatrix(x_test_fold, y_test_fold, polydegree)

		z_train_fold = FrankeFunction(x_train_fold, y_train_fold) + noise_level*np.random.randn(len(x_train_fold))
		z_test_fold = FrankeFunction(x_test_fold, y_test_fold)+noise_level*np.random.randn(len(x_test_fold))
		beta = model(X_train_fold, z_train_fold, lamb)
		estimated = X_test_fold@beta
		mse_vec[i] = MSE(estimated, z_test_fold)
		r2_vec[i] = r2_score(estimated, z_test_fold)
		betas[i] = beta
		s_vec[i] = np.mean((estimated - np.mean(estimated))**2)
		bias_vec[i] = np.mean(z_test_fold-np.mean(estimated))
	r2 = sum(r2_vec)/k
	mse = sum(mse_vec)/k
	bias = sum(bias_vec)/k
	s = sum(s_vec)/k
	return (r2, mse, s, bias)
def ridge(X, z, l):
	I = np.identity(X.shape[1])
	H = np.linalg.pinv(X.T@X + l*I)
	beta = H@(X.T@z)
	return beta
def ols(X, z):
	H = np.linalg.pinv(X.T@X)
	beta = H@(X.T@z)
	return beta
def lasso(X, z, l):
	clf = linear_model.Lasso(alpha=l, fit_intercept=False)
	clf.fit(X, z)
	beta = clf.coef_
	return beta

def printConfidenceIntervals(beta, variance_matrix):
    percent=0.95
    if len(variance_matrix.shape)>1:
        n = variance_matrix.shape[0]
        for i in range(n):
            print("Estimated %.2f confidence interval for for Beta %i : %f (-/+) %.4f" %(percent,i, beta[i], 1.96*np.sqrt(variance_matrix[i][i]) )   )
        print(" - - - - - - - - - - - - ")
    else:
        n = len(variance_matrix)
        for i in range(n):
            print("Estimated %.2f confidence interval for for Beta %i : %f (-/+) %.4f" %(percent,i, beta[i], 1.96*np.sqrt(variance_matrix[i]) )   )
        print(" - - - - - - - - - - - - ")

def textEx(exfunc, num_points, n):
#This is a method that performs the OLS-regression, calculates the MSE and R2-score
#and plots the results as a function of the polynomial degree.
#This is for data that has not been splitted into test training, and with no noise added.
    x = np.ones(n)
    y = np.ones(n)
    z = np.ones(n)
    for i in range(0,n):
        y[i] = exfunc(num_points, i+1)[0]
        z[i] = exfunc(num_points, i+1)[1]
        x[i] = i
    fig = plt.figure(figsize = (6, 4))
    ax1 = fig.add_subplot(2,1, 1)
    ax1.plot(x, y)
    plt.ylabel("R2-score")
    # ax1.set_title("R2-score(polynomial degree)")
    ax2 = fig.add_subplot(2, 1, 2)
    plt.plot(x, z)
    plt.ylabel("MSE")
    plt.xlabel("Polynomial degree n")

def testExNoisy(exfunc, num_points, n,noise):
#This is a method that performs the OLS-regression, calculates the MSE and R2-score
#and plots the results as a function of the polynomial degree.
#This is for data that has not been splitted into test training, and with no noise added.
    x = np.ones(n)
    y = np.ones(n)
    z = np.ones(n)
    for i in range(0,n):
        y[i] = exfunc(num_points, i+1, False, noise)[0]
        z[i] = exfunc(num_points, i+1, False, noise)[1]
        x[i] = i

    return x, y, z

def testEx2(exfunc, num_points, n):
#This is a method that performs the OLS-regression, calculates the MSE and R2-score
#and plots the results as a function of the polynomial degree.
#This is for data that has not been splitted into test training, and with no noise added.
    y = np.ones(n)
    z = np.ones(n)
    x = np.array(range(n))
    for i in range(0,n):
        y[i] ,z[i] =  exfunc(num_points, i+1)

    return x, y, z
def biasVarianceTradeoff(exfunc, num_points, n, plot=False):
	#This is a method that performs the regression, calculates the MSE and R2-score
	#and plots the results as a function of the polynomial degree.
	#This is for data that has not been splitted into test training, and with no noise added.
	x = np.ones(n)
	r2 = np.ones(n)
	mse = np.ones(n)
	b = np.ones(n)
	v = np.ones(n)
	x = np.array(range(n))
	for i in range(0,n):
		r2[i] ,mse[i] ,  v[i] ,b[i] =  exfunc(num_points, i+1)
		print(mse[i]- (b[i]**2 + v[i]))
	if plot:
		fig = plt.figure(figsize = (10, 10))
		ax1 = fig.add_subplot(2, 2, 1)
		ax1.set_title("R2-score(polynomial degree)")
		ax1.plot(x, r2)
		ax2 = fig.add_subplot(2, 2, 2)
		ax2.set_title("MSE(polynomial degree)")
		ax2.plot(x, mse)
		ax3 = fig.add_subplot(2, 2, 3)
		ax3.set_title("Bias(polynomial degree)")
		ax3.plot(x, b)
		ax4 = fig.add_subplot(2, 2, 4)
		ax4.plot(x, v)
		ax4.set_title("Variance(polynomial degree)")
		ax4.plot(x, v)
	return r2, mse


def train_test(model, noise_degree, polydegree, num_points, ridgeparam=0):
	x = np.linspace(0, 1, num_points)
	y = np.linspace(0, 1, num_points)
	x, y = np.meshgrid(x, y)
	x = np.ravel(x)
	y = np.ravel(y)
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.36)
	X_train = CreateDesignMatrix(x_train, y_train, polydegree)
	X_test = CreateDesignMatrix(x_test, y_test, polydegree)
	z_train = FrankeFunction(x_train, y_train) + noise_degree*np.random.randn(len(x_train))
	z_test = FrankeFunction(x_test, y_test) + noise_degree*np.random.randn(len(x_test))
	beta = model(X_train, z_train, ridgeparam)
	z_pred = X_test@beta
	return(MSE(z_test, z_pred), r2_score(z_test, z_pred))

def exc(ridge, num_points, noise_degree, ridgeparam):
	k = 20
	arr = np.arange(k)
	mse = np.zeros(k)
	r = np.zeros(k)
	for i in range(0, k):
	    mse[i], r[i] = train_test(ols, noise_degree, i, num_points, ridgeparam)
	fig = plt.figure(figsize = (6, 4))
	ax1 = fig.add_subplot(2,1, 1)
	ax1.set_title("MSE & r2-score as functions of complexity. Variance = %.2f" %(noise_degree**2))
	ax1.plot(arr, mse)
	plt.ylabel("MSE")
	# ax1.set_title("R2-score(polynomial degree)")
	ax2 = fig.add_subplot(2, 1, 2)
	plt.plot(arr, r)
	plt.ylabel("r2-score")
	plt.xlabel("Polynomial degree n")

# Code to generate heatmap of hyperparameter tuning.
def tuneParameters(model, noise_level, n = 4, num_points = 100):
	fig = plt.figure()
	#This is a method to explore the relationship between the ridge parameter lambda,
	# the model complexity and the MSE.
	x = np.linspace(0, 1, num_points)
	y = np.linspace(0, 1, num_points)
	x, y = np.meshgrid(x, y)
	x = np.ravel(x)
	y = np.ravel(y)
	M = np.empty((n, n))
	T = np.zeros(n)
	ind = np.linspace(0, 2*(n/100), n)
	for i in range(len(ind)):
		ind[i] = (2*i)/100
	for r in range(1, n+1):
		for i in range(1, n+1):
			T[i-1] = k_fold( model,x, y, num_points, i, 2*((r-1)/100), noise_level)[1]
			# T[i-1] = k_fold(ridge,x, y, 50, i, 10**(10-n+i), noise_level)[1]
		M[r-1]=T
	arr = np.array(np.where(M == np.min(M)	)).flatten()
	lamb, comp = indToLambdaN(n, arr)
	print("Optimal complexity: ", comp, " lambda: ", lamb, ". MSE = ", np.min(M))
	# ax = sns.heatmap(M)
	plt.imshow(M, origin='lower', extent = [0, n, 0, n])
	ax = plt.gca()
	ax.set(yticks= (np.arange(1, n+1)-0.5))
	ax.set(yticklabels = ind)
	ax.set(xticks= (np.arange(1, n+1)-0.5))
	ax.set(xticklabels = np.arange(1, n+1))
	plt.xlabel("Polynomial degree")
	plt.ylabel("Ridge parameter lambda")
	plt.title("MSE as a function of lambda and complexity")
	plt.colorbar()

def tuneParametersLasso(model, noise_level,  n = 4,  num_points = 100 ):
	fig = plt.figure()
	#This is a method to explore the relationship between the ridge parameter lambda,
	# the model complexity and the MSE.
	x = np.linspace(0, 1, num_points)
	y = np.linspace(0, 1, num_points)
	x, y = np.meshgrid(x, y)
	x = np.ravel(x)
	y = np.ravel(y)
	M = np.empty((n, n))
	T = np.zeros(n)
	ind = np.linspace(0, 2*(n/100), n)
	for i in range(len(ind)):
		ind[i] = (5*i)/10000
	for r in range(1, n+1):
		for i in range(1, n+1):
			T[i-1] = k_fold(model,x, y, num_points, i, (5*r/10000), noise_level)[1]
		M[r-1]=T

	plt.imshow(M, origin='lower', extent = [0, n, 0, n])
	ax = plt.gca()
	ax.set(yticks= (np.arange(1, n+1)-0.5))
	ax.set(yticklabels = ind)
	ax.set(xticks= (np.arange(1, n+1)-0.5))
	ax.set(xticklabels = np.arange(1, n+1))
	plt.xlabel("Polynomial degree")
	plt.ylabel("Lasso parameter lambda")
	plt.title("MSE as a function of lambda and complexity")
	plt.colorbar()
def indToLambdaN(n, ind):
	lamb = (ind[0]*2)/100
	n = ind[1]
	return lamb, n

def initialize(num_points):
	x = np.linspace(0, 1, num_points)
	y = np.linspace(0, 1, num_points)
	x, y = np.meshgrid(x, y)
	x = np.ravel(x)
	y = np.ravel(y)
	return x, y

def exd(model, lamb, num_points):
	#Initialization
	x, y = initialize(num_points)
	z = FrankeFunction(x, y)
	M = np.zeros(6)
	R = np.zeros(6)
	for i in range(1, 6):
		X = CreateDesignMatrix(x, y, i)
		beta = ridge(X, z, lamb)
		pred = X@beta
		mse = MSE(z, pred)
		r2 = r2_score(z, pred)
		M[i-1] = mse
		R[i-1] = r2
		print("Degree: %i MSE: %.3f r2-score: %.3f" %(i, mse, r2))
	X = CreateDesignMatrix(x, y, 50)
	beta = model(X, y, lamb)
	pred = X@beta
	mse = MSE(z, pred)
	r2 = r2_score(z, pred)
	print("Degree: %i MSE: %.3f r2-score: %.3f" %(50, mse, r2))

def exd2(model, noise_degree, lamb, num_points):
	M = np.zeros(20)
	R = np.zeros(20)
	x = np.linspace(0, 1, num_points)
	y = np.linspace(0, 1, num_points)
	x, y = np.meshgrid(x, y)
	x = np.ravel(x)
	y = np.ravel(y)

	for i in range(1, 21):
		ret = k_fold(model, x, y, num_points, i, lamb, noise_degree)
		M[i-1] = ret[1]
		R[i-1] = ret[0]
	ind = np.arange(1, 21)
	fig = plt.figure(figsize = (6, 4))
	ax1 = fig.add_subplot(2,1, 1)
	ax1.set_title("MSE & r2-score as functions of complexity. K-fold. variance = %.2f." %(noise_degree**2))
	ax1.plot(ind, M)
	plt.ylabel("MSE-score")
	# ax1.set_title("R2-score(polynomial degree)")
	ax2 = fig.add_subplot(2, 1, 2)
	plt.plot(ind, R)
	plt.ylabel("r2")
	plt.xlabel("Polynomial degree n")


def surface_plot(surface,title, surface1=None):
    M,N = surface.shape
    ax_rows = np.arange(M)
    ax_cols = np.arange(N)
    [X,Y] = np.meshgrid(ax_cols, ax_rows)
    fig = plt.figure()
    if surface1 is not None:
        ax = fig.add_subplot(1,2,1,projection='3d')
        ax.plot_surface(X,Y,surface,cmap=cm.coolwarm,linewidth=0)
        plt.title(title)
        ax = fig.add_subplot(1,2,2,projection='3d')
        ax.plot_surface(X,Y,surface1,cmap=cm.coolwarm,linewidth=0)
        plt.title(title)
    else:
        ax = fig.gca(projection='3d')
        ax.plot_surface(X,Y,surface,cmap=cm.viridis,linewidth=0)
        plt.title(title)
def surface_plot_1(x, y, z, title):
	fig = plt.figure()
	ax = fig.add_subplot(projection='3d')
	ax.plot_surface(x,y,z, cmap=cm.coolwarm,linewidth=0)
	plt.title(title)


def k_fold_terrain(model, x, y, data,  polydegree, lamb, folds,  noise_level = 0):
	k = folds #Nuber of folds
	ind =  np.arange(len(data))
	np.random.shuffle(ind)
	fold_size = int((len(x))/k)
	x_train_fold = np.zeros( (k-1)*fold_size)
	y_train_fold = np.zeros( (k-1)*fold_size)
	z_train_fold = np.zeros((k-1)*fold_size)
	x_test_fold = np.zeros(fold_size)
	y_test_fold = np.zeros(fold_size)
	z_test_fold = np.zeros(fold_size)
	betas = np.empty((k, int((polydegree+1)*(polydegree+2)/2)))
	mse_vec = np.zeros(k)
	r2_vec = np.zeros(k)
	bias_vec = np.zeros(k)
	s_vec = np.zeros(k)
	for i in range(k):
		ind1 = ind[:fold_size*i]
		ind2 = ind[fold_size*(i+1):]
		train_ind = np.array(list(ind1)+list(ind2))
		test_ind = ind[fold_size*i: fold_size*(i+1)]
		for j in range(len(train_ind)-1):
			x_train_fold[j] = x[train_ind[j]]
			y_train_fold[j] = y[train_ind[j]]
			z_train_fold[j] = data[train_ind[j]]
		for l in range(fold_size):
			x_test_fold[l] = x[test_ind[l]]
			y_test_fold[l] = y[test_ind[l]]
			z_test_fold[l] = data[test_ind[l]]
		X_train_fold = CreateDesignMatrix(x_train_fold, y_train_fold, polydegree)
		X_test_fold = CreateDesignMatrix(x_test_fold, y_test_fold, polydegree)
		beta = model(X_train_fold, z_train_fold, lamb)
		estimated = X_test_fold@beta
		mse_vec[i] = metrics.mean_squared_error(estimated, z_test_fold)
		r2_vec[i] = r2_score(estimated, z_test_fold)
		betas[i] = beta
		s_vec[i] = np.var(estimated)
		bias_vec[i] = np.mean((z_test_fold-np.mean(estimated) )**2,  keepdims=True)
	r2 = sum(r2_vec)/k
	mse = sum(mse_vec)/k
	bias = sum(bias_vec)/k
	s = sum(s_vec)/k
	return (r2, mse, s, bias)

def terrain_function(model, terrain, polynomial_degree, lamb, folds):
	x = np.linspace(0, 1.8, 1801)
	y = np.linspace(0, 3.6, 3601)
	x, y = np.meshgrid(x, y)
	x = np.ravel(x)
	y = np.ravel(y)
	ret = k_fold_terrain(model, x, y, np.ravel(terrain), polynomial_degree, lamb, folds)
	print("Polynomial degree: %i. Lambda: %f \nMSE: %f \nR2-score: %f" % (polynomial_degree, lamb, ret[1], ret[0]))
	# f.write("%i & %6.3f & %f & %f\\\ \n" % (polynomial_degree, lamb,  ret[1], ret[0]))


def terrain_ols(model, terrain, polynomial_degree, lamb):
	# x = np.linspace(0, 1.8, 1801)
	# y = np.linspace(0, 3.6, 3601)
	dim_x = terrain.shape[1]
	dim_y = terrain.shape[0]
	x = np.linspace(0, (dim_x-1)/1000, dim_x)
	y = np.linspace(0, (dim_y-1/1000), dim_y)
	print(x, y)
	x, y = np.meshgrid(x, y)
	x = np.ravel(x)
	y = np.ravel(y)

	terrain = np.ravel(terrain)
	X = CreateDesignMatrix(x, y, polynomial_degree)

	beta = model(X, terrain, lamb)
	print("Betas:" , beta)
	return beta, X




# x = np.arange(terrain.shape[1])
# y = np.arange(terrain.shape[0])
# x, y = np.meshgrid(x, y)

# surface_plot_1(x, y, np.reshape(predicted, (3601,1801)), "Parameterized surface" )
# # data = terrain_function(terrain, 2)
# f = open("Kjoring.txt", "w")
# for i in range(0, 7):
# 	terrain_function(ridge, terrain, 9, 0, 5)

# for polydegree in range(1, 7):
# 	for lamb in range(1, 7):
# 		terrain_function(ridge, terrain, polydegree, lamb/100, 5)
# for polydegree in range(1, 6):
# 	for lamb in range(1, 5):
# 		terrain_function(lasso, terrain, polydegree, (5*lamb)/10000, 5)
# f.close()
# x = np.arange(1801)
# y = np.arange(3601)
# x, y = np.meshgrid(x, y)
# x = np.ravel(x)
# y = np.ravel(y)
# X = CreateDesignMatrix(x, y, 5)
# reg = skl.linear_model.LinearRegression(fit_intercept = True, normalize = True)
# reg.fit(X, np.ravel(terrain))
# beta = reg.coef_
# data = X@beta
# print(beta)
# y = np.arange(3601)
# x = np.arange(1801)
# x, y = np.meshgrid(x, y)
#
# data = np.reshape(data, (3601, 1801))
#
#
# plt.imshow(data)


def normalized_terrain(model, data, polydegree, lamb):
	plt.figure()
	xd = data.shape[1]
	yd = data.shape[0]
	scalex = xd/max(xd,yd)
	scaley = yd/max(xd,yd)
	x_ind = np.linspace(0, scalex, xd)
	y_ind = np.linspace(0, scaley, yd)
	x, y = np.meshgrid(x_ind, y_ind)
	X = CreateDesignMatrix(np.ravel(x), np.ravel(y), polydegree)
	beta = model(X, np.ravel(data), lamb)
	print(beta)
	dat = X@beta
	data = np.reshape(dat, (yd, xd))
	plt.imshow(data, cmap = 'coolwarm')
	plt.colorbar()
	plt.title("Parameterized terrain")
	plt.figure()
	plt.imshow(data)
	plt.show()
	return data

'''CODE THAT PRODUCES FIGURE 3'''
# num_points = 100
# polydeg = 3
# x, y = createMesh(num_points)
# frankeData = FrankeFunction(x, y)
# X = CreateDesignMatrix(np.ravel(x), np.ravel(y), polydeg)
# betas = ols(X, np.ravel(frankeData))
# estimated = X@betas
# estmated = np.reshape(estimated, (num_points, num_points))
# surface_plot_1( np.reshape(x, (num_points, num_points)), np.reshape(y, (num_points, num_points)), np.reshape(estimated, (num_points, num_points)),"Polynomial degree =%i"%polydeg)
#
# polydeg = 5
# x, y = createMesh(num_points)
# frankeData = FrankeFunction(x, y)
# X = CreateDesignMatrix(np.ravel(x), np.ravel(y), polydeg)
# betas = ols(X, np.ravel(frankeData))
# estimated = X@betas
# estmated = np.reshape(estimated, (num_points, num_points))
# surface_plot_1( np.reshape(x, (num_points, num_points)), np.reshape(y, (num_points, num_points)), np.reshape(estimated, (num_points, num_points)),"Polynomial degree = %i"%polydeg)
#
# polydeg = 20
# x, y = createMesh(num_points)
# frankeData = FrankeFunction(x, y)
# X = CreateDesignMatrix(np.ravel(x), np.ravel(y), 3)
# betas = ols(X, np.ravel(frankeData))
# estimated = X@betas
# estmated = np.reshape(estimated, (num_points, num_points))
# surface_plot_1( np.reshape(x, (num_points, num_points)), np.reshape(y, (num_points, num_points)), np.reshape(estimated, (num_points, num_points)),"Polynomial degree = 3")
#END OF CODE THAT PRODUCES FIGURE 3

'''Code that procuces figure 4'''
#textEx(exploreOLS, 100, 15)
#END OF CODE THAT PRODUCES FIGURE 4






'''Code that produces figure 6'''
# k = 30
# mse = np.zeros(k)
# ind = np.arange(k)
# fig5 = plt.figure()
# for i in range(0, k):
#     mse[i] = exploreOLS(100, i, False, 0.3)[1]
# plt.plot(ind, mse)
# for i in range(0, k):
# mse[i] = exploreOLS(100, i, False, 0.6)[1]
# plt.plot(ind, mse)
#
# for i in range(0, k):
#     mse[i] = exploreOLS(100, i, False, 0.9)[1]
# plt.plot(ind, mse)
#
# plt.legend(["lambda = 0.3", "lambda = 0.6", "lambda = 0.9"])
# plt.title("MSE as a function of complexity")
#END OF CODE THAT GENERATES FIGURE 6

'''CODE THAT GENERATES FIGURE 7 '''
# Figure 7a
# exc(ridge, 20, 0, 0)
# Figure 7b
# exc(ridge, 20, 0.5, 0)
#CODE THAT GENERATES FIGURE 7

'''CODE THAT GENERATES FIGURE 8'''
# Figure 8a
# exc(ridge, 100, 0, 0)
# Figure 8b
# exc(ridge, 100, 0.5, 0)
# END OF CODE THAT GENERATES FIGURE 8

'''CODE THAT GENERATES FIGURE 9'''
# 9a
# exd2(ridge, 0.5,0, 100)
# 9b
# exd2(ridge, 0.5,0, 100)
# END OF CODE THAT GENERATES FIGURE 9

'''CODE THAT GENERATES FIGURE 10'''
# tuneParameters(ridge, 0, 15, 100)
# END OF CODE THAT GENERATES FIGURE 10


'''CODE THAT GENERATES FIGURE 11'''
#11a
# tuneParameters(ridge, 0.5, 15, 100)
# 11b
# tuneParameters(ridge, 0.5, 15, 20)
# END OF CODE THAT GENERATES FIGURE 11

'''CODE THAT GENERATES FIGURE 12'''
# 12a
# exd2(ridge, 0.5, 0.04, 10)
# 12b
# exd2(ridge, 0.5, 0.04, 100)

# END OF ODE THAT GENERATES FIGURE 12


'''CODE THAT GENERATES FIGURE 13'''
# 13a
# tuneParametersLasso(lasso, 0.5,15, 20)
# 13b
# tuneParametersLasso(lasso, 0.5, 15, 100)
# END OF CODE THAT GENERATES FIGURE 13


terrain = imread('SRTM_data_Norway_1.tif')
plt.figure()
plt.title('Terrain over Norway 1')
plt.imshow(terrain, cmap='coolwarm')
plt.xlabel('X')
plt.ylabel('Y')
plt.colorbar()


plt.show()
