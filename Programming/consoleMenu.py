import numpy as np
from random import random, seed
from sklearn.model_selection import train_test_split
import sklearn.linear_model as linear_model     # LinearRegression, Ridge, Lasso

import matplotlib.pyplot as plt
from imageio import imread
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def FrankeFunction(x, y):
    """Returns Franke's function"""
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

def CreateDesignMatrix_X(x_, y_, n):
    """
	Function for creating a design X-matrix with rows [1, x, y, x^2, xy, xy^2 , etc.]
	Input is x and y mesh or raveled mesh, keyword agrument n is the degree of the polynomial you want to fit.
	"""
    if len(x_.shape) > 1:
        x_ = np.ravel(x_)
        y_ = np.ravel(y_)
    N = len(x_)
    l = int(((n+1)*(n+2))/2)  # Number of elements in beta
    X = np.ones((N,l))
    for i in range(1, n+1):
        q = int((i)*(i+1)/2)
        for j in range(i+1):
            X[:,q+j] = x_**(i-j) * y_**j
    return X

def olsPrint(x, y, method=FrankeFunction):
    """
    Calculates and prints out the variance, bias, error and R2-score,
    after asking the user for the degree of polynomial and noise.
    """
    # Get degree of noise:
    noise_deg = getNumberInput("  Degree of noise (from 0 to 1): ", True)
    while True:
        degree = polynomialDegreeMenu()
        if degree in ["Q","q"]:
            break
        elif degree in ["B","b"]:
            olsMenu(x, y)
            break
        # Run calculations:
        z, z2, x_new, y_new, beta, H = ols(x, y, noise_deg, degree, method)
        variance, bias, error, R2 = calculation(z, z2)
        # Run prints:
        printCalculation(variance, bias, error, R2)
        printBeta(beta, np.var(z-z2)*H)

def ols(x_, y_, noise_deg, degree, method=FrankeFunction, lamb=0, compare=False):
    """
    Regular Ordinary Least Square algorithm.
    Returns:
    - z: True values of Frankefunction with added noise of choice.
    - z_pred: Predicted z-values after running regular OLS.
    - x_, y_: Meshed version of x and y.
    - beta: Estimated beta-values from OLS.
    - H: The transposed of design matrix dotted with it self (with added ridge).
    """
    if compare:
        noise = noise_deg*np.random.randn(len(x_))
    else:
        noise = noise_deg*np.random.randn(len(x_)*len(y_))
        x_, y_ = np.meshgrid(x_,y_)

    X = CreateDesignMatrix_X(x_, y_, degree)
    z = np.ravel(method(x_, y_)) + noise
    H_tmp = (X.T) @ X
    H = np.linalg.pinv(H_tmp + lamb*np.identity(len(H_tmp)))
    beta = H @ (X.T @ z)
    z_pred = X @ beta
    return z, z_pred, x_, y_, beta, H

def trainTestSplit(x, y, method, method2=FrankeFunction, lamb=0, bool_lasso=False):
    """Trains the model with OLS and runs calculations with test-data and obtained value of beta."""
    # Get test-data-size:
    size = getNumberInput("  Size of test-data-set (from 0 to 1): ", "both")
    # Get noise degree:
    noise_deg = getNumberInput("  Degree of noise (from 0 to 1): ", True)
    # Split data into train and test:
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=size, random_state=42)
    while True:
        degree = polynomialDegreeMenu()
        if degree in ["Q","q"]:
            break
        elif degree in ["B","b"]:
            if method=="ols": olsMenu(x, y)
            elif method=="ridge": ridgeMenu(x, y, lamb)
            else: mainMenu(x, y)
            break
        # Run OLS to get beta:
        beta = ols(x_train, y_train, noise_deg, degree, lamb=lamb)[-2]

        #OBS: husk at det må legges til støy på alle y-verdiene. Dvs. både på test- og på treningssettet.
        # Test model:
        xtest, ytest = np.meshgrid(x_test, y_test)
        X_test = CreateDesignMatrix_X(xtest, ytest, degree)
        z_test = method2(xtest, ytest)  + noise_deg*np.random.randn(len(xtest))
        z2_test = X_test @ beta
        # Run calculations:
        variance, bias, error, R2 = calculation(z_test, z2_test)
        # Print results:
        printCalculation(variance, bias, error, R2)

def olsRidgeCV(x, y, method, method2=FrankeFunction, lamb=0):
    """k-fold cross validation algorithm for OLS and Ridge."""
    # Get k:
    k = getNumberInput("  Number of folds: ", "greater1", False)
    # Get noise degree:
    noise_deg = getNumberInput("  Degree of noise (from 0 to 1): ", True)
    while True:
        # Get polynomial degree:
        degree = polynomialDegreeMenu()
        if degree in ["Q","q"]:
            break
        elif degree in ["B","b"]:
            if method=="ols": olsMenu(x, y)
            elif method=="ridge": ridgeMenu(x, y, lamb)
            else: mainMenu(x, y)
            break
        # Initialize arrays and values:
        mesh_size, fold_size, noise_train, noise_test, x_rav, y_rav, ind = CVInitialize(x, y, k, noise_deg)
        x_train, y_train = np.zeros(mesh_size - fold_size), np.zeros(mesh_size - fold_size)
        x_test, y_test = np.zeros(fold_size), np.zeros(fold_size)
        MSE_test_arr, R2_test_arr, var_test_arr, bias_test_arr = np.zeros(k), np.zeros(k), np.zeros(k), np.zeros(k)
        beta_arr = np.zeros([k, int((degree+1)*(degree+2)/2)])
        # Start k-fold algorithm:
        for i in range(k):
            # Set up train-test-data:
            train_ind = np.array(list(ind[:i*fold_size])+list(ind[(i+1)*fold_size:]))
            test_ind = ind[i*fold_size:(i+1)*fold_size]
            for j in range(mesh_size - fold_size):
                x_train[j] = x_rav[train_ind[j]]
                y_train[j] = y_rav[train_ind[j]]
            for l in range(fold_size):
                x_test[l] = x_rav[test_ind[l]]
                y_test[l] = y_rav[test_ind[l]]
            # Train model:
            X_train = CreateDesignMatrix_X(x_train, y_train, degree)
            z_train = method2(x_train, y_train) + noise_train
            H_tmp = X_train.T @ X_train
            H_train = np.linalg.pinv(H_tmp + lamb*np.identity(len(H_tmp)))
            beta = (H_train @ X_train.T) @ z_train
            # Test model:
            X_test = CreateDesignMatrix_X(x_test, y_test, degree)
            z_test = method2(x_test, y_test) + noise_test
            z2_test = X_test @ beta
            # Run calculations:
            var_test_arr[i],bias_test_arr[i],MSE_test_arr[i],R2_test_arr[i] = calculation(z_test, z2_test)
            beta_arr[i] = beta
        # Calculate the mean of arrays:
        MSE_test = np.sum(MSE_test_arr)/k
        R2_test = np.sum(R2_test_arr)/k
        var_test = np.sum(var_test_arr)/k
        bias_test = np.sum(bias_test_arr)/k
        # Create the final beta array:
        beta_average = np.zeros(int((degree+1)*(degree+2)/2))
        beta_var = np.zeros(int((degree+1)*(degree+2)/2))
        for m in range(int((degree+1)*(degree+2)/2)):
            sum = 0
            for n in range(k):
                sum += beta_arr[n][m]
            beta_average[m] = sum/k
            beta_var[m] = np.var(beta_arr[:,m])
        # Print bias, variance, MSE and R2:
        printCalculation(var_test, bias_test, MSE_test, R2_test)
        printCI(bias_test, np.var(bias_test_arr), "Bias")
        printCI(var_test, np.var(var_test_arr), "Var ")
        printCI(MSE_test, np.var(MSE_test_arr), "MSE ")
        printCI(R2_test, np.var(R2_test_arr), "R2  ")
        # Print beta values:
        printBeta(beta_average)
        for ind in range(len(beta_average)):
            if len(str(ind))==1:
                text = str(ind) + " "
            else:
                text = str(ind)
            printCI(beta_average[ind], beta_var[ind], "beta"+text)

def lassoPrint(x, y, alph, method=FrankeFunction):
    """
    Runs the implemented lasso function to get values,
    then calculated the variance, bias, MSE and R2-score,
    and finally prints out the results.
    """
    # Get test-data-size and noise degree:
    size = getNumberInput("  Size of test-data-set (from 0 to 1): ", "both")
    # Get noise degree:
    noise_deg = getNumberInput("  Degree of noise (from 0 to 1): ", True)
    while True:
        degree = polynomialDegreeMenu()
        if degree in ["Q","q"]:
            break
        elif degree in ["B","b"]:
            lassoMenu(x, y, alph)
            break
        beta, H, z, z2 = lasso(x, y, alph, size, noise_deg, degree, method)
        variance, bias, error, R2 = calculation(z, z2)
        printCalculation(variance, bias, error, R2)
        printBeta(beta, np.var(z-z2)*H)

def lasso(x, y, alph, size, noise_deg, degree, method=FrankeFunction):
    """
    Runs the regular Lasso routine and returns
    the beta, H-matrix, true z and predicted z.
    """
    # Mesh, ravel and split data:
    x_, y_ = np.meshgrid(x, y)
    x_, y_ = np.ravel(x_), np.ravel(y_)
    x_train, x_test, y_train, y_test = train_test_split(x_, y_, test_size=size, random_state=42)

    # Create noise:
    noise_train = noise_deg*np.random.randn(int(np.sqrt(len(x_train)*len(y_train))))
    noise_test = noise_deg*np.random.randn(int(np.sqrt(len(x_test)*len(y_test))))

    # Find design-matrix X_train and find beta
    X_train = CreateDesignMatrix_X(x_train, y_train, degree)
    X_test = CreateDesignMatrix_X(x_test, y_test, degree)
    z_train = method(x_train, y_train) + noise_train
    z_test = method(x_test, y_test) + noise_test
    clf = linear_model.Lasso(alpha=alph, fit_intercept=False)
    clf.fit(X_train, z_train)
    beta = clf.coef_
    z_pred = X_test @ beta

    return beta, (X_train.T @ X_train), z_test, z_pred

def lassoCV(x, y, alph, method=FrankeFunction):
    """Runs Lasso routine with k-fold cross validation."""
    # Get k:
    k = getNumberInput("  Number of folds: ", "greater1", False)
    # Get noise degree:
    noise_deg = getNumberInput("  Degree of noise (from 0 to 1): ", True)
    while True:
        # Get polynomial degree:
        degree = polynomialDegreeMenu()
        if degree in ["Q","q"]:
            break
        elif degree in ["B","b"]:
            lassoMenu(x, y, alph)
            break
        # Initializing values and arrays:
        mesh_size, fold_size, noise_train, noise_test, x_rav, y_rav, ind = CVInitialize(x, y, k, noise_deg)
        x_train, y_train = np.zeros(mesh_size - fold_size), np.zeros(mesh_size - fold_size)
        x_test, y_test = np.zeros(fold_size), np.zeros(fold_size)
        MSE_test_arr, R2_test_arr, var_test_arr, bias_test_arr = np.zeros(k), np.zeros(k), np.zeros(k), np.zeros(k)
        beta_arr = np.zeros([k, int((degree+1)*(degree+2)/2)])
        # Starting k-fold algorithm
        for i in range(k):
            # Setting up index arrays:
            train_ind = np.array(list(ind[:i*fold_size])+list(ind[(i+1)*fold_size:]))
            test_ind = ind[i*fold_size:(i+1)*fold_size]
            # Setting up train-test-split:
            for j in range(mesh_size - fold_size):
                x_train[j] = x_rav[train_ind[j]]
                y_train[j] = y_rav[train_ind[j]]
            for l in range(fold_size):
                x_test[l] = x_rav[test_ind[l]]
                y_test[l] = y_rav[test_ind[l]]
            # Running Lasso routine:
            X_train = CreateDesignMatrix_X(x_train, y_train, degree)
            X_test = CreateDesignMatrix_X(x_test, y_test, degree)
            z_train = method(x_train, y_train) + noise_train
            z_test = method(x_test, y_test) + noise_test
            clf = linear_model.Lasso(alpha=alph, fit_intercept=False)
            clf.fit(X_train, z_train)
            beta = clf.coef_
            z_pred = X_test @ beta
            # Saving results:
            var_test_arr[i],bias_test_arr[i],MSE_test_arr[i],R2_test_arr[i] = calculation(z_test, z_pred)
            beta_arr[i] = beta
        # Calculate the mean of arrays:
        MSE_test = np.sum(MSE_test_arr)/k
        R2_test = np.sum(R2_test_arr)/k
        var_test = np.sum(var_test_arr)/k
        bias_test = np.sum(bias_test_arr)/k
        # Create the final beta array:
        beta_average = np.zeros(int((degree+1)*(degree+2)/2))
        for m in range(int((degree+1)*(degree+2)/2)):
            sum = 0
            for n in range(k):
                sum += beta_arr[n][m]
            beta_average[m] = sum/k
        # Print bias, variance, MSE and R2:
        printCalculation(var_test, bias_test, MSE_test, R2_test)
        printCI(bias_test, np.var(bias_test_arr), "Bias")
        printCI(var_test, np.var(var_test_arr), "Var ")
        printCI(MSE_test, np.var(MSE_test_arr), "MSE ")
        printCI(R2_test, np.var(R2_test_arr), "R2  ")
        # Print beta values:
        printBeta(beta_average)
        var_beta = np.var(beta_average)
        for ind in range(len(beta_average)):
            if len(str(ind))==1:
                text = str(ind) + " "
            printCI(beta_average[ind], var_beta, "beta"+text)

def compare(x, y, method=FrankeFunction):
    """ Algorithm for comparing results from each of the regression methods."""
    # Get k:
    k = getNumberInput("  Number of folds: ", "greater1", False)
    # Get noise degree:
    noise_deg = getNumberInput("  Degree of noise (from 0 to 1): ", True)
    # Get Ridge-parameter:
    lamb = getNumberInput("  Choice of Ridge parameter (>0): ", "greater")
    # Get Lasso parameter:
    alph = getNumberInput("  Choice of Lasso parameter (<0.05): ", "greater")
    while True:
        # Get polynomial degree:
        degree = polynomialDegreeMenu()
        if degree in ["Q","q"]:
            break
        elif degree in ["B","b"]:
            mainMenu(x, y)
            break
        # Initializing values and arrays:
        mesh_size, fold_size, noise_train, noise_test, x_rav, y_rav, ind = CVInitialize(x, y, k, noise_deg)
        x_train, y_train = np.zeros(mesh_size - fold_size), np.zeros(mesh_size - fold_size)
        x_test, y_test = np.zeros(fold_size), np.zeros(fold_size)
        # OLS:
        MSE_test_ols, R2_test_ols, var_test_ols, bias_test_ols = np.zeros(k), np.zeros(k), np.zeros(k), np.zeros(k)
        beta_ols = np.zeros([k, int((degree+1)*(degree+2)/2)])
        # Ridge:
        MSE_test_ridge, R2_test_ridge, var_test_ridge, bias_test_ridge = np.zeros(k), np.zeros(k), np.zeros(k), np.zeros(k)
        beta_ridge = np.zeros([k, int((degree+1)*(degree+2)/2)])
        # Lasso:
        MSE_test_lasso, R2_test_lasso, var_test_lasso, bias_test_lasso = np.zeros(k), np.zeros(k), np.zeros(k), np.zeros(k)
        beta_lasso = np.zeros([k, int((degree+1)*(degree+2)/2)])
        # Starting k-fold algorithm
        for i in range(k):
            train_ind = np.array(list(ind[:i*fold_size])+list(ind[(i+1)*fold_size:]))
            test_ind = ind[i*fold_size:(i+1)*fold_size]

            for j in range(mesh_size - fold_size):
                x_train[j] = x_rav[train_ind[j]]
                y_train[j] = y_rav[train_ind[j]]

            for l in range(fold_size):
                x_test[l] = x_rav[test_ind[l]]
                y_test[l] = y_rav[test_ind[l]]
            X_train = CreateDesignMatrix_X(x_train, y_train, degree)
            X_test = CreateDesignMatrix_X(x_test, y_test, degree)
            z_train = method(x_train, y_train) + noise_train
            z_test = method(x_test, y_test) + noise_test
            # Get OLS prediction
            z_ols, z_pred_ols = ols(x_train, y_train, noise_deg, degree, method = method, compare=True)[:2]
            # Get Ridge prediction
            z_ridge, z_pred_ridge = ols(x_train, y_train, noise_deg, degree, method = method,  lamb=lamb, compare=True)[:2]
            # Find Lasso prediction:
            clf = linear_model.Lasso(alpha=alph, fit_intercept=False, normalize=True)
            clf.fit(X_train, z_train)
            beta_tmp_lasso = clf.coef_
            z_pred_lasso = X_test @ beta_tmp_lasso
            var_test_ols[i],bias_test_ols[i],MSE_test_ols[i],R2_test_ols[i] = calculation(z_ols, z_pred_ols)
            var_test_ridge[i],bias_test_ridge[i],MSE_test_ridge[i],R2_test_ridge[i] = calculation(z_ridge, z_pred_ridge)
            var_test_lasso[i],bias_test_lasso[i],MSE_test_lasso[i],R2_test_lasso[i] = calculation(z_test, z_pred_lasso)
            # beta_ols[i] = beta_tmp_ols
            # beta_ridge[i] = beta_tmp_ridge
            beta_lasso[i] = beta_tmp_lasso
        # Calculate the averages of OLS arrays:
        MSE_ols, R2_ols, var_ols, bias_ols, beta_average_ols = averageCV(k, degree, MSE_test_ols, R2_test_ols, var_test_ols, bias_test_ols, beta_ols)
        # Calculate the averages of Ridge arrays:
        MSE_ridge, R2_ridge, var_ridge, bias_ridge, beta_average_ridge = averageCV(k, degree, MSE_test_ridge, R2_test_ridge, var_test_ridge, bias_test_ridge, beta_ridge)
        # Calculate the averages of Lasso arrays:
        MSE_lasso, R2_lasso, var_lasso, bias_lasso, beta_average_lasso = averageCV(k, degree, MSE_test_lasso, R2_test_lasso, var_test_lasso, bias_test_ols, beta_lasso)

        ols_return = ols(x_rav, y_rav, noise_deg, degree, method = method, compare=True)
        z_ols, z_pred_ols = ols_return[:2]
        beta_ols, H_ols = ols_return[-2:]
        var_ols = calculation(z_ols, z_pred_ols)[0]
        beta_var_ols = np.diag(H_ols*var_ols)

        ridge_return = ols(x_rav, y_rav, noise_deg, degree, method = method, lamb=lamb, compare=True)
        z_ridge, z_pred_ridge = ridge_return[:2]
        beta_ridge, H_ridge = ridge_return[-2:]
        var_ridge = calculation(z_ridge, z_pred_ridge)[0]
        beta_var_ridge = np.diag(H_ridge)*var_ridge

        writeBetaCItoFile(beta_var_ols, beta_var_ridge, beta_lasso, beta_ols, beta_ridge, beta_average_lasso)

        printCompare(MSE_ols, MSE_ridge,MSE_lasso,R2_ols,R2_ridge,R2_lasso,\
            var_ols, var_ridge, var_lasso, bias_ols, bias_ridge, bias_lasso,\
            beta_ols,beta_ridge,beta_average_lasso)

def writeBetaCItoFile(beta_var_ols, beta_var_ridge, beta_lasso, beta_average_ols, beta_average_ridge, beta_average_lasso):
    size = beta_lasso.shape[1]
    # var_ols = np.zeros(size)
    # var_ridge = np.zeros(size)
    var_lasso = np.zeros(size)
    for i in range(size):
    #     var_ols[i] = np.var(beta_ols[:,i])
    #     var_ridge[i] = np.var(beta_ridge[:,i])
        var_lasso[i] = np.var(beta_lasso[:,i])
    file1 = open("confidenceIntervals.txt","w")
    for j in range(len(beta_average_ols)):
        file1.write("$\\beta_{"+str(j)+"}$ & %6.3f $\pm$ %6.3f & %6.3f $\pm$ %6.3f & %6.3f $\pm$ %6.3f\\\\"\
        %(beta_average_ols[j],1.96*np.sqrt(beta_var_ols[j]),beta_average_ridge[j],1.96*np.sqrt(beta_var_ridge[j]),beta_average_lasso[j],1.96*np.sqrt(var_lasso[j])))
        file1.write("\n")
    file1.close()


def printCompare(MSE_ols, MSE_ridge, MSE_lasso, R2_ols, R2_ridge, R2_lasso, \
                var_ols, var_ridge, var_lasso, bias_ols, bias_ridge, bias_lasso, \
                beta_average_ols, beta_average_ridge, beta_average_lasso):
    print("---------------------------------------------------")
    print("|                      RESULTS                    |")
    print("|-------------------------------------------------|")
    print("|          |    OLS     |   Ridge    |   Lasso    |")
    print("|----------+------------+------------+------------|")
    print("| MSE      | %10.4f | %10.4f | %10.4f |" % (MSE_ols, MSE_ridge, MSE_lasso))
    print("| R2-score | %10.4f | %10.4f | %10.4f |" % (R2_ols, R2_ridge, R2_lasso))
    print("| Variance | %10.4f | %10.4f | %10.4f |" % (var_ols, var_ridge, var_lasso))
    print("| Bias     | %10.4f | %10.4f | %10.4f |" % (bias_ols, bias_ridge, bias_lasso))
    print("|----------+------------+------------+------------|")
    for i in range(len(beta_average_ols)):
        print("| Beta%4d | %10.4f | %10.4f | %10.4f |" % (i, beta_average_ols[i], beta_average_ridge[i], beta_average_lasso[i]))
    print("---------------------------------------------------")

def averageCV(k, deg, MSE_arr, R2_arr, var_arr, bias_arr, beta_arr):
    """ Returns the average of MSE, R2, variance, bias and beta wrt. cross validation."""
    MSE = np.sum(MSE_arr,keepdims=True)/k
    R2 = np.sum(R2_arr,keepdims=True)/k
    var = np.sum(var_arr,keepdims=True)/k
    bias = np.sum(bias_arr,keepdims=True)/k
    beta_average = np.zeros(int((deg+1)*(deg+2)/2))
    for i in range(int((deg+1)*(deg+2)/2)):
        sum = 0
        for j in range(k):
            sum += beta_arr[j][i]
        beta_average[i] = sum/k
    return (MSE, R2, var, bias, beta_average)

def CVInitialize(x, y, k, noise_deg):
    """Returns initialization of k-fold cross validation"""
    mesh_size = len(x)*len(y)
    fold_size = int(mesh_size/k)
    noise_train = noise_deg*np.random.randn(mesh_size - fold_size)
    noise_test = noise_deg*np.random.randn(fold_size)
    x_mesh, y_mesh = np.meshgrid(x, y)
    x_rav, y_rav = np.ravel(x_mesh), np.ravel(y_mesh)
    ind = np.array(range(mesh_size))
    np.random.shuffle(ind)
    return (mesh_size, fold_size, noise_train, noise_test, x_rav, y_rav, ind)

def R2_score(y, y_):
    """Calculates and returns the R2-score."""
    n = len(y)
    nom = np.mean(np.mean((y - y_)**2), keepdims=True )*n
    denom = np.var(y, ddof=1)*(n-1)
    return (1 - nom/denom)

def calculation(z, z2, all=True):
    """Runs calculation for variance, MSE, bias and R2-score."""
    if len(z.shape)>1:
        z = np.ravel(z)
    if len(z2.shape)>1:
        z2 = np.ravel(z2)
    variance = np.mean(np.var(z2, keepdims=True))
    if all:
        bias =  np.mean(z - np.mean(z2, keepdims=True))**2
        error = np.mean((z - z2)**2)
        R2 = R2_score(z, z2)
        return (variance, bias, error, R2)
    else:
        return (variance, None, None, None)

def printCalculation(variance, bias, error, R2):
    print("      -------------------")
    print("      |     RESULTS     |")
    print("      -------------------")
    if (bias==None) or (error==None) or (R2==None):
        print("      dim(z) and dim(z2) are not equal.|")
        print("      Var(z2) = %.4f" % variance)
    else:
        print("      Bias    = %.4f" % bias)
        print("      Var(z2) = %.4f" % variance)
        print("      MSE     = %.4f" % error)
        print("      R2      = %.4f" % R2)

def printBeta(beta, covar_beta=None):
    for i in range(len(beta)):
        print("      beta%3d  = %.4f" % (i,beta[i]))
    if (covar_beta is not None):
        for i in range(len(beta)):
            print("      Var(beta%3d) = %.4f" % (i,np.sqrt(covar_beta[i][i])))
        for i in range(len(beta)):
            print("      CI (95%%) - beta%3d: (%.4f, %.4f)"%(i, beta[i]-1.96*np.sqrt(covar_beta[i][i]), beta[i]+1.96*np.sqrt(covar_beta[i][i])))

def printCI(x, var_x, text):
    """Prints confidence interval"""
    print("      CI (95%%) - "+text+": (%.4f, %.4f)" % (x-1.96*np.sqrt(var_x), x+1.96*np.sqrt(var_x)))

def getNumberInput(text, cond, flt=True):
    """Gets number input from user. int of float."""
    while True:
        inp = input(text)
        try:
            if flt: output = float(inp)
            else: output = int(inp)
            if cond=="greater":
                breakCondition = (0<=output)
            elif cond=="greater1":
                breakCondition = (1<output)
            elif cond=="both":
                breakCondition = ((0<output)and(output<1))
            else: breakCondition = cond
            if breakCondition: break
        except: print("  Type valid input...")
    return output

def getTerrainData(name):
    from imageio import imread
    terrain = imread(name+'.tif')#[0:501,0:501]
    m, n = terrain.shape
    x, y = np.linspace(0, 1, m), np.linspace(0, 1, n)
    x_ind, y_ind = np.arange(0, m, 1), np.arange(0, n, 1)
    return x, y, x_ind, y_ind, terrain


def terrainFunction(x,y):
    if len(x.shape) > 1:
        x = np.ravel(x)
        y = np.ravel(y)
    x = (x*3600).astype(int)
    y = (y*1800).astype(int)
    # x = (x*500).astype(int)
    # y = (y*500).astype(int)
    z = getTerrainData("SRTM_data_Norway_1")[-1]
    z_new = np.zeros(len(x))
    z_new[:] = z[x[:], y[:]]
    return z_new


def dataMenu():
    """Menu for choice of data"""
    while True:
        print("-----------------------------")
        print("|         DATA MENU         |")
        print("|---------------------------|")
        print("| 1) Terrain data           |")
        print("| 2) Franke's function      |")
        print("| Q) Quit                   |")
        print("-----------------------------")
        inp = input("Input: ")
        if inp in ["Q","q"]:
            break
        elif inp == '1':
            x, y, x_ind, y_ind, z = getTerrainData("SRTM_data_Norway_1")
            mainMenu(x, y, terrain=True)
            break
        elif inp == '2':
            size = getNumberInput("Size of the data: ", "greater1", False)
            x, y = np.linspace(0, 1, size), np.linspace(0, 1, size)
            mainMenu(x, y)
            break

def mainMenu(x, y, terrain=False):
    """Main menu of user interface."""
    while True:
        print("-----------------------------")
        print("|         MAIN MENU         |")
        print("|---------------------------|")
        print("| 1) Ordinary Least Squares |")
        print("| 2) Ridge regression       |")
        print("| 3) Lasso regression       |")
        print("| 4) Compare methods        |")
        print("| B) Back                   |")
        print("| Q) Quit                   |")
        print("-----------------------------")
        inp = input("Input: ")
        if inp in ["Q","q"]:
            break
        elif inp in ["B","b"]:
            dataMenu()
            break
        elif terrain:
            if inp == '1':
                olsMenu(x, y, terrain=True)
                break
            elif inp == '2':
                # Get Ridge-parameter:
                lamb = getNumberInput("Choice of Ridge parameter (>0): ", "greater")
                ridgeMenu(x, y, lamb, terrain=True)
                break
            elif inp == '3':
                # Get test-data-size:
                alph = getNumberInput("Choice of Lasso parameter (<0.05): ", "greater")
                lassoMenu(x, y, alph, terrain=True)
                break
            elif inp == '4':
                compare(x, y, method=terrainFunction)
                break
        else:
            if inp == '1':
                olsMenu(x, y)
                break
            elif inp == '2':
                # Get Ridge-parameter:
                lamb = getNumberInput("Choice of Ridge parameter (>0): ", "greater")
                ridgeMenu(x, y, lamb)
                break
            elif inp == '3':
                # Get test-data-size:
                alph = getNumberInput("Choice of Lasso parameter (<0.05): ", "greater")
                lassoMenu(x, y, alph)
                break
            elif inp == '4':
                compare(x, y)
                break

def olsMenu(x, y, terrain=False):
    """OLS menu of user interface."""
    while True:
        print("  --------------------------------")
        print("  |          OLS menu            |")
        print("  |------------------------------|")
        print("  | 1) Train-test-split          |")
        print("  | 2) k-fold cross validation   |")
        print("  | 3) Regular OLS without split |")
        print("  | B) Back                      |")
        print("  | Q) Quit                      |")
        print("  --------------------------------")
        inp = input("  Input: ")
        if inp in ["Q","q"]:
            break
        elif inp in ["B","b"]:
            mainMenu(x, y, terrain)
            break
        elif terrain:
            if inp == '1':
                trainTestSplit(x, y, "ols", terrainFunction)
                break
            elif inp == '2':
                olsRidgeCV(x, y, "ols", terrainFunction)
                break
            elif inp == '3':
                olsPrint(x, y, terrainFunction)
                break
        else:
            if inp == '1':
                trainTestSplit(x, y, "ols")
                break
            elif inp == '2':
                olsRidgeCV(x, y, "ols")
                break
            elif inp == '3':
                olsPrint(x, y)
                break

def ridgeMenu(x, y, lamb, terrain=False):
    """Ridge menu of user interface."""
    while True:
        print("  ------------------------------")
        print("  |         Ridge menu         |")
        print("  |----------------------------|")
        print("  | 1) Train-test-split        |")
        print("  | 2) k-fold cross validation |")
        print("  | B) Back                    |")
        print("  | Q) Quit                    |")
        print("  ------------------------------")
        inp = input("  Input: ")
        if inp in ["Q","q"]:
            break
        elif inp in ["B","b"]:
            mainMenu(x, y, terrain)
            break
        elif terrain:
            if inp == '1':
                trainTestSplit(x, y, "ridge", terrainFunction, lamb=lamb)
                break
            elif inp == '2':
                olsRidgeCV(x, y, "ridge", terrainFunction, lamb=lamb)
                break
        else:
            if inp == '1':
                trainTestSplit(x, y, "ridge", lamb=lamb)
                break
            elif inp == '2':
                olsRidgeCV(x, y, "ridge", lamb=lamb)
                break

def lassoMenu(x, y, alph, terrain=True):
    """Lasso menu of user interface."""
    while True:
        print("  ------------------------------")
        print("  |         Lasso menu         |")
        print("  |----------------------------|")
        print("  | 1) Train-test-split        |")
        print("  | 2) k-fold cross validation |")
        print("  | B) Back                    |")
        print("  | Q) Quit                    |")
        print("  ------------------------------")
        inp = input("  Input: ")
        if inp in ["Q","q"]:
            break
        elif inp in ["B","b"]:
            mainMenu(x, y, terrain)
            break
        elif terrain:
            if inp == '1':
                lassoPrint(x, y, alph, terrainFunction)
                break
            elif inp == '2':
                lassoCV(x, y, alph, terrainFunction)
                break
        else:
            if inp == '1':
                lassoPrint(x, y, alph)
                break
            elif inp == '2':
                lassoCV(x, y, alph)
                break

def polynomialDegreeMenu():
    """Returns the polynomial degree after asking user."""
    while True:
        print("    -----------------------------------------------")
        print("    | Choose a polynomial degree of type integer. |")
        print("    |     Q) Quit               B) Back           |")
        print("    -----------------------------------------------")
        inp = input("    Input: ")
        if inp in ["Q","q","B","b"]:
            return inp
        try:
            degree = int(inp)
            return degree
        except: print("Type valid input...")

if __name__=='__main__':
    seed(np.random.randint(0,10000))
    dataMenu()
