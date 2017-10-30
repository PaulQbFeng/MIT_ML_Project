import numpy as np
from proj1_helpers import load_csv_data, predict_labels, create_csv_submission

# ------------------------------- USEFUL FUNCTIONS -------------------------------

def compute_mse(y, tx, w):
    """Calculate the loss using mse."""
    N = y.shape[0]
    e = y - tx @ w.T
    return 1 / (2 * N) * np.linalg.norm(e) ** 2

def ridge_regression(y, tx, lambda_):
    """Ridge regression."""
    N = tx.shape[0]
    a = (tx.T @ tx) + 2 * N * lambda_ * np.eye(tx.shape[1])
    b = tx.T @ y
    w = np.linalg.solve(a, b)
    return w, compute_mse(y, tx, w)

def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x, axis=0)
    x = x - mean_x
    std_x = np.std(x, axis=0)
    x = x / std_x
    return x, mean_x, std_x

def split_data(x, y, ratio, seed=None):
    """Split the dataset based on the split ratio. If ratio is 0.8 
    you will have 80% of your data set dedicated to training 
    and the rest dedicated to testing
    """
    # set seed
    if not seed is None:
        np.random.seed(seed)
        
    d = x.shape[0]
    di = int(d * ratio)
    
    per = np.random.permutation(d)
    
    xtraining = x[per][:di]
    ytraining = y[per][:di]
    xtesting = x[per][di:]
    ytesting = y[per][di:]
    
    return xtraining, ytraining, xtesting, ytesting

def build_poly(x, degree):
    """Polynomial basis functions for input data x, for j=0 up to j=degree."""
    
    # Columns of 1's
    phi = np.ones((x.shape[0], 1))
    for deg in range(1, degree+1):
        phi = np.c_[phi, x ** deg]
    return phi

def replace_nans_with_median(arr, nan=-999):
    '''Creates a copy and replaces the nan values by the median (without thos nan values) in the column'''
    N, D = arr.shape
    copy = arr.copy()
    
    for d in range(D):
        copy[:,d][copy[:,d] == nan] = np.median(arr[:,d][arr[:,d] != nan])
        
    return copy

def prediction(w, x_test, y_test, small=-1, big=1, verbose=False):
    '''Returns the value of how good the predictions were acoording to those parameters'''
    y_pred = x_test @ w
    sep_val = (small + big) / 2
    y_pred[y_pred < sep_val] = small
    y_pred[y_pred >= sep_val] = big
    
    bad = np.count_nonzero(y_pred - y_test)
    good = y_test.shape[0] - bad
    
    ratio = good / (good + bad)
    
    if verbose:
        print('Good: ', good)
        print('Bad: ', bad)
        print('Ratio: ', ratio)
    
    return ratio

def separate_by_col22(x, idd, y):    
    '''Separates the data by the value in column 22 of x (and removex said column for x)'''
    x_22 = [np.delete(x[x[:,22] == i], 22, 1) for i in range(4)]
    idd_22 = [idd[x[:,22] == i] for i in range(4)]
    y_22 = [y[x[:,22] == i] for i in range(4)]
    
    return x_22, idd_22, y_22

def delete_useless_col(x):
    '''Delete the columns we do not want in the sub-arrays'''
    # These are the columns for which the percentage of nan is 100%
    useless_cols = [[4, 5, 6, 12, 22, 23, 24, 25, 26, 27, 28], [4, 5, 6, 12, 25, 26, 27], [], []]
    return [np.delete(x[i], useless_cols[i], 1) for i in range(4)]

def predict(w, x_test, small=-1, big=1):
    '''Returns the prediction for x_test with w, for two values small and big. The prediction is done by choosing the nearest value'''
    y_pred = x_test @ w
    sep_val = (small + big) / 2
    y_pred[y_pred < sep_val] = small
    y_pred[y_pred >= sep_val] = big
    
    return y_pred

# ------------------------------- BEGINNING -------------------------------
print('Reading data')

yb_full, input_data_full, ids_full = load_csv_data('data/train.csv')
yb_test, input_data_test, ids_test = load_csv_data('data/test.csv')

# Shuffling a bit the data to get some subsample that is picked at random
np.random.seed(16)
per = np.random.permutation(250000)

# Picking sumsamples
yb, input_data, ids = yb_full[per][::10], input_data_full[per][::10], ids_full[per][::10]

print('Data read')
print('Treating data')

# Separating each np.array into 4 sub-arrays by category (number of jets aka column 22)
input_data_by_22, ids_by_22, yb_by_22 = separate_by_col22(input_data, ids, yb)

# Removing useless columns in the 4 sub-arrays
only_good_data = delete_useless_col(input_data_by_22)

def pseudo_cross_validation():
    bests = []
    
    # For each sub-array
    for i_22 in range(4):
        print('Starting for jet', i_22)
        # Finishing cleaning the data by removing last nan values and standardizing the data
        input_data_clean = replace_nans_with_median(only_good_data[i_22])
        input_data_std, _, _ = standardize(input_data_clean)

        # We will do the work 200 times to get nice values
        times = 2
        degrees = np.linspace(2, 10, 9).astype(int) # [1, 2, 3]
        lambdas = np.logspace(-5, -1, 14) # [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]

        # Where we store the predictions
        predictions = np.zeros([times, len(degrees), len(lambdas)])

        for time in range(times):
            # Splitting data into training and testing, 30% of data goes to training, 70% to testing
            x_tr, y_tr, x_te, y_te = split_data(input_data_std, yb_by_22[i_22], 0.3, seed=time)

            for i, degree in enumerate(degrees):
                phi_tr = build_poly(x_tr, degree)
                phi_te = build_poly(x_te, degree)

                for j, lambda_ in enumerate(lambdas):
                    try:
                        # Get best parameter
                        w, _ = ridge_regression(y_tr, phi_tr, lambda_)
                        
                        # Get prediction for value w given by Ridge, with the testing data
                        predictions[time, i, j] = prediction(w, phi_te, y_te)
                    except:
                        predictions[time, i, j] = 0

            # Stupid info about how much work we've done so far
            if time == 0.1 * times:
                print('   10%')
            if time == 0.2 * times:
                print('   20%')
            if time == 0.3 * times:
                print('   30%')
            if time == 0.4 * times:
                print('   40%')
            if time == 0.5 * times:
                print('   50%')
            if time == 0.6 * times:
                print('   60%')
            if time == 0.7 * times:
                print('   70%')
            if time == 0.8 * times:
                print('   80%')
            if time == 0.9 * times:
                print('   90%')
            if time == times - 1:
                print('  100%')
        
        # Getting the index where the degrees and lambdas gave the best prediction (as average for all 200 times)
        pos = np.unravel_index(np.median(predictions, axis=0).argmax(), np.median(predictions, axis=0).shape)
        bests.append((degrees[pos[0]], lambdas[pos[1]]))
        
    return bests

# Getting the result of the work above
bests = pseudo_cross_validation()

'''   _
     / \
    / | \
   /  |  \
  /   |   \
 /    O    \
 \_________/
The next array is the one for which we got the best results in the Kaggle leaderboard. We do not remember what we
used as parameters as we used different steps for the lambdas, and we also used random values without seeds when
separating the data... But we did save it so here it is. If you want to run the proper algorithm, comment the next line
'''
bests = [(7, 0.00059948425031894088), (10, 0.0016681005372000592), (7, 0.0016681005372000592), (6, 0.012915496650148827)]

# Now, we separate the full data, and the test data where we have to do rpedictions
input_data_full_by_22, ids_full_by_22, yb_full_by_22 = separate_by_col22(input_data_full, ids_full, yb_full)
input_data_test_by_22, ids_test_by_22, yb_test_by_22 = separate_by_col22(input_data_test, ids_test, yb_test)


print('Percentages of nan per column')
# We print the percentages of nans by category in the  full data, just as info
# For every sub-array
for i_22 in range(4):
    print('Jet', i_22)
    for c in range(input_data_full_by_22[i_22].shape[1]):
        tmp = input_data_full_by_22[i_22][:,c]
        
        if np.any(tmp[tmp == -999]):
            print('  ', c, ':', int(len(tmp[tmp == -999]) / len(tmp) * 100), '%')
            
only_good_data_full = delete_useless_col(input_data_full_by_22)
only_good_data_test = delete_useless_col(input_data_test_by_22)

preds = np.zeros(4)

print('Starting predictions')

for i_22 in range(4):
    full_std, _, _ = standardize(only_good_data_full[i_22])
    test_std, _, _ = standardize(only_good_data_test[i_22])
    
    phi_full = build_poly(full_std, bests[i_22][0])
    phi_test = build_poly(test_std, bests[i_22][0])
    
    w, _ = ridge_regression(yb_full_by_22[i_22], phi_full, bests[i_22][1])
    
    # Get predictions by nearest value
    yb_test_by_22[i_22] = predict(w, phi_test)
    
    preds[i_22] = prediction(w, phi_full, yb_full_by_22[i_22])
    print('Ratio of good predictions for jet', i_22, ':', preds[i_22])
    
# Weighted average for predictions
overall = (preds[0] * yb_full_by_22[0].shape[0] + preds[1] * yb_full_by_22[1].shape[0] + preds[2] * yb_full_by_22[2].shape[0] + preds[3] * yb_full_by_22[3].shape[0]) / (yb_full_by_22[0].shape[0] + yb_full_by_22[1].shape[0] + yb_full_by_22[2].shape[0] + yb_full_by_22[3].shape[0])

print('Overall prediction', overall)
    
print('Creating submission')
yb_submit = np.concatenate(yb_test_by_22)
ids_submit = np.concatenate(ids_test_by_22)

create_csv_submission(ids_submit, yb_submit, 'submission_by_cat.csv')
print('Done')