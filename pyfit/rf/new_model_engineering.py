#%% imports
import time
START = time.time()
def elapsed_time(prefix='Time since last call = '):
    global START
    print(prefix, end=' ')
    now =time.time()
    print(round(now - START, 2))
    START = now 
elapsed_time()
#from private_versions.malleable_constants import C_DROPBOX_PATH as DROPBOX_PATH
from private_versions.private_constants import DROPBOX_PATH
CENTURY = 20
elapsed_time()
from general_helpers import drop_columns_containing, get_columns_containing, safe_make_folder
elapsed_time()
import pandas as pd 
elapsed_time()
import numpy as np 
elapsed_time()
#from RandomForest import randomForestRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
elapsed_time()
import matplotlib.pyplot as plt
import joblib



def drop_back_and_forward_features(df, back_steps, forward_steps, target):

    bad_backward = back_steps.symmetric_difference({1,2,3,6})
    bad_forward = forward_steps.symmetric_difference({1,2,3,4,5,6})
    
    bad_columns = []
    bad_columns.append('total_points_N')
    for i in bad_backward:
        bad_columns.append('_L'+str(i))
    for i in bad_forward:
        bad_columns.append('FIX'+str(i))
    if max(back_steps) < 6: 
        bad_columns = bad_columns + ['SALOC'] #may have nans early in year, prob won't after 6 games

    regressors = drop_columns_containing(bad_columns, df)
    try: #want to keep num_opponents
        regressors = regressors.drop('opponent', axis=1)
    except KeyError:
        pass
    return regressors.replace([np.inf, -np.inf], 0)


def drop_nan_rows(X,y):
    full = pd.concat([X,y],axis=1).dropna()
    X = full.iloc[:, :-1]
    y = full.iloc[:, -1]
    #print('without na we got: ', X.shape,y.shape)
    return X,y

def prepare_for_regression(df, back, forward, target, remove_inactive=False):
    # rid of no minutes in 6 weeks
    if remove_inactive:#6 in back:
        df = df.loc[df['minutes_L6']>0]

    # rid of blanks for 1wk predictions
    if target == 1:
        df = df.loc[df['FIX1_num_opponents']>0]

    # rid of gameweeks for model forward/back
    df = df.loc[(df['gw']>max(back)) & (df['gw']<40-target)]

    # get the y column
    targ = 'total_points_N' + str(target)
    target_df = df[targ]

    # rid of features for model forward/back
    print('drop features')
    df = drop_back_and_forward_features(df, back, forward, target)

    #get the matrices for regression & train model
    X,y = drop_nan_rows(df, target_df)
    print('nan dropped')
    return X, y 

# model should implement fit_model(X,y)
def train_model(train_df, features, back, forward, target, fit_model_func, train_benchmark):
    # TRAINING
    df = train_df.loc[:, features]
    remove_inactive = False
    if train_benchmark == 'six':
        remove_inactive = True
    print('prepare for regression')
    X, y = prepare_for_regression(df, back, forward, target, remove_inactive=remove_inactive)
    print('before fit')
    model = fit_model_func(X, y)
    feature_names = X.columns.to_list() 
    
    print('OOB SCORE IS ::==:: ', model.oob_score_)
    return model, feature_names

# Various benchmarks are available
# 'full' test on every player
# 'six'  test only people who have played a minute in last 6
#  int   test only people who play at least that many minutes this week
def test_model(test_df, features, back, forward, target, model, benchmark):
    # test
    df = test_df.loc[:, features]
    remove_inactive = False
    if benchmark == 'six':
        remove_inactive = True
    Xtest, ytest = prepare_for_regression(df, back, forward, target, remove_inactive=remove_inactive)

    # errors
    ypred = model.predict(Xtest)
    err = np.mean(np.square((ypred - ytest)))    

    return err, ypred, ytest


def plot_feature_importances(model, feature_names, filename, num_features_to_show = 30):
    importances = model.feature_importances_
    std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
    df = pd.DataFrame([importances, std], columns=feature_names)
    df.sort_values(0, axis=1, ascending=False, inplace=True)
    df = df.iloc[:, :num_features_to_show]
    forest_importances = df.loc[0, :]
    std = df.loc[1, :]
    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=std, ax=ax)
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()

    saveas = DROPBOX_PATH + 'experiment_plots/' + filename
    folder = saveas[:-len(saveas.split('/')[-1])]
    safe_make_folder(folder)
    fig.savefig(saveas)



# preprocessing
season_strings = [f'{CENTURY}{sy}-{sy+1}' for sy in (16, 17, 18, 20, 21)]
season_ints = [int(f[2:4] + f[5:7]) for f in season_strings]
yearly_dataset_paths = [DROPBOX_PATH + 'Our_Datasets/' + season_string + '/' + f'Processed_Dataset_{season_int}.csv' for season_string, season_int in zip(season_strings, season_ints)]
elapsed_time()
#_L1, _L3, _L6, _SALOC, _SAT

# choosing features 
FEATURES_NECESSARY = ['gw', 'FIX1_num_opponents', 'position']
FEATURES_Y = ['total_points_N1']
FEATURES_OPTIONAL = ['day']
for ending in ('_L1', '_L6', '_SALOC'):
    for stat in ['minutes', 'influence', 'threat', 'creativity', 'bps', 'assists', 'goals_scored', 'saves']:
        FEATURES_OPTIONAL.append(f'{stat}{ending}')
FEATURES = FEATURES_NECESSARY  + FEATURES_OPTIONAL + FEATURES_Y
NUM_FEATURES = len(FEATURES)

# downloading dataset
TRAIN_DF = pd.read_csv(yearly_dataset_paths[0])
TEST_DF0 = pd.read_csv(yearly_dataset_paths[0])
TEST_DF1 = pd.read_csv(yearly_dataset_paths[1])
TEST_DF2 = pd.read_csv(yearly_dataset_paths[2])
TEST_DF3 = pd.read_csv(yearly_dataset_paths[3])
TEST_DF4 = pd.read_csv(yearly_dataset_paths[4])
elapsed_time('Loading datasets')


#%%

# returns a fit model on the data
def randomForestRegression(X, y, n, max_features=1.0):
    model = RandomForestRegressor(n_estimators=n, max_features=max_features, oob_score=True)
    model.fit(X,y)
    return model

def print_error(err, i, max_features, n, train_benchmark, test_benchmark):
    def make_string(obj, available):
        word = str(obj)
        return word + ' '*(max(0, available-len(word)))

    i = '| ' + str(i) # to signal divide between train info (the model) and test info (the benchmark)
    
    final = ''
    for thing, available in zip((max_features, n, train_benchmark, i, test_benchmark), (7, 5, 6, 4, 6)):
        final += make_string(thing, available) + '| '
    final += str(round(err, 5))
    print(final)
    return final 


def plot_confusion_matrix(ypred, ytest, filename):
    # compare regions ## CONFUSION MATRICES
    def myround(x, base=1):
        return base * round(x/base)
    ypred = [myround(x) for x in ypred]
    ytest = [myround(x) for x in ytest]

    cm = confusion_matrix(ytest, ypred, normalize=None, labels=list(range(0, 9)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()
    #disp.savefig(DROPBOX_PATH + 'experiment_plots/' + filename)

# set parameters to scan over
"""
ns = range(10, 120, 10)
max_featuress = (1.0, 'sqrt', 'log2', round(1 / NUM_FEATURES, 4))
train_benchmarks = ['full', 'six']
test_benchmarks = ['full', 'six']
test_dfs = [TEST_DF0, TEST_DF1, TEST_DF2, TEST_DF3, TEST_DF4]
"""
ns = range(60, 70, 10)
max_featuress = (1.0, 'sqrt', 'log2', round(1 / NUM_FEATURES, 4))
train_benchmarks = ['full', 'six']
test_benchmarks = ['full', 'six']
test_dfs = [TEST_DF0, TEST_DF1, TEST_DF4]

PAST_YEAR_MODEL = DROPBOX_PATH + 'models/Current/no_ictANY_transfers_price/1236-1-1.sav'

ns = [10, 150, 175, 190]
max_featuress = reversed([1.0, 'sqrt', 'log2', round(3 / NUM_FEATURES, 5), round(1 / NUM_FEATURES, 5)]) # also trying choice of 3
print(max_featuress)
train_benchmarks = ['full', 'six']
test_benchmarks = ['six']
test_dfs = [TEST_DF4]

TRAIN_DF = pd.concat([TEST_DF0, TEST_DF1, TEST_DF2, TEST_DF3], axis=0)
TRAIN_DF = TEST_DF1

# TO DO A FULL TESTING
TRAIN_DF = drop_columns_containing(['season_'], TRAIN_DF) #### IN CASE WE FORGET TO REMOVE FROM THE END OF SEASON IF THIS WAS NOT SOME SORT OF ONE TIME ERROR
TRAIN_DF = drop_columns_containing(['element', 'season', 'name', 'minutes_N'],TRAIN_DF)
TRAIN_DF = TRAIN_DF.drop('team', axis=1)
FEATURES = TRAIN_DF.columns

train_back = {1,2,3,6}
train_forward = {1}
train_target = 1
test_back = {1,2,3,6}
test_forward = {1}
test_target = 1

EXPERIMENT_NAME = 'full_data_bign'

error_prints = []
print('hello')
elapsed_time(prefix='Test Year')
for n in ns:
    for max_features in max_featuress: # last one get's no choice on the split
        print('started a max features')
        for train_benchmark in train_benchmarks:
            print('in train_benchmark')
            fit_model_func = lambda x,y: randomForestRegression(x,y,n, max_features=max_features)
            model, feature_names = train_model(TRAIN_DF, FEATURES, train_back, train_forward, train_target, fit_model_func, train_benchmark)
            print('trained model')
            #model, FEATURES = joblib.load(PAST_YEAR_MODEL)
            #FEATURES += ['total_points_N1']
            filename = EXPERIMENT_NAME + '/' + str(train_benchmark) + str(max_features) + str(n) + '.png'
            plot_feature_importances(model, feature_names, filename)
            
            for i, test_df in enumerate(test_dfs):
                print('started the test df')
                for test_benchmark in test_benchmarks:
                    filename = str(i) + str(max_features) + str(n) + str(train_benchmark) + str(test_benchmark) + '.png'
                    err, ypred, ytest = test_model(test_df, FEATURES, test_back, test_forward, test_target, model, test_benchmark)
                    #plot_confusion_matrix(ypred, ytest, filename)
                    elapsed_time()
                    error_prints.append(print_error(err, i, max_features, n, train_benchmark, test_benchmark))

print('printing all the errors')
for errorprint in error_prints:
    print(errorprint)

## TEST THIS YEARS MODEL 



#TRAIN_DF = pd.concat([pd.read_csv(path, index_col=0) for path in yearly_dataset_paths], axis=0).reset_index(drop=True)
#TRAIN_DF = drop_columns_containing(['season_'], TRAIN_DF) #### IN CASE WE FORGET TO REMOVE FROM THE END OF SEASON IF THIS WAS NOT SOME SORT OF ONE TIME ERROR
#TRAIN_DF = drop_columns_containing(['element', 'season', 'name', 'minutes_N'],TRAIN_DF)
#TRAIN_DF = TRAIN_DF.drop('team', axis=1)

#KEEPER_RAW_DF = TRAIN_DF.loc[TRAIN_DF['position']==1.0]
#KEEPER_DF = drop_columns_containing(['creativity', 'ict', 'threat', 'transfers', 'selected'],KEEPER_RAW_DF)
#FIELD_DF = TRAIN_DF.loc[TRAIN_DF['position']!=1.0]


# train two seperate models 

# RF, STANDARD METHODS

# NEURAL NETWORK


# EVALUATE TIMING, ACCURACY TRAIN/TEST, ACCURACY ON THE 21-22 SEASON, AND SCORE ON THE 21-22 SEASON


# make cyclic so repeatable as we add complexity to model and parameters. 