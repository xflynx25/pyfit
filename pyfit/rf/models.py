""" 
SUMMARY: old code back when I was implementing some sort of bayesian grid search on my own. Will be heavily edited later
"""

PATH = r"C://Users/___/FOLDER"
CENTURY = 20
from general_helpers import get_columns_containing, safe_make_folder, safer_eval
from Oracle_helpers import save_model, load_model, drop_back_and_forward_features, train_rf_with_threshold,\
    drop_nan_rows, get_backward, long_term_benchwarmer, get_sets_from_name
from general_helpers import drop_columns_containing, get_columns_containing
import time
import pandas as pd 
#TRAIN_DF = pd.read_csv(UPDATED_TRAINING_DB, index_col=0)
season_strings = [f'{CENTURY}{sy}-{sy+1}' for sy in (16, 17, 18, 20, 21)]
season_ints = [int(f[2:4] + f[5:7]) for f in season_strings]
TRAIN_DF = pd.concat([pd.read_csv(f'{PATH}Our_Datasets/{season_str}/Processed_Dataset_{season_int}.csv', index_col=0) for (season_str, season_int) in zip(season_strings, season_ints)], axis=0).reset_index(drop=True)
TRAIN_DF = drop_columns_containing(['season_'], TRAIN_DF) #### IN CASE WE FORGET TO REMOVE FROM THE END OF SEASON IF THIS WAS NOT SOME SORT OF ONE TIME ERROR
TRAIN_DF = drop_columns_containing(['element', 'season', 'name', 'minutes_N'],TRAIN_DF)
TRAIN_DF = TRAIN_DF.drop('team', axis=1)

KEEPER_RAW_DF = TRAIN_DF.loc[TRAIN_DF['position']==1.0]
KEEPER_DF = drop_columns_containing(['creativity', 'ict', 'threat', 'transfers', 'selected'],KEEPER_RAW_DF)
FIELD_DF = TRAIN_DF.loc[TRAIN_DF['position']!=1.0]

FULL_MODEL_NAMES = ['1236-136-6', '123-136-6','1-136-6','1-1-1','123-1-1',\
    '1236-1-1'] #just for testing everything is working well
FULL_MODEL_NAMES = ['1236-136-6', '123-136-6','12-136-6','1-136-6','1-1-1','123-1-1',\
    '1236-1-1','1236-135-5', '1236-124-4', '1236-123-3', '1236-12-2']
    

CROSSVAL = True
THRESHOLD = [.0001, .0004,.0008, .0011, .0015, .002 , .003, .004, .005, .006,\
    .007, .0085, .01, .012, .015, .02, .025, .03, .035,.04,.045, .05, .06,.07,.08,.09,.1, .15, .2, .3, .4]
TREE_SIZE = 150 #first few of 2021-22 season were at 175 (full, full_mse, onehot)
N_STARTER_COLS = 375 # we want to speed it up so since almost none use more than 300 cols in final choice we boost it

# model returns folder and df with only columns/rows we would like to use in suite
# suite takes care of applicable to all like beg/end of year, blanks, long-bench, 


# Create Suite of Model-Type
# df is TrainingDf but with possibly some players rid of per specific model
def train_model_suite(training_db, folder, names, n, threshold, crossval=False, metric='mse', n_starter_cols=None):
    print("Folder= ", folder,"\n")
    safe_make_folder( PATH + 'models/' + folder + "/")
    for name in names:
        print("model= ", name,"\n")
        start = time.time()

        back, forward, target = get_sets_from_name(name)
        df = training_db.copy()
    
        # rid of no minutes in 6 weeks
        if 6 in back:
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
        df = drop_back_and_forward_features(df, back, forward, target)
        
        #get the matrices for regression & train model
        X,y = drop_nan_rows(df, target_df)
        print('training model')
        model, feature_names = train_rf_with_threshold(X,y,n,threshold, crossval=crossval, metric=metric, n_starter_cols=n_starter_cols)
        model_name = PATH + 'models/' + folder + "/" + name + '.sav' 
        save_model((model, feature_names), model_name)

        end = time.time()
        print('took ', (end-start)/60, " minutes.")

# all the representations
def full_positional_representation(folder):
    df = FIELD_DF
    train_model_suite(df, folder, FULL_MODEL_NAMES, TREE_SIZE, THRESHOLD, n_starter_cols=N_STARTER_COLS, crossval=CROSSVAL, metric = 'mae')

''' TRAINING THE MODEL SUITES '''# %%
# first train full and test a regular season make sure it is still working, then train all the models
print(TRAIN_DF.shape)
folder_names = ['full']
models = [lambda x: full_positional_representation(x)]
for folder, model in zip(folder_names, models):
    print('starting')
    start = time.time()
    model(folder)
    end = time.time() 
    print("\n", folder, "took ", round((end-start)/60), " minutes\n\n")

