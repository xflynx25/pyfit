import pandas as pd
import numpy as np 
import math 
from RandomForest import randomForestRegression
import pickle
from general_helpers import drop_columns_containing, get_columns_containing
from constants import MAX_FORWARD_CREATED, TRANSFER_MARKET_VISUALIZATION_ROUNDING
import joblib

# players that play less than this many minutes in the whole season will be dropped
# this is with the goal of reducing these players filled with 0s on 0s, that unnecessarily
# increase the dataset size and possibly give it some frusteration
def drop_players_with_minutes_below(df, min_minutes):
    processed = []
    for season in df['season'].unique():
        season_df = df.loc[df['season']==season]
        bad_players = []
        for player in season_df['element'].unique():
            player_df = season_df.loc[season_df['element']==player]
            minutes_played = player_df['minutes_L1'].sum()
            if minutes_played < min_minutes:
                bad_players.append(player)
        only_good_df = season_df.loc[~season_df['element'].isin(bad_players)]
        processed.append(only_good_df)
    return pd.concat(processed, axis=0, ignore_index=True)


def visualize_top_transfer_market(full_transfer_market, name_df, sort_key, n, healthy=False, allowed_healths=['a']):
    if type(healthy) != bool:
        full_transfer_market = full_transfer_market.loc[full_transfer_market['status'].isin(allowed_healths)]
    sorted_market = full_transfer_market.sort_values(sort_key,ascending=False).reset_index(drop=True)
    sorted_market = sorted_market.iloc[:n, :]
    print('\nplayer',' '*9 ,'This_wk',' '*8 ,'Next-6\n')
    for _,row in sorted_market.iterrows():
        name = name_df.loc[name_df['element']==row['element']]['name'].tolist()[0]
        name_length = len(name)
        next_pts = round(row['expected_pts_N1'], TRANSFER_MARKET_VISUALIZATION_ROUNDING)
        full_pts = round(row['expected_pts_full'], TRANSFER_MARKET_VISUALIZATION_ROUNDING)
        buffer1 = [max(15-name_length, 1) if full_pts >= 10 else max(16-name_length, 1)][0]
        buffer2 = 8
        print(name, ' '*buffer1, next_pts, ' '*buffer2, full_pts) 


# returns pos_1, pos_2, pos_3, pos_4 1-hot rather than position
def one_hot_the_positions(df):
    def one_hot_row(row):
        position = int(row['position'])
        new_cols = [0] * 4
        new_cols[position - 1] = 1
        new_cols = pd.Series(new_cols, index=['pos_1', 'pos_2', 'pos_3', 'pos_4'])
        df = pd.concat([row, new_cols], axis=0)
        return drop_columns_containing(['position'], df)
    df = df.apply(one_hot_row, axis=1, result_type='expand')
    return df


def undo_one_hot_the_positions(df):
    def undo_one_hot_row(row):
        relevant = get_columns_containing(['is_pos_'], row)#relevant = get_columns_containing(['pos_'], row)
        position = int(relevant.index[relevant.argmax()][-1])
        row = drop_columns_containing(['pos_'], row)
        row['position'] = position
        return row
    df = df.apply(undo_one_hot_row, axis=1, result_type='expand')
    return df


"""
# Should just be dumping a binary
def save_model(model, filename):
    pickle.dump(model, open(filename, 'wb'))


# There have been some issues loading these pickles,  
##### _pickle.UnpicklingError: invalid load key, '\x01'.
# but it seems that loading them with joblib can avoid 
# this problem and load it exactly correctly
#
# adding verify param to be function to pass the loaded object to 
# to make sure we are loading what we are expecting (ex. check that it is a tuple size 2)
def load_model(filename, verify_func=lambda x: True):
    try:
        data = pickle.load(open(filename, 'rb'))
    except:
        data = joblib.load(filename)
        print('had to use joblib')

    if not verify_func(data):
        raise Exception("Loaded data is not passing the test")
    
    return data

"""

"""NEW SAVE AND LOAD MODEL, WE WILL BE USING JOBLIB & CONSISTANT PYTHON VERSION AND CONDA ENV"""
# Should just be dumping a binary
def save_model(model, filename):
    joblib.dump(model, filename)


# @param: verify func will check that we don't have corrption
def load_model(filename, verify_func=lambda x: True):
    data = joblib.load(filename)

    if not verify_func(data):
        raise Exception("Loaded data is not passing the test")
    
    return data


### NEW HELPERS ####       

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
    print('without na we got: ', X.shape,y.shape)
    return X,y

def get_sets_from_name(name):
    params = name.split('-')
    form = set()
    opponents = set()
    guess = int(params[2])
    for let in params[0]:
        form.add(int(let))
    for let in params[1]:
        opponents.add(int(let))
    return form, opponents, guess

# I believe the original idea was to use thresholds but now we use a main sequence to do crossval descent
def train_rf_with_threshold(X,y,n,threshold, crossval=False, num_rounds=2, metric='mse', n_starter_cols=None):

    # bootstrap cv with mse/mae as train/eval metrics. Lowest cve are features chosen
    # threshold should be a list if we are doing crossval
    # keys num_features 

    def get_good_feature_columns_threshold(xx, yy, thresh): #problem is this isn't a descent so features might not be best
        # train 2 models 100 trees on full, cut features below threshold in both
        print('regression1')
        THRESHOLDTREES = 100
        good_features_1 = randomForestRegression(xx, yy, THRESHOLDTREES).feature_importances_ > thresh
        print('regression2')
        good_features_2 = randomForestRegression(xx, yy, THRESHOLDTREES).feature_importances_ > thresh
        good_features = good_features_1 | good_features_2
        return [col for col, good in zip(xx.columns, good_features) if good]

    if crossval:
        print('doing crossval')
        round_splits = [np.random.rand(X.shape[0]) < 0.8 for i in list(range(num_rounds))]
        scores = {}
        features = {}
        n_cols = len(X.columns.to_list())
        if n_starter_cols is not None and n_starter_cols < n_cols:
            n_cols = n_starter_cols

        main_seq = get_main_sequence(n_cols)# if type(threshold != list) else [int(threshold[i]*n_cols) for i in range(len(threshold))]
        while True: ###
            
            #if type(threshold) == list:
            #    pass
            if scores == {}:
                feature_options, num_features = X.columns, main_seq[0]
            else:
                feature_options, num_features = feature_information(scores, features, main_seq) 
            if num_features == 'done':
                break

            X_prev = X[feature_options]
            print('about to do a regression for feature importance')
            feature_importances = randomForestRegression(X_prev, y, 100).feature_importances_ 
            good_features = get_good_features(feature_options, feature_importances, num_features) 
                 
            X_rnd = X[good_features] 


            # train num_rounds models 2/3 * n trees, evaluate against crossval
            error = 0
            for msk in round_splits:
                Xtrain, Xtest, ytrain, ytest = X_rnd[msk], X_rnd[~msk], y[msk], y[~msk]
                model = randomForestRegression(Xtrain, ytrain, int(n*2/3))
                if metric == 'mae':
                    err = np.mean(np.abs((model.predict(Xtest) - ytest)))
                if metric == 'mse':
                    err = np.mean(np.square((model.predict(Xtest) - ytest)))
                print('round mae error for num_features=', num_features, ' :  ', round(err, 5))
                error += err / num_rounds

            print('avg ',str(metric),' error for ', num_features,' features:  ', round(error,5))
            scores[num_features] = error
            features[num_features] = good_features
            
        best_index = min(scores.keys(), key=lambda k: scores[k])
        chosen_features = features[best_index]


    if not crossval: 
        print('no crossval')
        chosen_features = get_good_feature_columns_threshold(X, y, threshold)

    print("Original: ", X.columns.size, "   New: ", len(chosen_features))
        
    # take better model and train with full n trees on full data set
    X = X[chosen_features]
    model = randomForestRegression(X,y,n)
    feature_names = X.columns.to_list() #trying this to maintain regressors
    return model, feature_names




def is_useful_model(key, gw):
    back, target = key
    if gw - back <= 0:
        return False 
    elif target==1:
        return True 

    if gw > 33 and target+gw != 39:
        return False 
    return True

def get_backward(row, backward_ops, forward):
    
    bad_columns = []
    # DEALING WITH THE FACT THAT IN PREVIOUS SEASONS WE HAVE TOTAL_POINTS_AND_MINUTES_COLUMNS 
    bad_columns += [f'_N{x}' for x in range(1, 38)]

    # first dealing with the FIX being nan canceling out the second part 
    for i in range(forward + 1, MAX_FORWARD_CREATED+1):
        bad_columns.append('FIX' + str(i))
    # also deal with the SALOC being nan
    bad_columns.append('_SALOC')
    row = drop_columns_containing(bad_columns, row)
    

    backward = 0
    for max_past in backward_ops:
        row = drop_columns_containing(['_L' + str(back) for back in range(max_past+1, MAX_FORWARD_CREATED+1)], row)
        if (row.isnull().values).sum() == 0:
            backward = max_past 
            break
        else:
            bad_cols = [row.columns[i] for i in range(len(row.columns)) if (row.isnull().values)[0][i]]
            #print(bad_cols)
    return backward

def long_term_benchwarmer(row):
    return row['minutes_L6'].to_list()[0] == 0.0

def blank_row(row):
    return row['FIX1_num_opponents'].to_list()[0] == 0.0



# Go down a main sequence of 4/5
# when you are lower than your 4/5 and 16/25, search your 9/10
# if 9/10 lower, search 19/20, else, search 17/20 --> pick minimum after this
#@return good_features, num_features (for the next), 'done' if should terminate
def feature_information(scores, features, main_seq):
    main_seq_scores = {k:v for k,v in scores.items() if k in main_seq}
    top_main_score = min(main_seq_scores, key=lambda k: main_seq_scores[k])
    index = [i for i, v in enumerate(main_seq) if v == top_main_score][0]
    next_1, next_2 = main_seq[index+1], main_seq[index+2]
    if index <= len(main_seq) -3:
        if (next_1 in scores) and (next_2 in scores): #so won't get to 1 oh well
            #we search between index - 1 and index:
            if index != 0 and scores[next_1] > scores[main_seq[index-1]]:
                train_from, num_features = search_ending(scores, top_main_score, backward=scores[main_seq[index-1]])
                if num_features == 'done':
                    return 'done', 'done'

            # we search between index and index + 1
            else:
                train_from, num_features = search_ending(scores, top_main_score)
                if num_features == 'done':
                    return 'done', 'done'

        else: #still on main sequence 
            if (next_1 not in scores):
                train_from = top_main_score
                num_features = next_1 
            elif (next_2 not in scores):
                num_features = next_2 
                train_from = next_1
    else: 
        return 'done', 'done'

    return features[train_from], num_features



def get_good_features(feature_options, feature_importances, num_features): 
    alpha = sorted(list(feature_importances), reverse = True)[num_features-1]
    good_features = [feature_options[i] for i in list(range(feature_importances.size)) if feature_importances[i] >= alpha]
    return good_features

def get_main_sequence(n, drop_rate = 13/20):#4/5) #SPEEDING THINGS UP WITH THE 13/20
    seq = [n]
    while n >=1:
        n = math.floor(n * drop_rate)
        seq.append(n)
    return seq
    

def search_ending(scores, top_main_score, backward=False):
    middle = math.floor(top_main_score * 9/10)
    if middle not in scores:
        train_from = top_main_score
        num_features = middle 
    else: #go up or down for final 
        checker_score = [backward if backward else top_main_score][0]
        shift = [1 if scores[middle]<checker_score else -1][0]
        if backward:
            shift *= 1
        where_to = math.floor(top_main_score * (18 + shift)/20)
        if where_to in scores:
            return 'done', 'done'
        else:
            train_from = [top_main_score if shift == 1 else middle][0]
            num_features = where_to
    return train_from, num_features

def keeper_outfield_split(current_gw_stats, health_df):
    keepers = current_gw_stats.loc[current_gw_stats['position']==1.0]
    outfield = current_gw_stats.loc[current_gw_stats['position']!=1.0]
    health_keepers = health_df.loc[health_df['element'].isin(keepers['element'])]
    health_outfield = health_df.loc[health_df['element'].isin(outfield['element'])]
    return keepers, health_keepers, outfield, health_outfield

# takes a list of strings and gets the closest player id to that name 
# metric will be minimum substring distance of length target
def get_elements_from_namestrings(bad_players, name_df, visualize=False):
    from jellyfish import levenshtein_distance
    #import distance 
    def str_distance(name, target):
        name, target = name.lower(), target.lower()
        min_score = float('inf')
        size = len(target) 
        buffer = ' ' * (size-1) 
        buffered_name = buffer + name + buffer 
        for start in range(len(buffered_name)-size+1):
            end = start+size
            score = levenshtein_distance(buffered_name[start:end],target)
            min_score = min(min_score, score)
        return min_score

    bad_elements = []
    for bad_guy in bad_players:
        scores = {}
        for _, row in name_df.iterrows():
            name = row['name']
            element = row['element']
            scores[element] = str_distance(name, bad_guy) 
        closest = sorted(scores, key=lambda x: scores[x])[0]
        bad_elements.append(closest)
        if visualize:
            bad_name = name_df.loc[name_df['element']==closest]['name'].to_list()[0]
            print('ELIMINATED ', bad_name)
    return bad_elements


#tm: full_transfer_market
# if someone doesn't want certain players on their team
def eliminate_players(full_transfer_market, bad_players, name_df, visualize=False):
    bad_player_elements = get_elements_from_namestrings(bad_players, name_df, visualize=visualize)
    
    """
    print('indicator1')
    for bad_player in bad_player_elements:  
        man = full_transfer_market.loc[full_transfer_market['element']==bad_player]
        time.sleep(2)
        print('indicator2')
        man.loc[:, 'expected_pts_full'] = 0
        man.loc[:, 'expected_pts_N1'] = 0
        time.sleep(2)
        print('indicator3')
        full_transfer_market = full_transfer_market.loc[full_transfer_market['element']!=bad_player]
        full_transfer_market = pd.concat([full_transfer_market, man], axis=0)
    """
        
    full_transfer_market.loc[full_transfer_market['element'].isin(bad_player_elements), ['expected_pts_full','expected_pts_N1']] = 0
    return full_transfer_market



#tm: full_transfer_market
# if someone doesn't want certain players on their team
def nerf_players(full_transfer_market, bad_players, name_df, scale_down, visualize=False):
    bad_player_elements = get_elements_from_namestrings(bad_players, name_df, visualize=visualize)
    for i in range(len(bad_player_elements)):  
        bad_player, downscale = bad_players[i], scale_down[i]
        man = full_transfer_market.loc[full_transfer_market['element']==bad_player]
        man.loc[:, 'expected_pts_full'] *= downscale
        #man.loc[:, 'expected_pts_N1'] = 0
        full_transfer_market = full_transfer_market.loc[full_transfer_market['element']!=bad_player]
        full_transfer_market = pd.concat([full_transfer_market, man], axis=0)
    return full_transfer_market


def save_market(current_gw, transfer_market, path):
    try:
        df = pd.read_csv(path, index_col=0)
        df = df.loc[df['gw']!=current_gw] #replace data for this gameweek
    except:
        df = pd.DataFrame()

    transfer_market = transfer_market.reset_index(drop=True)
    transfer_market.loc[:, 'gw'] = current_gw
    final_df = pd.concat([df, transfer_market], axis=0)
    final_df.to_csv(path)