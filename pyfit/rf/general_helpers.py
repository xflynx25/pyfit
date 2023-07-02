import pandas as pd
from constants import DROPBOX_PATH
from os import makedirs
from Requests import proper_request
from datetime import datetime
import time
import ast
import sys, os
import json

def get_user_folder_from_user(user):
    return DROPBOX_PATH + user + '/'

def save_user_personality(user, personality):
    user_folder = get_user_folder_from_user(user)
    with open(user_folder + "personality.json", "w") as outfile:
        json.dump(personality, outfile)
    print('User Personality Saved')

# json auto changes the inner dict keys to be strings rather than integers, so we have to change back
def unpack_personality(filename):    
    personality = json.load(open(filename))
    for key in ['hesitancy_dict', 'min_delta_dict']:
        personality[key] = {int(ok):{int(k):v for k,v in inner_dict.items()} \
                for ok,inner_dict in personality[key].items()}
    return personality

def get_user_personality(user):
    user_folder = get_user_folder_from_user(user)
    return unpack_personality(user_folder + 'personality.json')
    
# GETTING UTC TIMES
def get_year_month_day_hour():
    t = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
    return [int(t[s:e]) for s,e in zip([0,5,8,11],[4,7,10,13])] 
    
def get_year_month_day_hour_minute_second():
    t = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
    timings = [int(t[s:e]) for s,e in zip([0,5,8,11,14,17],[4,7,10,13,16,19])] 
    print(timings)
    return timings

def get_year_month_day_string():
    return datetime.utcnow().strftime('%Y-%m-%d')

def get_hour_minute_second_string():
    return datetime.utcnow().strftime('%H:%M:%S')

def get_deadline_difference(deadline_date, deadline_time):
    current_date = [int(x) for x in get_year_month_day_string().split('-')]
    day_diff =  difference_in_days(current_date, deadline_date)
    if day_diff > 0:
        return day_diff
    else: #check time diff 
        current_time =  [int(x) for x in get_hour_minute_second_string().split(':')]
        if which_time_comes_first(current_time, deadline_time) == 0: 
            return 0
        else:
            return -1

#2019-08-10T11:30:00Z
def difference_in_days(start_day, end_day):
    root_year, root_month, root_day = start_day
    year, month, day = end_day

    def is_leap_year(year):
        if year % 400 == 0:
            return True
        elif year % 100 == 0:
            return False
        elif year % 4 == 0:
            return True
        else: 
            return False
    # Check Leap Year
    leap_year = is_leap_year(root_year + 1)
    if leap_year:
        feb = 29
    else:
        feb = 28

    # month dict
    days_in_month = {
        1: 31,
        2: feb,
        3: 31,
        4: 30,
        5: 31,
        6: 30,
        7: 31,
        8: 31,
        9: 31,
        10: 31,
        11: 30,
        12: 31,
    } 
    #convert dates into days based on day0
    difference = 0
    if year < root_year or (year==root_year and (month<root_month or (month==root_month and day < root_day))):
        raise Exception("start is after end")

    for this_year in range(root_year, year+1):
        if this_year not in (root_year, year): #go through full year 
            difference += 365 + [1 if is_leap_year(this_year) else 0][0]
        else:
            if this_year == year: # go until the stop date in the year 
                if year != root_year:
                    root_month, root_day = 1,0 #if we are coming from previous year 

                if month == root_month:
                    difference += day - root_day
                else:
                    difference += days_in_month[root_month] - root_day
                    for m in range(root_month + 1, month):
                        difference += days_in_month[m]
                    difference += day

            elif this_year == root_year: #go to end of year 
                difference += days_in_month[root_month] - root_day
                for m in range(root_month + 1, 13):
                    difference += days_in_month[m]
                #for m in range(1, month):
                #    difference += days_in_month[m]
                #difference += day
     
    return difference
    

def daystring_to_daylist_and_vv(original):
    if type(original) == str:
        root_year = int(original[:4])
        root_month = int(original[5:7])
        root_day = int(original[8:10])
        day = root_year, root_month, root_day
    elif type(original) == list:
        day = f'{original[0]}-{original[1]}-{original[2]}'
    else:
        raise Exception("not a proper date")
    return day

def get_current_day():
    url = 'https://fantasy.premierleague.com/api/fixtures/'
    response = proper_request("GET", url, headers=None)
    df_raw = pd.DataFrame(response.json())
    day0 = sorted(df_raw['kickoff_time'].dropna().unique())[0]
    day0 = daystring_to_daylist_and_vv(day0)
    daynow = get_year_month_day_string()
    daynow = daystring_to_daylist_and_vv(daynow)

    return difference_in_days(day0, daynow)

# h, m, s  # DOES NOT ACCOUNT FOR SAME EXACT TIME
def which_time_comes_first(a,b):
    if a[0] != b[0]: 
        first = b[0] < a[0] 
    else:
        if a[1] != b[1]: 
            first = b[1] < a[1] 
        else:
            if a[2] != b[2]:
                first = b[2] < a[2]
            else:
                return -1 

    return [1 if first else 0][0]


'''returns dataframe without any columns containing the str in combos'''
'''now with support for series'''
def drop_columns_containing(patterns, df):
    drop_indices = set()
    is_frame = 2 == len(df.shape)

    if is_frame:
        cols = df.columns
    else:
        cols = df.index
    for pattern in patterns:
        if is_frame:
            truth_table = df.columns.str.contains(pattern)
        else:
            truth_table = df.index.str.contains(pattern)
        for index in range(len(cols)):
            if truth_table[index]:
                drop_indices.add(index)
    
    drop_cols = []
    for index in drop_indices:
        drop_cols.append(cols[index])
    if is_frame:
        return df.drop(drop_cols, axis=1)
    else:
        return df.drop(drop_cols)

'''returns df with columns having some pattern in patterns'''
'''now with support for series'''
def get_columns_containing(patterns, df):
    is_frame = 2 == len(df.shape)

    if is_frame:
        cols = df.columns
    else:
        cols = df.index
    truth_table = [False for x in cols]
    for pattern in patterns:
        if is_frame:
            this_pattern = df.columns.str.contains(pattern)
        else: 
            this_pattern = df.index.str.contains(pattern)
        truth_table = this_pattern | truth_table

    if is_frame:
        return df.iloc[:, truth_table]
    else:
        return df.loc[truth_table]
    



# reads the information for a specific player
def get_meta_gwks_dfs(season, league, interval, rank):
    group = (int(rank)-1) // interval
    start, end = (interval * group) + 1, interval*(group + 1)
    meta_path = DROPBOX_PATH + f"Human_Seasons/{season}/{league}_{start}-{end}/meta.csv"
    gwks_path = DROPBOX_PATH + f"Human_Seasons/{season}/{league}_{start}-{end}/weekly.csv"
    meta_df = pd.read_csv(meta_path, index_col=0)
    meta_df = meta_df.loc[meta_df['rank']==rank]
    gwks_df = pd.read_csv(gwks_path, index_col=0)
    gwks_df = gwks_df.loc[gwks_df['rank']==rank]
    return meta_df, gwks_df

def get_data_df(century, season):
    hypenated_season = f'{century}{str(season)[:2]}-{str(season)[2:]}'
    training_path = DROPBOX_PATH + f"Our_Datasets/{hypenated_season}/Processed_Dataset_{season}.csv"
    df = pd.read_csv(training_path, index_col=0)
    data_df = df.loc[df['season']==season]
    return data_df

# will make the directory if needed 
# make sure we are reading into a dataframe not a series
def safe_read_csv(path):
    try:
        df = pd.read_csv(path, index_col=0)
    except:
        try:
            folder = ''
            for line in path.split('/')[:-1]:
                folder += line + '/'
            makedirs(folder)
        except:
            pass
        df = pd.DataFrame()
    return df

def safe_make_folder(folder):
    try: 
        makedirs(folder)
    except:
        pass

# will make the directory if needed 
def safe_to_csv(df, path):
    folder = ''
    for line in path.split('/')[:-1]:
        folder += line + '/'
    safe_make_folder(folder)
    df.to_csv(path)

# creates count dictionaries for a column of a dictionary
def get_counts(df, key):
    counts = {}
    for _, row in df.iterrows():
        counts[row[key]] = counts.get(row[key], 0) + 1
    return counts

# returns a list of the opponents for the upcoming week, in double gameweeks simply taking the opponennt will not suffice because we encode it as a number like 20*a + b
def get_opponents(opp):
    if opp == 0:
        return [0]
        
    opps = []
    while opp > 0:
        this_opp = ((opp-1) % 20)+1
        opps.append(this_opp)
        opp = (opp-1) // 20
    return opps

# For unwrapping listlikes that have been stored as strings in a database
def safer_eval(string):
    if string == 'set()':
        return set()
    else:
        return ast.literal_eval(string)



##### THESE HELPERS ARE FOR THE SPECIFIC FANTASY WEBSITE NOT FOR A GENERAL WEBSITE ####


# WILL THIS STILL WORK IF THEY ARE SOMEHOW CAPTCHA THE BOTS?? 
# MIGHT HAVE TO USE THE MANUAL SIGN IN METHOD 
USERNAME_ID = "loginUsername"
PASSWORD_ID = "loginPassword"
SUBMIT_BUTTON_CLASSES = ["ArrowButton-thcy3w-0","hHgZrv"]
def login_to_website(driver, login_url, email, password):
    driver.get(login_url)
    time.sleep(.25)
    driver.find_element_by_id(USERNAME_ID).send_keys(email)
    driver.find_element_by_id(PASSWORD_ID).send_keys(password)
    driver.find_element_by_xpath("//button[@type='submit']").submit()
    time.sleep(2)

def logout_from_website(driver, login_url):
    driver.get(login_url)
    time.sleep(.25)
    try:
        driver.find_element_by_link_text('Sign Out').click()
        time.sleep(1)
    except:
        driver.find_element_by_class_name("Dropdown__MoreButton-qc9lfl-0").click()
        time.sleep(2)
        driver.find_element_by_link_text('Sign Out').click()
        time.sleep(1)


def login_to_website_manual(driver, login_url, wait_time):
    driver.get(login_url)
    time.sleep(wait_time)

def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__