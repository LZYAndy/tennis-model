import datetime
import numpy as np
import os
import pandas as pd
import ELO_538 as elo
from tqdm import tqdm
import re
import glob

RET_STRINGS = (
    'ABN', 'DEF', 'In Progress', 'RET', 'W/O', ' RET', ' W/O', 'nan', 'walkover'
)
ABD_STRINGS = (
    'abandoned', 'ABN', 'ABD', 'DEF', 'def', 'unfinished', 'Walkover'
)

def normalize_name(s, tour='atp'):
    if tour=='atp':
        s = s.replace('-',' ')
        s = s.replace('Stanislas','Stan').replace('Stan','Stanislas')
        s = s.replace('Alexandre','Alexander')
        s = s.replace('Federico Delbonis','Federico Del').replace('Federico Del','Federico Delbonis')
        s = s.replace('Mello','Melo')
        s = s.replace('Cedric','Cedrik')
        s = s.replace('Bernakis','Berankis')
        s = s.replace('Hansescu','Hanescu')
        s = s.replace('Teimuraz','Teymuraz')
        s = s.replace('Vikor','Viktor')
        s = s.rstrip()
        s = s.replace('Alex Jr.','Alex Bogomolov')
        s = s.title()
        sep = s.split(' ')
        return ' '.join(sep[:2]) if len(sep)>2 else s
    else:
        return s


def format_match_df(df,tour,ret_strings=[],abd_strings=[]):
    cols = [u'tourney_id', u'tourney_name', u'surface', u'draw_size', u'tourney_date',
            u'match_num', u'winner_name', u'loser_name', u'score', u'best_of', u'w_svpt',
            u'w_1stWon', u'w_2ndWon', u'l_svpt', u'l_1stWon', u'l_2ndWon']
    df = df[cols]
    df = df.rename(columns={'winner_name':'w_name','loser_name':'l_name','tourney_id':'tny_id',\
                            'tourney_name':'tny_name','tourney_date':'tny_date'})

    df['w_name'] = [normalize_name(x,tour) for x in df['w_name']]
    df['l_name'] = [normalize_name(x,tour) for x in df['l_name']]
    df['tny_name'] = ['Davis Cup' if 'Davis Cup' in s else s for s in df['tny_name']]
    df['tny_name'] = [s.replace('Australian Chps.','Australian Open').replace('Australian Open-2',\
                'Australian Open').replace('U.S. National Chps.','US Open') for s in df['tny_name']]
    df['is_gs'] = (df['tny_name'] == 'Australian Open') | (df['tny_name'] == 'Roland Garros') |\
                  (df['tny_name'] == 'Wimbledon')       | (df['tny_name'] == 'US Open')

    # format dates
    df['tny_date'] = [datetime.datetime.strptime(str(x), "%Y%m%d").date() for x in df['tny_date']]
    df['match_year'] = [x.year for x in df['tny_date']]
    df['match_month'] = [x.month for x in df['tny_date']]
    df['match_year'] = df['match_year'] + (df['match_month'] == 12) # correct december start dates
    df['match_month'] = [1 if month==12 else month for month in df['match_month']] # to following year
    df['score'] = [re.sub(r"[\(\[].*?[\)\]]", "", str(s)) for s in df['score']] # str(s) fixes any nans
    df['score'] = ['RET' if 'RET' in s else s for s in df['score']]
    df['w_swon'], df['l_swon'] = df['w_1stWon']+df['w_2ndWon'], df['l_1stWon']+df['l_2ndWon']
    df['w_rwon'], df['l_rwon'] = df['l_svpt']-df['l_swon'], df['w_svpt']-df['w_swon']
    df['w_rpt'], df['l_rpt'] = df['l_svpt'], df['w_svpt']
    df.drop(['w_1stWon','w_2ndWon','l_1stWon','l_2ndWon'], axis=1, inplace=True)

    # remove matches involving a retirement
    abd_d, ret_d = set(abd_strings), set(ret_strings)
    df['score'] = ['ABN' if score.split(' ')[-1] in abd_d else score for score in df['score']]
    df['score'] = ['RET' if score in ret_d else score for score in df['score']]
    return df.loc[(df['score'] != 'ABN') & (df['score'] != 'RET')].reset_index(drop=True)


def concat_data(start_y, end_y, tour):
    match_year_list = []
    for i in range(start_y, end_y+1):
        f_name = '../match_data/{}_matches_{}.csv'.format(tour, i)
        df = pd.read_csv(f_name)
        format_df = format_match_df(df, tour, RET_STRINGS, ABD_STRINGS)
        format_df.to_csv('../match_data_formatted/{}_matches_{}.csv'.format(tour, i), index=False)
        try:
            match_year_list.append(pd.read_csv('../match_data_formatted/{}_matches_{}.csv'.format(tour, i)))
        except:
            print('could not find file for year: ', i)
    full_match_df = pd.concat(match_year_list, ignore_index = True)
    return full_match_df.sort_values(by=['tny_date','tny_name','match_num'], ascending=True).reset_index(drop=True)


def generate_elo_dict(arr, counts_538):
    player_names = arr[:, :2].flatten()
    players_set = np.where(player_names!=player_names, '', player_names).tolist()
    players_elo = {}
    for player in players_set:
        players_elo[player] = elo.Rating()

    match_elos = np.zeros([arr.shape[0], 2])
    elo_obj = elo.Elo_Rater()

    # update player elo from every recorded match
    for i in range(arr.shape[0]):
        w_name, l_name = arr[i][:2]
        if w_name != w_name or l_name != l_name:
            match_elos[i] = np.nan, np.nan
            continue
        match_elos[i] = players_elo[w_name].value, players_elo[l_name].value
        elo_obj.rate_1vs1(players_elo[w_name], players_elo[l_name], arr[i][2], counts_538)

    return players_elo


def generate_elo_columns(arr, counts_538):
    player_names = arr[:, :2].flatten()
    players_set = np.where(player_names!=player_names, '', player_names).tolist()
    players_elo = {}
    for player in players_set:
        # print('player: ', player)
        players_elo[player] = elo.Rating()

    match_elos = np.zeros([arr.shape[0], 2])
    elo_obj = elo.Elo_Rater()

    # update player elo from every recorded match
    for i in range(arr.shape[0]):
        w_name, l_name = arr[i][:2]
        if w_name != w_name or l_name != l_name:
            match_elos[i] = np.nan, np.nan
            continue

        match_elos[i] = players_elo[w_name].value, players_elo[l_name].value
        elo_obj.rate_1vs1(players_elo[w_name], players_elo[l_name], arr[i][2], counts_538)

    return match_elos[:,0], match_elos[:,1]


