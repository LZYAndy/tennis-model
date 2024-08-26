import pandas as pd
import numpy as np
from tqdm import tqdm as tqdm
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
import ELO_538 as elo
from Generate_ELO import *
import math
import warnings

warnings.simplefilter("ignore")

# Load match info
matches = pd.read_csv('../matches_updated.csv')

# obtain players' elo rankings
history_df_atp = concat_data(1968, 2023, 'atp')
history_df_wta = concat_data(1968, 2023, 'wta')


# evaluate pcsp file
def eval_pcsp(dir, file_name):
    os.system('cp %s ../../PAT351/%s' % (os.path.join(dir, file_name), file_name))
    os.chdir('/home/user/zhaoyu/PAT351')
    os.system('mono PAT3.Console.exe -pcsp %s %s.txt' % (file_name, file_name[:-5]))
    with open('%s.txt' % file_name[:-5]) as f:
        lines = f.readlines()
    os.system('rm %s' % file_name)
    os.system('rm %s.txt' % file_name[:-5])
    os.chdir('/home/user/zhaoyu/TKDE/src')
    if len(lines) < 5 or '[' not in lines[3] or ']' not in lines[3]:
        print(lines)
        return -1, -1
    # print(lines[3])
    min_max_probs = [float(score) for score in lines[3].split('[')[1].split(']')[0].split(',')]
    return min_max_probs


# generate pcsp file
def generate_pcsp(params, date, ply1_name, ply2_name, ply1_hand, ply2_hand, ply_dim=8):
    ply1_hand = 'RH' if int(ply1_hand) == 1 else 'LH'
    ply2_hand = 'RH' if int(ply2_hand) == 1 else 'LH'
    VAR = '../pcsp_files_complex_6/var.txt'
    HAND = '../pcsp_files_complex_6/%s_%s.txt' % (ply1_hand, ply2_hand)
    dir = '../pcsp_files_complex_6/%s/' % (date[:4])
    file_name = '%s_%s_' % (ply1_hand, ply2_hand)
    file_name += '%s_%s_%s_complex_6-elo100-sim%02d_0.01_equal_serves.pcsp' % (date, ply1_name.replace(' ', '-'), ply2_name.replace(' ', '-'), ply_dim)
    print(file_name)
    # pcsp year folder have not been created
    if not os.path.isdir(dir):
        os.mkdir(dir)
    lines = []
    with open(VAR) as f:
        lines_1 = f.readlines()
    count = 0
    with open(HAND) as f:
        lines_2 = f.readlines()
        for i in range(len(lines_2)):
            if 'p:' in lines_2[i]:
                lines_2[i] = lines_2[i].replace('p:', '%d:' % params[count])
                count += 1
    lines = lines_1 + lines_2
    with open(os.path.join(dir, file_name), 'w') as f:
        for line in lines:
            f.write(line)
    return eval_pcsp(dir, file_name)


# obtain player info from matches
def obtain_ply_info(ply_name):
    ply_name = ply_name.replace('-', ' ').split(' ')
    if '' in ply_name:
        ply_name.remove('')
    ply_surname = ' '.join(ply_name[:-1])
    ply_given = ply_name[-1][0]
    P1 = matches.query('P1Name.str.contains(@ply_surname) and P1Name.str[0]==@ply_given')
    P2 = matches.query('P2Name.str.contains(@ply_surname) and P2Name.str[0]==@ply_given')
    if len(P1) > 0:
        return P1.iloc[0].P1Name, P1.iloc[0].P1Hand
    elif len(P2) > 0:
        return P2.iloc[0].P2Name, P2.iloc[0].P2Hand
    else:
        return None, None


# obtain parameters
def get_params(df, hand):
    # define player position
    volley = [5, 26]
    # specific df
    ply_position = []
    opp_position = []
    for index, row in df.iterrows():
        if row.prev_shot_approach_shot == 1 or row.shot in volley:  # or row.hit_at_depth == 1 or row.prev_shot in [9, 30]:
            ply_position.append(1)
        else:
            ply_position.append(0)
        if row.prev_shot in volley:  # or row.prev_shot_hit_at_depth == 1 or row.prev_prev_shot in [9, 30]:
            opp_position.append(1)
        else:
            opp_position.append(0)
    df['ply_position'] = ply_position
    df['opp_position'] = opp_position

    De_Serve = df.query('shot_type==1 and from_which_court==1')
    De_Serve_2nd = df.query('shot_type==2 and from_which_court==1')
    Ad_Serve = df.query('shot_type==1 and from_which_court==3')
    Ad_Serve_2nd = df.query('shot_type==2 and from_which_court==3')

    DeT_Return = df.query('shot_type==3 and prev_shot_from_which_court==1 and prev_shot_direction==6')
    DeB_Return = df.query('shot_type==3 and prev_shot_from_which_court==1 and prev_shot_direction==5')
    DeW_Return = df.query('shot_type==3 and prev_shot_from_which_court==1 and prev_shot_direction==4')
    AdT_Return = df.query('shot_type==3 and prev_shot_from_which_court==3 and prev_shot_direction==6')
    AdB_Return = df.query('shot_type==3 and prev_shot_from_which_court==3 and prev_shot_direction==5')
    AdW_Return = df.query('shot_type==3 and prev_shot_from_which_court==3 and prev_shot_direction==4')

    De_Stroke_at_base = df.query('shot_type==4 and from_which_court==1 and ply_position==0')
    De_Stroke_at_net = df.query('shot_type==4 and from_which_court==1 and ply_position==1')

    Mid_Stroke_at_base = df.query('shot_type==4 and from_which_court==2 and ply_position==0')
    Mid_Stroke_at_net = df.query('shot_type==4 and from_which_court==2 and ply_position==1')

    Ad_Stroke_at_base = df.query('shot_type==4 and from_which_court==3 and ply_position==0')
    Ad_Stroke_at_net = df.query('shot_type==4 and from_which_court==3 and ply_position==1')

    num_return_stroke = len(df.query('shot_type in [3, 4]'))
    num_approach = len(df.query('shot_type in [3, 4] and approach_shot==1'))
    num_serve = len(df.query('shot_type in [1, 2]'))
    num_serve_approach = len(df.query('shot_type in [1, 2] and approach_shot==1'))

    results = []
    # Serve
    for Serve in [De_Serve, De_Serve_2nd, Ad_Serve, Ad_Serve_2nd]:
        ServeT = Serve.query('direction==6')
        ServeB = Serve.query('direction==5')
        ServeW = Serve.query('direction==4')
        serves = [ServeT, ServeB, ServeW]
        serve_in = [len(x.query('shot_outcome==7')) for x in serves]
        serve_win = [len(Serve.query('shot_outcome in [1, 5, 6]'))]
        serve_err = [len(Serve.query('shot_outcome in [2, 3, 4]'))]
        results.append(serve_in + serve_win + serve_err)

    # Return
    directions = [[3, 1, 2], [1, 3, 2], [1, 3, 2], [1, 3, 2], [1, 3, 2], [3, 1, 2]]
    for i, Return in enumerate([DeT_Return, DeB_Return, DeW_Return, AdT_Return, AdB_Return, AdW_Return]):
        shots = [Return.query('to_which_court==@to_dir') for to_dir in directions[i]]
        return_in = [len(x.query('shot_outcome==7')) for x in shots]
        return_win = [len(Return.query('shot_outcome in [1, 5, 6]'))]
        return_err = [len(Return.query('shot_outcome in [2, 3, 4]'))]
        return_list = return_in + return_win + return_err
        results.append(return_list)

    # Rally
    directions = [[[1, 3, 2], [3, 1, 2]], [[3, 1, 2], [1, 3, 2]], [[3, 1, 2], [3, 1, 2]]]
    for i, Strokes in enumerate([[De_Stroke_at_base, De_Stroke_at_net],
                                 [Mid_Stroke_at_base, Mid_Stroke_at_net],
                                 [Ad_Stroke_at_base, Ad_Stroke_at_net]]):
        for j, Stroke in enumerate(Strokes):
            if hand == 'RH':
                Stroke_1 = Stroke.query('shot<=20')  # forehand
                Stroke_2 = Stroke.query('shot<=40 and shot>20')  # backhand
            else:
                Stroke_1 = Stroke.query('shot<=40 and shot>20')  # backhand
                Stroke_2 = Stroke.query('shot<=20')  # forehand
            if j == 0:
                shots_1 = [Stroke_1.query('to_which_court==@to_dir') for to_dir in directions[i][0]]
                shots_2 = [Stroke_2.query('to_which_court==@to_dir') for to_dir in directions[i][1]]
                shots = shots_1 + shots_2
                stroke_in = [len(x.query('shot_outcome==7')) for x in shots]
                stroke_win = [len(Stroke.query('shot_outcome in [1, 5, 6]'))]
                stroke_err = [len(Stroke.query('shot_outcome in [2, 3, 4]'))]
                stroke_list = stroke_in + stroke_win + stroke_err
                results.append(stroke_list)
            else:
                shots_1 = [Stroke_1.query('to_which_court==@to_dir') for to_dir in directions[i][0]]
                shots_2 = [Stroke_2.query('to_which_court==@to_dir') for to_dir in directions[i][1]]
                shots = shots_1 + shots_2
                stroke_in = [len(x.query('shot_outcome==7')) for x in shots]
                stroke_win = [len(Stroke.query('shot_outcome in [1, 5, 6]'))]
                stroke_err = [len(Stroke.query('shot_outcome in [2, 3, 4]'))]
                approach_stroke_list = stroke_in + stroke_win + stroke_err
                if sum(approach_stroke_list) > 0:
                    t = 1 
                    approach_stroke_list = [s * t + stroke_list[k] for k, s in enumerate(approach_stroke_list)]
                else:
                    approach_stroke_list = stroke_list
                results.append(approach_stroke_list)

    results.append([num_serve_approach, num_serve - num_serve_approach])
    results.append([num_approach, num_return_stroke - num_approach])
    return sum(results, []) #results


def generate_elo_columns(arr, counts_538):
    player_names = arr[:, :2].flatten()
    players_set = np.where(player_names != player_names, '', player_names).tolist()
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
    return players_elo


def obtain_similar_elo_players(ply_name, names, players_elo, thresh=100):
    ply_elo = 1500
    if ply_name in players_elo:
        ply_elo = players_elo[ply_name].value
    # thresh = max(thresh, ply_elo * 0.06)
    similar_elo_as_ply = []
    most_similar_ply = None
    min_elo_diff = 1500
    for tmp_name in names:
        if tmp_name not in players_elo:
            continue
        if abs(players_elo[tmp_name].value - ply_elo) < min_elo_diff:
            min_elo_diff = abs(players_elo[tmp_name].value - ply_elo)
            most_similar_ply = tmp_name
        if abs(players_elo[tmp_name].value - ply_elo) <= thresh:
            similar_elo_as_ply.append(tmp_name)
    return similar_elo_as_ply, ply_elo


def obtain_ply_chars(TLSE_date, players_elo, ply_name):
    volley = [5, 13, 15, 26, 34, 36]
    if ply_name not in players_elo:
        ply_elo = 1500 / 3000
    else:
        ply_elo = players_elo[ply_name].value / 3000
    TLSE_ply = TLSE_date.query('ply1_name==@ply_name')

    # Shot quality
    ## Serve quality
    ply_Serve = TLSE_ply.query('shot_type in [1, 2]')
    ply_Serve_win = ply_Serve.query('final_outcome==1')
    ply_Serve_rate = len(ply_Serve_win) / len(ply_Serve)
    ## Return quality
    ply_Return = TLSE_ply.query('shot_type==3')
    ply_Return_win = ply_Return.query('final_outcome==1')
    ply_Return_rate = len(ply_Return_win) / len(ply_Return)
    ## Forehand quality
    ply_FH_Stroke = TLSE_ply.query('shot_type==4 and shot<=20')
    ply_FH_Stroke_win = ply_FH_Stroke.query('final_outcome==1')
    ply_FH_Stroke_rate = len(ply_FH_Stroke_win) / len(ply_FH_Stroke)
    ## Backhand quality
    ply_BH_Stroke = TLSE_ply.query('shot_type==4 and shot<=40 and shot>20')
    ply_BH_Stroke_win = ply_BH_Stroke.query('final_outcome==1')
    ply_BH_Stroke_rate = len(ply_BH_Stroke_win) / len(ply_BH_Stroke)

    # Player style
    ## Big Server
    ply_Serve_ace = ply_Serve.query('shot_outcome in [1, 5, 6]')
    ply_BigServe_rate = len(ply_Serve_ace) / len(ply_Serve)
    ## Serve and Volley
    ply_Serve_volley = ply_Serve.query('approach_shot == 1')
    ply_ServeVolley_rate = len(ply_Serve_volley) / len(ply_Serve)
    ## All-court / baseliner
    ply_At_Net = TLSE_ply.query('shot_type in [3, 4] and (approach_shot == 1 or shot in @volley or hit_at_depth == 1)')
    ply_Net_rate = len(ply_At_Net) / (len(ply_Return) + len(ply_FH_Stroke) + len(ply_BH_Stroke))

    ply_chars = np.array([ply_elo, ply_Serve_rate, ply_Return_rate, ply_FH_Stroke_rate, ply_BH_Stroke_rate,
                          ply_BigServe_rate, ply_ServeVolley_rate, ply_Net_rate])
    return ply_chars


def generate_transition_probs(TLSE, match, prev_date):
    date = match.Date.strftime('%Y-%m-%d')
    surface = match.Surface
    prev_n_days = (match.Date - timedelta(days=7)).strftime('%Y-%m-%d')
    ply1_name, ply1_hand = obtain_ply_info(match.Winner)
    ply2_name, ply2_hand = obtain_ply_info(match.Loser)

    # do not have players
    if ply1_name is None or ply2_name is None:
        return None, None, None, None

    is_ply1_win = 1
    gender = 'F' if math.isnan(match.ATP) else 'M'

    # check whether match date is correct
    check_match_date = TLSE.query('date>=@prev_n_days and date<@date and ply1_name==@ply1_name and ply2_name==@ply2_name')
    if len(check_match_date) > 0:
        date = check_match_date.iloc[0].date

    # fine related historical matches for ply1
    TLSE_date = TLSE.query('date>=@prev_date and date<@date')
    TLSE_ply1 = TLSE_date.query('ply1_name==@ply1_name and ply2_hand==@ply2_hand')
    if len(TLSE_ply1) == 0:
        return None, None, None, None
    
    # fine related historical matches for ply1
    TLSE_ply2 = TLSE_date.query('ply1_name==@ply2_name and ply2_hand==@ply1_hand')
    if len(TLSE_ply2) == 0:
        return None, None, None, None
    ply1_prev_n, ply2_prev_n = [], []
    ply1_cond, ply2_cond = -1, -1

    # calculate players' elo rankings
    history_df = history_df_atp if gender == 'M' else history_df_wta
    history_df = history_df.query('tny_date<@date')
    players_elo = generate_elo_columns(history_df[['w_name', 'l_name', 'is_gs']].to_numpy(), True)

    # players with similar elo as ply1
    similar_elo_as_ply1, ply1_elo = obtain_similar_elo_players(ply1_name, TLSE_ply2.ply2_name.unique(), players_elo)
    if len(similar_elo_as_ply1) == 0:
        return None, None, None, None

    # players with similar elo as ply1
    similar_elo_as_ply2, ply2_elo = obtain_similar_elo_players(ply2_name, TLSE_ply1.ply2_name.unique(), players_elo)
    if len(similar_elo_as_ply2) == 0:
        return None, None, None, None

    # obtain target players' characters
    target_ply_chars = []
    for ply_name in [ply1_name, ply2_name]:
        target_ply_chars.append(obtain_ply_chars(TLSE_date, players_elo, ply_name))
    target_ply_chars = np.array(target_ply_chars)

    # similarity scores between ply1's opponents and ply2
    similarity = []
    ply_dim = 0
    for ply_name in similar_elo_as_ply2:
        ply_chars = obtain_ply_chars(TLSE_date, players_elo, ply_name)
        if ply_dim == 0:
            ply_dim = len(ply_chars)
        dist = np.linalg.norm(target_ply_chars[1] - ply_chars)
        dist = dist if dist > 0.01 else 0.01
        similarity.append(1 / dist)
    similarity = [x / sum(similarity) for x in similarity]  # normalize

    # get ply1 MDP params
    ply1_params = None
    num_ply1_prev_n = 0  # num of prev matches
    for i, ply_name in enumerate(similar_elo_as_ply2):
        ply1_prev_n = TLSE_ply1.query('ply2_name==@ply_name')
        num_ply1_prev_n += len(ply1_prev_n.date.unique())
        params = np.array([int(x * similarity[i] * 100) for x in get_params(ply1_prev_n, ply1_hand)])
        if ply1_params is None:
            ply1_params = params
            continue
        ply1_params += params
    if num_ply1_prev_n <= 3:  # make sure there are at least 4 related historical matches
        return None, None, None, None

    # similarity scores between ply2's opponents and ply1
    similarity = []
    for ply_name in similar_elo_as_ply1:
        ply_chars = obtain_ply_chars(TLSE_date, players_elo, ply_name)
        dist = np.linalg.norm(target_ply_chars[0] - ply_chars)
        dist = dist if dist > 0.01 else 0.01
        similarity.append(1 / dist)
    similarity = [x / sum(similarity) for x in similarity]  # normalize

    # get ply2 MDP params
    ply2_params = None
    num_ply2_prev_n = 0  # num of prev matches
    for i, ply_name in enumerate(similar_elo_as_ply1):
        ply2_prev_n = TLSE_ply2.query('ply2_name==@ply_name')
        num_ply2_prev_n += len(ply2_prev_n.date.unique())
        params = np.array([int(x * similarity[i] * 100) for x in get_params(ply2_prev_n, ply2_hand)])
        if ply2_params is None:
            ply2_params = params
            continue
        ply2_params += params
    if num_ply2_prev_n <= 3:  # make sure there are at least 4 related historical matches
        return None, None, None, None

    # sample
    sample = [match.Date.strftime('%Y-%m-%d'), gender, match.Tournament, match.Surface, ply1_name, ply2_name,
              int(ply1_hand == 'RH'), int(ply2_hand == 'RH')]
    sample = sample + list(ply1_params) + list(ply2_params) + [is_ply1_win, num_ply1_prev_n, num_ply2_prev_n, 1, 1]

    return np.array(sample), ply1_elo, ply2_elo, ply_dim


def read_csv_file(file, save=False):
    match = pd.read_csv(file)
    return match


start_year = int(input('Start year: '))
end_year = int(input('End year: '))
for target_year in range(start_year, end_year):
    print('Target year: %d' % (target_year))

    # obtain shot-by-shot data for 3 years
    TLSE = []
    files = glob.glob('../tennisabstract-csv-v4/%d*.csv' % (target_year))
    files += glob.glob('../tennisabstract-csv-v4/%d*.csv' % (target_year - 1))
    files += glob.glob('../tennisabstract-csv-v4/%d*.csv' % (target_year - 2))
    for file in tqdm(files):
        TLSE.append(read_csv_file(file, save=True))
    TLSE = pd.concat(TLSE, ignore_index=True)

    # Load betting info
    betting_men = pd.read_excel('../betting/men/%d.xlsx' % (target_year))
    betting_women = pd.read_excel('../betting/women/%d.xlsx' % (target_year))
    betting = pd.concat([betting_men, betting_women], ignore_index=True)
    betting = betting.replace('Ramos-Vinolas A.', 'Ramos A.')

    # all matches
    start_date = '%d-12-30' % (target_year - 1)
    end_date = '%d-12-31' % target_year
    total_matches = len(betting.query('Date>=@start_date and Date<=@end_date'))
    result = []

    for index, match in tqdm(betting.query('Date>=@start_date and Date<=@end_date').iterrows()):
        prev_2_years = (match.Date - relativedelta(years=2)).strftime('%Y-%m-%d')
        # print(TLSE)
        sample, ply1_elo, ply2_elo, ply_dim = generate_transition_probs(TLSE, match, prev_2_years)
        if sample is not None:
            print('Total matches in %d: %d' % (target_year, total_matches))
            params = sample[8:-5].astype(int)
            ply1_prob0, ply1_prob1 = generate_pcsp(params, sample[0], sample[4], sample[5], sample[6], sample[7], ply_dim=ply_dim)
            ply2_prob0, ply2_prob1 = 1 - ply1_prob0, 1 - ply1_prob1
            ply1_prob0 = round(ply1_prob0, 4)
            ply1_prob1 = round(ply1_prob1, 4)
            ply2_prob0 = round(ply2_prob0, 4)
            ply2_prob1 = round(ply2_prob1, 4)
            if ply1_prob0 < 0:
                continue
            print(sample[-4], sample[-3])
            result.append([sample[0], sample[1], sample[2], sample[3], sample[4], sample[5], sample[6], sample[7],
                           ply1_prob0, ply1_prob1, ply2_prob1, ply2_prob0, sample[-5], sample[-4], sample[-3],
                           sample[-2], sample[-1], round((ply1_prob0 + ply1_prob1) / 2, 4),
                           round((ply2_prob0 + ply2_prob1) / 2, 4)])
            out_csv = pd.DataFrame(result, columns=['date', 'gender', 'tournament_name', 'surface', 'P1Name', 'P2Name',
                                                    'P1Hand', 'P2Hand', 'P1WinProb_min', 'P1WinProb_max',
                                                    'P2WinProb_min', 'P2WinProb_max', 'P1Win', 'P1PrevNum',
                                                    'P2PrevNum', 'P1Cond', 'P2Cond', 'P1WinProb', 'P2WinProb'])
            print(round((ply1_prob0 + ply1_prob1) / 2, 4), round((ply2_prob0 + ply2_prob1) / 2, 4))
            out_csv.to_csv('../csv_files/MDP_prev_2_years_complex6_%d-elo100-sim%02d_0.01_equal_serves.csv' % (target_year, ply_dim), index=False)
