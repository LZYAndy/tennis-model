import pandas as pd
import numpy as np
import math
from tqdm import tqdm
from datetime import datetime, timedelta
import ELO_538 as elo
from Generate_ELO import *
import warnings
from sklearn.metrics import brier_score_loss, log_loss
import glob
import os
warnings.simplefilter("ignore")

# obtain players' elo rankings
history_df_atp = concat_data(1968, 2023, 'atp')
history_df_wta = concat_data(1968, 2023, 'wta')

# target_year = int(input('Target year: '))
n_bins = 5

# Calculate ECE score
def ece_score(py, y_test, n_bins=5):
    py = np.array(py)
    y_test = np.array(y_test)
    if y_test.ndim > 1:
        y_test = np.argmax(y_test, axis=1)
    py_index = np.argmax(py, axis=1)
    py_value = []
    for i in range(py.shape[0]):
        py_value.append(py[i, py_index[i]])
    py_value = np.array(py_value)
    acc, conf = np.zeros(n_bins), np.zeros(n_bins)
    Bm = np.zeros(n_bins)
    for m in range(n_bins):
        a, b = m / n_bins, (m + 1) / n_bins
        for i in range(py.shape[0]):
            if py_value[i] > a and py_value[i] <= b:
                Bm[m] += 1
                if py_index[i] == y_test[i]:
                    acc[m] += 1
                conf[m] += py_value[i]
        if Bm[m] != 0:
            acc[m] = acc[m] / Bm[m]
            conf[m] = conf[m] / Bm[m]
    # print(acc, conf)
    ece = 0
    for m in range(n_bins):
        ece += Bm[m] * np.abs((acc[m] - conf[m]))
    return ece / sum(Bm)

def ece_score_v2(py, y_test, n_bins=5):
    py = np.array(py)
    y_test = np.array(y_test)
    py_value = []
    acc, conf = np.zeros(n_bins), np.zeros(n_bins)
    Bm = np.zeros(n_bins)
    for m in range(n_bins):
        a, b = m / n_bins, (m + 1) / n_bins
        for i in range(py.shape[0]):
            if py[i] > a and py[i] <= b:
                Bm[m] += 1
                if y_test[i] == 1:
                    acc[m] += 1
                conf[m] += py[i]
        if Bm[m] != 0:
            acc[m] = acc[m] / Bm[m]
            conf[m] = conf[m] / Bm[m]
    ece = 0
    for m in range(n_bins):
        ece += Bm[m] * np.abs((acc[m] - conf[m]))
    return ece / sum(Bm)


# calculate elo ranking
def generate_elo_columns(arr, counts_538):
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


# Use which winning prob
choice = 'mdp'
elo_thresh = 100 #int(input('Similar Elo threshold: '))
prev_n = 2 #int(input('Previous n years: '))
min_num_prev_matches = 4 #int(input('Min num of prev matches: '))
strategy = 2 #int(input('Strategy (0: higher_than_opp, 1: higher_than_bookmaker, 2: kelly): '))

# Initialization
probs_mdp = []
probs_Avg = []
probs_elo = []
probs_bayes = []
y_test = []
bankroll_log = []
expected_win_probs = []
dates = []
match_ids = []

if strategy == 1:
    Delta = float(input('Delta: '))
elif strategy >= 2:
    kelly_multiplier = 0.1 #float(input('Kelly multiplier: '))

all_profit = 0
all_total_input = 0
all_num_of_bet = 0
all_num_of_win = 0
all_diff_mdp = []
all_diff_book = []
diff_mdp, diff_elo, diff_Avg = 0, 0, 0
bankroll = 10000

for target_year in range(2014, 2024):
    print()
    print('Betting on all matches in %d...' % target_year)
    # Prediction by MDP
    pred_mdp = pd.read_csv('../csv_files/MDP_prev_%d_years_complex6_%d-elo100-sim08.csv' % (prev_n, target_year))

    pred_mdp = pred_mdp.query('P1PrevNum>=@min_num_prev_matches and P2PrevNum>=@min_num_prev_matches')
    pred_mdp = pred_mdp.query('P1Cond==1 and P2Cond==1')

    betting_men = pd.read_excel('../betting/men/%d.xlsx' % target_year)
    betting_women = pd.read_excel('../betting/women/%d.xlsx' % target_year)
    betting = pd.concat([betting_men, betting_women], ignore_index=True)
    # betting = betting_women
    betting = betting.replace('Ramos-Vinolas A.', 'Ramos A.')
    # only keep matches that were completed
    betting = betting.query('Comment=="Completed"')
    betting = betting.query('AvgW<2 or AvgL<2')
    betting = betting.query('AvgW<=MaxW and AvgL<=MaxL')

    profit = 0
    total_input = 0
    num_of_win = 0
    num_of_bet = 0
    diff_mdp = []
    diff_book = []
    result = []
    
    for index, tmp_match in tqdm(pred_mdp.iterrows()):
        betonP1 = False
        betonP2 = False
        betAmount = 100
        modelChoice = None
        match = pred_mdp.query('date==@tmp_match.date and tournament_name==@tmp_match.tournament_name and P1Name==@tmp_match.P1Name and P2Name==@tmp_match.P2Name')
        if len(match) == 0:
            continue

        match = match.iloc[0]

        # find corresponding match
        P1Name = match.P1Name.split(' ')[-1]
        P2Name = match.P2Name.split(' ')[-1]
        market = betting.query('Date==@match.date and Winner.str.contains(@P1Name) and Loser.str.contains(@P2Name)')
        if len(market) != 1 or math.isnan(market.AvgW) or math.isnan(market.AvgL):
            continue
        market = market.iloc[0]
        P1Gms = np.nansum(market[['W1', 'W2', 'W3', 'W4', 'W5']].to_numpy())
        P2Gms = np.nansum(market[['L1', 'L2', 'L3', 'L4', 'L5']].to_numpy())

        # mdp prediction
        P1WinProb = match.P1WinProb
        P2WinProb = 1 - P1WinProb
        diff_mdp.append(abs(P1WinProb - P1Gms / (P1Gms + P2Gms)))

        # bookmaker prediction
        marketP1WinProb = 1 / market.AvgW
        marketP2WinProb = 1 / market.AvgL
        o = (marketP1WinProb + marketP2WinProb - 1) / 2
        marketOddsP1 = 1 / (marketP1WinProb - o)
        marketOddsP2 = 1 / (marketP2WinProb - o)
        diff_book.append(abs(marketP1WinProb - P1Gms / (P1Gms + P2Gms)))

        if index % 2 == 0:
            probs_mdp.append(max(0, min(1, P1WinProb)))
            probs_Avg.append(max(0, min(1, marketP1WinProb - o)))
            y_test.append(1)
        else:
            probs_mdp.append(max(0, min(1, P2WinProb)))
            probs_Avg.append(max(0, min(1, marketP2WinProb - o)))
            y_test.append(0)

        win_prob1 = P1WinProb
        win_prob2 = P2WinProb

        # Kelly Criteria
        if win_prob1 > 0.5:
            betonP1 = True
            odds = market.AvgW
            prob = win_prob1
        elif win_prob2 > 0.5:
            betonP2 = True
            odds = market.AvgL
            prob = win_prob2
        else:
            continue

        # adpt_kelly_multiplier = prob * kelly_multiplier
        frac =  ((odds - 1) * prob - (1 - prob)) / (odds - 1)

        if frac > 0:
            betAmount = bankroll * frac * kelly_multiplier
        else:
            continue

        # Update bankroll and profits
        winnings = 0
        if betonP1:
            winnings += (market.AvgW - 1) * betAmount
            num_of_win += 1
            num_of_bet += 1
            total_input += betAmount
        elif betonP2:
            winnings -= betAmount
            num_of_bet += 1
            total_input += betAmount
        bankroll += winnings
        profit += winnings
        bankroll_log.append(bankroll)
        dates.append(market.Date)

        # Run out of money
        if bankroll <= 10:
            print('Run out of money in %d bets!' % (all_total_input / 100))
            exit(0)

    print('Profits: %d, total input:%d, percentage: %f, total win: %d, num of bet: %d'%(profit, total_input, profit/total_input, num_of_win, num_of_bet))
    print('Bank Roll: %d'%bankroll)
    all_profit += profit
    all_total_input += total_input
    all_num_of_win += num_of_win
    all_num_of_bet += num_of_bet

    print('Win prob diff for MDP prediction: %.4f' % np.mean(diff_mdp))
    print('Win prob diff for bookmaker prediction: %.4f' % np.mean(diff_book))

    all_diff_mdp.extend(diff_mdp)
    all_diff_book.extend(diff_book)

    print('ECE score for MDP prediction: %.4f' % ece_score_v2(probs_mdp, y_test, n_bins))
    print('ECE score for bookmaker prediction: %.4f' % ece_score_v2(probs_Avg, y_test, n_bins))
    print(len(y_test))

print()
print('Overall average')
print('Profits: %f, total input:%f, percentage: %f, total win: %f, total bet: %d'%(all_profit, all_total_input, all_profit/all_total_input, all_num_of_win, all_num_of_bet))
print('Bank Roll: %d'%bankroll)
ROI = all_profit / 10000
print('ROI: %.2f, Annual ROI: %.2f' % (ROI * 100, (math.pow(1+ROI, 0.1)-1)*100))

print('Win prob diff for MDP prediction: %.4f' % np.mean(all_diff_mdp))
print('Win prob diff for bookmaker prediction: %.4f' % np.mean(all_diff_book))

print('ECE score for MDP prediction: %.4f' % ece_score_v2(probs_mdp, y_test, n_bins))
print('ECE score for bookmaker prediction: %.4f' % ece_score_v2(probs_Avg, y_test, n_bins))
print(len(probs_mdp), len(probs_Avg))