import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from datetime import date,datetime
import dateutil
import click
import numpy as np
from odi.feature_engg import util as cricutil

from ipl.data_loader import data_loader as dl


PREPROCESS_DATA_LOACATION = 'data_ipl'+os.sep+'preprocess'


def get_latest_rank_file(rank_type,ref_date = None):

    rank_type_prefix = {
        'country':'country_rank_',
        'batsman':'batsman_rank_',
        'bowler': 'bowler_rank_'
    }

    country_list_files = [f for f in os.listdir(PREPROCESS_DATA_LOACATION) if f.startswith(rank_type_prefix[rank_type])]
    if ref_date is None:
        today = date.today()
        ref_date = datetime(year=today.year, month=today.month, day=today.day)
    date_list = list()
    #print('=====',ref_date)
    #print('=====', country_list_files)

    for file in country_list_files:
        try:
            date_str = file.split(rank_type_prefix[rank_type])[1].split('.')[0]
            file_date = datetime.strptime(date_str, '%Y-%m-%d')
            #print('===file date====',file_date)
            if file_date <= ref_date:
                date_list.append(file_date)
        except Exception as ex:
            print(ex)

    if len(date_list)==0:
        return None
    else:
        date_list.sort()
        latest = date_list[-1].date()
        #print('==latest=',latest)
        return PREPROCESS_DATA_LOACATION+os.sep+rank_type_prefix[rank_type]+str(latest)+'.csv'


# def get_latest_country_rank_file(ref_date = None):
#     country_list_files = [f for f in os.listdir(PREPROCESS_DATA_LOACATION) if f.startswith('country_rank_')]
#     if ref_date is None:
#         today = date.today()
#         ref_date = datetime(year=today.year, month=today.month, day=today.day)
#     date_list = list()
#
#     for file in country_list_files:
#         try:
#             date_str = file.split('country_rank_')[1].split('.')[0]
#             file_date = datetime.strptime(date_str, '%Y-%m-%d')
#             if file_date <= ref_date:
#                 date_list.append(file_date)
#         except Exception as ex:
#             print(ex)
#
#     if len(date_list)==0:
#         return None
#     else:
#         date_list.sort()
#         latest = date_list[-1].date()
#         return PREPROCESS_DATA_LOACATION+os.sep+'country_rank_'+str(latest)+'.csv'


def create_batsman_rank_for_date(performance_cutoff_date_start, performance_cutoff_date_end, match_list_df):
    if not os.path.isdir(PREPROCESS_DATA_LOACATION):
        os.makedirs(PREPROCESS_DATA_LOACATION)

    scaler = MinMaxScaler()

    batsman_performance_list = []

    # print(selected_country)
    games = match_list_df[(match_list_df['date'] >= performance_cutoff_date_start)
                                  & (match_list_df['date'] <= performance_cutoff_date_end)
                            ]
    summary_stats_df = pd.read_csv(dl.CSV_LOAD_LOCATION + os.sep + 'match_stats.csv')
    #summary_stats_df = summary_stats_df.merge(games[['match_id']],how='inner',on='match_id')
    match_id_list = list(games['match_id'])
    match_stat_list = []
    for match_id in match_id_list:

        match_df = pd.read_csv(dl.CSV_LOAD_LOCATION+ os.sep + str(match_id) + '.csv')

        match_stat_list.append(match_df)

    match_stat_df = pd.concat(match_stat_list)
    match_stat_df.fillna('NA', inplace=True)

    match_stat_df = match_stat_df.merge(games, how='inner', on='match_id')
    batsman_list = list(match_stat_df['batsman'].unique())

    for selected_batsman in tqdm(batsman_list):
        # print(selected_batsman)

        batsman_df = match_stat_df[match_stat_df['batsman'] == selected_batsman]
        team = match_stat_df[match_stat_df['batsman'] == selected_batsman]['team'].values[0].strip()

        total_runs = batsman_df['scored_runs'].sum()
        run_rate = batsman_df['scored_runs'].sum() / \
                   match_stat_df[match_stat_df['batsman'] == selected_batsman].shape[0]

        # opponent_mean

        # batsman_df.rename(columns={'opponent': 'country'}, inplace=True)
        matches_played = len(list(batsman_df['match_id'].unique()))
        player_of_the_match = games[games['player_of_match'] == selected_batsman].shape[0]

        # winning contribution(effectiveness)-% of winning score
        #country_win_list = list(games[games['winner'] == selected_country]['match_id'])
        country_win_list = list(games[games['winner'] == team]['match_id'].unique())
        winning_match_df = match_stat_df[match_stat_df['match_id'].isin(country_win_list)]
        # winning_contribution = winning_match_df[winning_match_df['batsman'] == selected_batsman][
        #                            'scored_runs'].sum() / \
        #                        winning_match_df[winning_match_df['team'] == selected_country]['scored_runs'].sum()
        # calculate winning contribution
        batsman_runs_in_winning_matches = winning_match_df[winning_match_df['batsman'] == selected_batsman]['scored_runs'].sum()
        summary_stats_winning_matches_df = summary_stats_df.merge(winning_match_df[['match_id']],how='inner',on='match_id')
        team_run_in_winning_matches = summary_stats_winning_matches_df[summary_stats_winning_matches_df['team_statistics']==team]['total_run'].sum()
        winning_contribution = batsman_runs_in_winning_matches/team_run_in_winning_matches

        # run_rate_effectiveness
        country_run_rate = team_run_in_winning_matches / summary_stats_winning_matches_df['match_id'].nunique()
        batsman_run_rate = winning_match_df[winning_match_df['batsman'] == selected_batsman]['scored_runs'].sum() / \
                           winning_match_df['match_id'].nunique()

        run_rate_effectiveness = batsman_run_rate / country_run_rate


        average_score = batsman_df.groupby(['match_id'])['scored_runs'].sum().reset_index()['scored_runs'].mean()

        batsman_dict = {
            'batsman': selected_batsman,
            'total_runs': total_runs,
            'run_rate': run_rate,
            'average_score': average_score,
            'matches_played':matches_played,
            'player_of_the_match': player_of_the_match,
            'winning_contribution': winning_contribution,
            'run_rate_effectiveness': run_rate_effectiveness,
        }

        batsman_performance_list.append(batsman_dict)

    batsman_performance_df = pd.DataFrame(batsman_performance_list)
    batsman_performance_df.fillna(0, inplace=True)
    batsman_performance_df['batsman_score'] = scaler.fit_transform(
        batsman_performance_df.drop(columns=['batsman'])).sum(axis=1)
    batsman_performance_df.sort_values('batsman_score', ascending=False, inplace=True)
    batsman_performance_df.to_csv(PREPROCESS_DATA_LOACATION+os.sep+'batsman_rank_' + str(performance_cutoff_date_end.date()) + '.csv', index=False)


def create_batsman_rank(year_list,no_of_years=1):
    custom_date_parser = lambda x: datetime.strptime(x, "%Y-%m-%d")
    match_list_df = pd.read_csv(dl.CSV_LOAD_LOCATION+os.sep+'match_list.csv',
                                parse_dates=['date'],
                                date_parser=custom_date_parser)

    if year_list is None or len(year_list)==0:
        today = date.today()
        performance_cutoff_date_end = datetime(year=today.year, month=today.month, day=today.day)
        a_year = dateutil.relativedelta.relativedelta(years=no_of_years)
        performance_cutoff_date_start = performance_cutoff_date_end - a_year

        create_batsman_rank_for_date(performance_cutoff_date_start, performance_cutoff_date_end, match_list_df)

    else:
        for year in tqdm(year_list):
            performance_cutoff_date_start = datetime.strptime(year + '-01-01', '%Y-%m-%d')
            performance_cutoff_date_end = datetime.strptime(year + '-12-31', '%Y-%m-%d')
            create_batsman_rank_for_date(performance_cutoff_date_start, performance_cutoff_date_end, match_list_df)


def create_bowler_rank_for_date(performance_cutoff_date_start, performance_cutoff_date_end, match_list_df):
    if not os.path.isdir(PREPROCESS_DATA_LOACATION):
        os.makedirs(PREPROCESS_DATA_LOACATION)

    scaler = MinMaxScaler()

    bowler_performance_list = []

    games = match_list_df[(match_list_df['date'] >= performance_cutoff_date_start)
                          & (match_list_df['date'] <= performance_cutoff_date_end)
                          ]
    summary_stats_df = pd.read_csv(dl.CSV_LOAD_LOCATION + os.sep + 'match_stats.csv')
    # summary_stats_df = summary_stats_df.merge(games[['match_id']],how='inner',on='match_id')
    match_id_list = list(games['match_id'])
    match_stat_list = []
    for match_id in match_id_list:
        match_df = pd.read_csv(dl.CSV_LOAD_LOCATION + os.sep + str(match_id) + '.csv')

        match_stat_list.append(match_df)

    match_stat_df = pd.concat(match_stat_list)
    match_stat_df.fillna('NA', inplace=True)

    match_stat_df = match_stat_df.merge(games, how='inner', on='match_id')

    bowler_list = list(match_stat_df['bowler'].unique())



    for selected_bowler in tqdm(bowler_list):
        # print(selected_batsman)

        bowler_df = match_stat_df[match_stat_df['bowler'] == selected_bowler]

        team = match_stat_df[match_stat_df['bowler'] == selected_bowler]['opponent'].values[0].strip()

        total_runs = bowler_df['total'].sum()
        run_rate = total_runs / bowler_df.shape[0]
        negative_rate = -run_rate

        # no_of_wickets,wicket_rate,wicket_per_runs
        no_of_wickets = bowler_df['wicket'].sum() - bowler_df[bowler_df['wicket_type'] == 'run out'].shape[0]
        wickets_per_match = no_of_wickets / len(list(bowler_df['match_id'].unique()))
        wickets_per_run = no_of_wickets / total_runs


        # opponent_mean

        matches_played = len(list(bowler_df['match_id'].unique()))

        # winning contribution(effectiveness)-% of wickets taken in winning matches
        country_win_list = list(games[games['winner'] == team]['match_id'].unique())
        winning_match_df = match_stat_df[match_stat_df['match_id'].isin(country_win_list)]

        if winning_match_df['wicket'].sum() != 0:
            winning_contribution = winning_match_df[winning_match_df['bowler'] == selected_bowler]['wicket'].sum() / \
                                   winning_match_df['wicket'].sum()
        else:
            winning_contribution = 0

        # winning_wicket_per_run rate contribution
        # winning wicket_per_match contirbution

        team_wickets_per_run = winning_match_df[winning_match_df['opponent'] == team]['wicket'].sum() / \
                               winning_match_df[winning_match_df['opponent'] == team]['total'].sum()
        bowler_wicket_per_run = winning_match_df[winning_match_df['bowler'] == selected_bowler]['wicket'].sum() / \
                                winning_match_df[winning_match_df['bowler'] == selected_bowler]['total'].sum()
        winning_wicket_per_run_rate_contribution = bowler_wicket_per_run / team_wickets_per_run

        team_wicket_per_match = winning_match_df[winning_match_df['opponent'] == team]['wicket'].sum() / \
                                winning_match_df['match_id'].nunique()
        bowler_wicket_per_match = winning_match_df[winning_match_df['bowler'] == selected_bowler]['wicket'].sum() / \
                                  winning_match_df[winning_match_df['bowler'] == selected_bowler][
                                      'match_id'].nunique()
        winning_wicket_per_match_contribution = bowler_wicket_per_match / team_wicket_per_match

        no_of_wins = winning_match_df[winning_match_df['bowler'] == selected_bowler]['match_id'].nunique()
        # consistency
        # consistency = 1/match_stat_df[match_stat_df['bowler']==selected_bowler].groupby(['match_id'])['wicket'].sum().reset_index()['wicket'].std()

        bowler_dict = {
            'bowler': selected_bowler,
            'negative_rate': negative_rate,
            'no_of_wickets': no_of_wickets,
            'wickets_per_match': wickets_per_match,
            'wickets_per_run': wickets_per_run,
            'no_of_wins': no_of_wins,
            'winning_contribution': winning_contribution,
            'winning_wicket_rate_contribution': winning_wicket_per_match_contribution,

        }

        bowler_performance_list.append(bowler_dict)

    bowler_performance_df = pd.DataFrame(bowler_performance_list)
    bowler_performance_df.fillna(0, inplace=True)
    bowler_performance_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    bowler_performance_df.dropna(inplace=True)
    bowler_performance_df['bowler_score'] = scaler.fit_transform(
        bowler_performance_df.drop(columns=['bowler'])).sum(axis=1)
    bowler_performance_df.sort_values('bowler_score', ascending=False, inplace=True)
    bowler_performance_df.to_csv(PREPROCESS_DATA_LOACATION+os.sep+'bowler_rank_' + str(performance_cutoff_date_end.date()) + '.csv', index=False)

def create_bowler_rank(year_list,no_of_years=1):
    custom_date_parser = lambda x: datetime.strptime(x, "%Y-%m-%d")
    match_list_df = pd.read_csv(dl.CSV_LOAD_LOCATION+os.sep+'match_list.csv',
                                parse_dates=['date'],
                                date_parser=custom_date_parser)

    if year_list is None or len(year_list)==0:
        today = date.today()
        performance_cutoff_date_end = datetime(year=today.year, month=today.month, day=today.day)
        a_year = dateutil.relativedelta.relativedelta(years=no_of_years)
        performance_cutoff_date_start = performance_cutoff_date_end - a_year

        create_bowler_rank_for_date(performance_cutoff_date_start, performance_cutoff_date_end, match_list_df)

    else:
        for year in tqdm(year_list):
            performance_cutoff_date_start = datetime.strptime(year + '-01-01', '%Y-%m-%d')
            performance_cutoff_date_end = datetime.strptime(year + '-12-31', '%Y-%m-%d')
            create_bowler_rank_for_date(performance_cutoff_date_start, performance_cutoff_date_end, match_list_df)


@click.group()
def rank():
    pass

@rank.command()
@click.option('--year_list', multiple=True, help='list of years.')
@click.option('--no_of_years', type=int, default=1,
              help='applicable if year list not provided. How many previous years to consider')
def all(year_list,no_of_years):

    year_list = list(year_list)
    # create_country_rank(year_list,no_of_years=no_of_years)
    create_batsman_rank(year_list,no_of_years=no_of_years)
    create_bowler_rank(year_list,no_of_years=no_of_years)

@rank.command()
@click.option('--year_list', multiple=True, help='list of years.')
@click.option('--no_of_years', type=int, default=1,
              help='applicable if year list not provided. How many previous years to consider')
def batsman(year_list,no_of_years):

    year_list = list(year_list)
    #create_country_rank(year_list,no_of_years=no_of_years)
    create_batsman_rank(year_list,no_of_years=no_of_years)

@rank.command()
@click.option('--year_list', multiple=True, help='list of years.')
@click.option('--no_of_years', type=int, default=1,
              help='applicable if year list not provided. How many previous years to consider')
def bowler(year_list,no_of_years):

    year_list = list(year_list)
    #create_country_rank(year_list,no_of_years=no_of_years)
    create_bowler_rank(year_list,no_of_years=no_of_years)

@rank.command()
@click.option('--year_list', multiple=True, help='list of years.')
@click.option('--no_of_years', type=int, default=1,
              help='applicable if year list not provided. How many previous years to consider')
def batsman_only(year_list,no_of_years):

    year_list = list(year_list)
    create_batsman_rank(year_list,no_of_years=no_of_years)

@rank.command()
@click.option('--year_list', multiple=True, help='list of years.')
@click.option('--no_of_years', type=int, default=1,
              help='applicable if year list not provided. How many previous years to consider')
def bowler_only(year_list,no_of_years):

    year_list = list(year_list)
    create_bowler_rank(year_list,no_of_years=no_of_years)

@rank.command()
@click.option('--year_list', multiple=True, help='list of years.')
@click.option('--no_of_years', type=int, default=1,
              help='applicable if year list not provided. How many previous years to consider')
def country(year_list,no_of_years):

    year_list = list(year_list)
    #create_country_rank(year_list,no_of_years)


if __name__=='__main__':
    rank()


