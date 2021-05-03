import pandas as pd
import numpy as np
from ipl.data_loader import data_loader as dl
from ipl.preprocessing import rank
# from ipl.retrain import create_train_test as ctt
from ipl.model_util import odi_util as outil
import os
from datetime import datetime,date
import dateutil
import pickle
import numpy as np

from sklearn.linear_model import LinearRegression

SELECTED_FIRST_INNINGS_FEATURE_LIST_CACHE = None
SELECTED_SECOND_INNINGS_FEATURE_LIST_CACHE = None
TEAM_OPPONENT_LOCATION_EMBEDDING_MODEL_CACHE = None
COUNTRY_ENC_MAP_CACHE = None
LOC_ENC_MAP_CACHE = None
BATSMAN_ENC_MAP_CACHE = None
BOWLER_ENC_MAP_CACHE = None
LOC_ENC_MAP_FOR_BATSMAN_CACHE = None
BATSMAN_EMBEDDING_MODEL_CACHE = None

ADVERSARIAL_BATSMAN_MODEL_CACHE = None
ADVERSARIAL_BOWLER_MODEL_CACHE = None
ADVERSARIAL_LOCATION_MODEL_CACHE = None
#ADVERSARIAL_LOCATION_MODEL_CACHE = None


def get_trend(input_df, team_or_opponent, team_name, target_field):
    if input_df.shape[0]==0:
        return None, None,None,None
    input_df.rename(columns={'winner': 'winning_team'}, inplace=True)

    selected_match_id_list = list(input_df['match_id'])
    match_detail_list = []
    for match_id in selected_match_id_list:
        match_info = pd.read_csv(dl.CSV_LOAD_LOCATION+os.sep
                                 + str(match_id) + '.csv')
        match_detail_list.append(match_info)
    match_detail_df = pd.concat(match_detail_list)
    match_detail_df.fillna('NA', inplace=True)

    match_detail_df = input_df.merge(match_detail_df, how='inner', on='match_id')

    sorted_df = match_detail_df[match_detail_df[team_or_opponent].isin(team_name)].groupby('match_id').agg(
        {'date': 'min', target_field: 'sum'}).reset_index()
    sorted_df.sort_values('date', inplace=True)

    y = np.array(sorted_df[target_field])
    x = np.array(range(sorted_df.shape[0])).reshape(-1, 1) + 1
    linear_trend_model = LinearRegression()
    linear_trend_model.fit(x, y)
    next_instance_num = x.shape[0] + 1

    base = linear_trend_model.intercept_
    trend = linear_trend_model.coef_[0]
    trend_predict = linear_trend_model.predict(np.array([next_instance_num]).reshape(-1, 1))[0]
    mean = sorted_df[target_field].mean()

    return base, trend, trend_predict, mean


def get_specified_summary_df(match_summary_df=None,ref_date=None,no_of_years=None):

    if match_summary_df is None:
        custom_date_parser = lambda x: datetime.strptime(x, "%Y-%m-%d")
        match_summary_df = pd.read_csv(dl.CSV_LOAD_LOCATION + os.sep + 'match_list.csv',
                                       parse_dates=['date'], date_parser=custom_date_parser)
    if ref_date is None:
        today = date.today()
        ref_date = datetime(year=today.year, month=today.month, day=today.day)

    match_summary_df = match_summary_df[match_summary_df['date'] < ref_date]

    if no_of_years is not None:
        a_year = dateutil.relativedelta.relativedelta(years=no_of_years)
        begin_date = ref_date - a_year
        match_summary_df = match_summary_df[match_summary_df['date'] >= begin_date]

    return match_summary_df


def get_trend_with_opponent(team,opponent,match_summary_df=None,ref_date=None,no_of_years=None):
    match_summary_df = get_specified_summary_df(match_summary_df=match_summary_df,
                                                ref_date=ref_date,
                                                no_of_years=no_of_years)

    last_5_opponent = match_summary_df[(match_summary_df['first_innings'] == team)
                                       & (match_summary_df['second_innings'] == opponent)
                                       ].sort_values('date', ascending=False).head(5)

    return get_trend(last_5_opponent, 'team', [team], 'total')


def get_trend_at_location(team,location,match_summary_df=None,ref_date=None,no_of_years=None):
    match_summary_df = get_specified_summary_df(match_summary_df=match_summary_df,
                                                ref_date=ref_date,
                                                no_of_years=no_of_years)

    last_5_location = match_summary_df[(match_summary_df['first_innings'] == team)
                                       & (match_summary_df['location'] == location)
                                       ].sort_values('date', ascending=False).head(5)

    return get_trend(last_5_location, 'team', [team], 'total')


def get_trend_recent(team,match_summary_df=None,ref_date=None,no_of_years=None):
    match_summary_df = get_specified_summary_df(match_summary_df=match_summary_df,
                                                ref_date=ref_date,
                                                no_of_years=no_of_years)

    last_5_match = match_summary_df[(match_summary_df['first_innings'] == team)
                                    ].sort_values('date', ascending=False).head(5)

    return get_trend(last_5_match, 'team', [team], 'total')


# def get_country_score(country, ref_date=None):
#
#     country_rank_file = rank.get_latest_rank_file('country',ref_date=ref_date)
#     country_rank_df = pd.read_csv(country_rank_file)
#     if country_rank_df[country_rank_df['country'] == country].shape[0]==0:
#         raise Exception('Country score not available')
#     score = country_rank_df[country_rank_df['country'] == country]['score'].values[0]
#
#     return score
#
#
def get_batsman_mean_max(country,batsman_list,ref_date=None):
    batsman_rank_file = rank.get_latest_rank_file('batsman',ref_date=ref_date)
    batsman_rank_df = pd.read_csv(batsman_rank_file)
    selected_batsman_df = batsman_rank_df[(batsman_rank_df['batsman'].isin(batsman_list))]
    if selected_batsman_df.shape[0]==0:
        raise Exception('No batsman score is available for '+country)
    selected_batsman_df = selected_batsman_df.sort_values('batsman_score',ascending=False)
    #batsman_mean = selected_batsman_df.head(7)['batsman_score'].mean()
    batsman_mean = selected_batsman_df['batsman_score'].mean()
    batsman_max = selected_batsman_df['batsman_score'].max()
    batsman_sum = selected_batsman_df['batsman_score'].sum()
    return batsman_mean,batsman_max,batsman_sum


def get_bowler_mean_max(country,bowler_list,ref_date=None):
    bowler_rank_file = rank.get_latest_rank_file('bowler',ref_date=ref_date)
    bowler_rank_df = pd.read_csv(bowler_rank_file)
    if bowler_rank_df[(bowler_rank_df['bowler'].isin(bowler_list))].shape[0]==0:
        raise Exception('No bowler score available for '+country)
    bowler_mean = bowler_rank_df[(bowler_rank_df['bowler'].isin(bowler_list))]['bowler_score'].mean()
    bowler_max = bowler_rank_df[(bowler_rank_df['bowler'].isin(bowler_list))]['bowler_score'].max()
    bowler_sum = bowler_rank_df[(bowler_rank_df['bowler'].isin(bowler_list))]['bowler_score'].sum()
    return bowler_mean,bowler_max,bowler_sum


def get_instance_feature_dict(team, opponent, location, team_player_list, opponent_player_list, ref_date=None, no_of_years=None):

    batsman_mean,batsman_max,batsman_sum= get_batsman_mean_max(team, team_player_list, ref_date=ref_date)
    bowler_mean,bowler_max,bowler_sum= get_bowler_mean_max(opponent, opponent_player_list, ref_date=ref_date)


    feature_dict = {
        'team': team,
        'opponent': opponent,
        'location': location,
        'batsman_mean': batsman_mean,
        'batsman_max': batsman_max,
        'batsman_sum':batsman_sum,
        'bowler_mean': bowler_mean,
        'bowler_max': bowler_max,
        'bowler_sum': bowler_sum


    }

    return feature_dict


def get_first_innings_feature_vector(team, opponent, location, team_player_list, opponent_player_list, ref_date=None, no_of_years=None):
    """ this is not scaled"""
    global SELECTED_FIRST_INNINGS_FEATURE_LIST_CACHE
    feature_dict = get_instance_feature_dict(team, opponent, location,
                                             team_player_list, opponent_player_list, ref_date=ref_date, no_of_years=no_of_years)

    if SELECTED_FIRST_INNINGS_FEATURE_LIST_CACHE is None:
        SELECTED_FIRST_INNINGS_FEATURE_LIST_CACHE = pickle.load(open(outil.MODEL_DIR + os.sep + outil.FIRST_INNINGS_FEATURE_PICKLE, 'rb'))

    selected_feature_list = SELECTED_FIRST_INNINGS_FEATURE_LIST_CACHE

    feature_values = list()
    for feaure_name in selected_feature_list:
        feature_values.append(feature_dict[feaure_name])

    return np.array(feature_values)


def get_second_innings_feature_vector(target, team, opponent, location, team_player_list, opponent_player_list, ref_date=None, no_of_years=None):
    """ this is not scaled"""
    global SELECTED_SECOND_INNINGS_FEATURE_LIST_CACHE
    feature_dict = get_instance_feature_dict(team, opponent, location,
                                             team_player_list, opponent_player_list, ref_date=ref_date, no_of_years=no_of_years)
    feature_dict['target_score'] = target

    if SELECTED_SECOND_INNINGS_FEATURE_LIST_CACHE == None:
        SELECTED_SECOND_INNINGS_FEATURE_LIST_CACHE = pickle.load(open(outil.MODEL_DIR + os.sep + outil.SECOND_INNINGS_FEATURE_PICKLE, 'rb'))
    selected_feature_list = SELECTED_SECOND_INNINGS_FEATURE_LIST_CACHE

    feature_values = list()
    for feaure_name in selected_feature_list:
        feature_values.append(feature_dict[feaure_name])

    return np.array(feature_values)


# def get_team_opponent_location_embedding(team,opponent,location):
#     global TEAM_OPPONENT_LOCATION_EMBEDDING_MODEL_CACHE
#     global COUNTRY_ENC_MAP_CACHE
#     global LOC_ENC_MAP_CACHE
#
#     if TEAM_OPPONENT_LOCATION_EMBEDDING_MODEL_CACHE is None:
#         TEAM_OPPONENT_LOCATION_EMBEDDING_MODEL_CACHE = outil.load_keras_model(outil.MODEL_DIR \
#                                                               + os.sep \
#                                                               + outil.TEAM_OPPONENT_LOCATION_EMBEDDING_MODEL)
#
#     team_opponent_location_embedding = TEAM_OPPONENT_LOCATION_EMBEDDING_MODEL_CACHE
#
#     if COUNTRY_ENC_MAP_CACHE is None:
#         COUNTRY_ENC_MAP_CACHE = pickle.load(open(outil.MODEL_DIR \
#                                    + os.sep \
#                                    + outil.COUNTRY_ENCODING_MAP,'rb'))
#     country_enc_map = COUNTRY_ENC_MAP_CACHE
#
#     if LOC_ENC_MAP_CACHE is None:
#         LOC_ENC_MAP_CACHE = pickle.load(open(outil.MODEL_DIR \
#                                              + os.sep \
#                                              + outil.LOC_ENCODING_MAP,'rb'))
#     loc_enc_map = LOC_ENC_MAP_CACHE
#
#     if team not in country_enc_map or opponent not in country_enc_map:
#         raise Exception('Team or opponent not available')
#     team_oh_v = np.array(country_enc_map[team]).reshape(1, -1)
#     opponent_oh_v = np.array(country_enc_map[opponent]).reshape(1, -1)
#     if location not in loc_enc_map:
#         #print('did not find in ',outil.MODEL_DIR+ os.sep+ outil.LOC_ENCODING_MAP)
#         raise Exception("location not available")
#     loc_oh_v = np.array(loc_enc_map[location]).reshape(1, -1)
#
#     country_embedding = team_opponent_location_embedding.predict([team_oh_v, opponent_oh_v, loc_oh_v]).reshape(-1)
#
#     return country_embedding


def get_oh_pos(pos):
    vec=np.zeros((11)).astype(int)
    vec[pos-1]=1
    return vec


# def get_batsman_embedding(batsman_list,team,opponent,location):
#
#     global BATSMAN_EMBEDDING_MODEL_CACHE
#     global COUNTRY_ENC_MAP_CACHE
#     global BATSMAN_ENC_MAP_CACHE
#     global LOC_ENC_MAP_FOR_BATSMAN_CACHE
#
#     if BATSMAN_EMBEDDING_MODEL_CACHE is None:
#         BATSMAN_EMBEDDING_MODEL_CACHE = outil.load_keras_model(outil.MODEL_DIR \
#                                                    + os.sep
#                                                    + outil.BATSMAN_EMBEDDING_MODEL)
#     batsman_embedding = BATSMAN_EMBEDDING_MODEL_CACHE
#
#     if COUNTRY_ENC_MAP_CACHE is None:
#         COUNTRY_ENC_MAP_CACHE = pickle.load(open(outil.MODEL_DIR \
#                                    + os.sep \
#                                    + outil.COUNTRY_ENCODING_MAP,'rb'))
#     country_enc_map = COUNTRY_ENC_MAP_CACHE
#
#     if BATSMAN_ENC_MAP_CACHE is None:
#         BATSMAN_ENC_MAP_CACHE = pickle.load(open(outil.MODEL_DIR \
#                                    + os.sep \
#                                    + outil.BATSMAN_ENCODING_MAP, 'rb'))
#     batsman_enc_map = BATSMAN_ENC_MAP_CACHE
#
#     if LOC_ENC_MAP_FOR_BATSMAN_CACHE is None:
#         LOC_ENC_MAP_FOR_BATSMAN_CACHE = pickle.load(open(outil.MODEL_DIR \
#                                    + os.sep \
#                                    + outil.LOC_ENCODING_MAP_FOR_BATSMAN, 'rb'))
#     loc_enc_map_for_batsman = LOC_ENC_MAP_FOR_BATSMAN_CACHE
#
#     loc_oh = loc_enc_map_for_batsman[location]
#     opposition_oh = country_enc_map[opponent]
#
#     batsman_oh_list = []
#     position_oh_list = []
#     loc_oh_list = []
#     opposition_oh_list = []
#     # print('getting batsman details')
#     for bi,batsman in enumerate(batsman_list):
#
#         if team.strip() + ' ' + batsman.strip() not in batsman_enc_map:
#             continue
#         batsman_oh = batsman_enc_map[team.strip() + ' ' + batsman.strip()]
#         position_oh = get_oh_pos(bi + 1)
#
#         batsman_oh_list.append(batsman_oh)
#         position_oh_list.append(position_oh)
#         loc_oh_list.append(loc_oh)
#         opposition_oh_list.append(opposition_oh)
#
#     if len(batsman_oh_list)==0:
#         raise Exception('No Batsman embedding available for '+team)
#
#     batsman_mat = np.stack(batsman_oh_list)
#     position_mat = np.stack(position_oh_list)
#     loc_mat = np.stack(loc_oh_list)
#     opposition_mat = np.stack(opposition_oh_list)
#     # print('encoding')
#     batsman_group_enc_mat = batsman_embedding.predict([batsman_mat, position_mat, loc_mat, opposition_mat])
#     batsman_embedding_sum = batsman_group_enc_mat.sum(axis=0)
#
#     return batsman_embedding_sum
#
#
# def get_first_innings_feature_embedding_vector(team, opponent, location, team_player_list, opponent_player_list,
#                                                ref_date=None, no_of_years=None):
#     feature_vector = get_first_innings_feature_vector(team, opponent, location,
#                                                       team_player_list, opponent_player_list,
#                                                       ref_date=ref_date, no_of_years=no_of_years)
#
#     country_embedding_vector = get_team_opponent_location_embedding(team,opponent,location)
#
#     batsman_embedding_vector = get_batsman_embedding(team_player_list, team, opponent, location)
#
#     final_vector = np.concatenate([batsman_embedding_vector, country_embedding_vector, feature_vector])
#
#     return final_vector
#
#
#
# def get_second_innings_feature_embedding_vector(target, team, opponent, location, team_player_list, opponent_player_list,
#                                                 ref_date=None, no_of_years=None):
#     feature_vector = get_second_innings_feature_vector(target, team, opponent, location,
#                                                        team_player_list, opponent_player_list,
#                                                        ref_date=ref_date, no_of_years=no_of_years)
#
#     country_embedding_vector = get_team_opponent_location_embedding(team,opponent,location)
#
#     batsman_embedding_vector = get_batsman_embedding(team_player_list, team, opponent, location)
#
#     final_vector = np.concatenate([batsman_embedding_vector, country_embedding_vector, feature_vector])
#
#     return final_vector
#
#
# ### Feature engineering for Batsman run prediction
#
#
# def get_single_batsman_embedding(batsman,position,team,opponent,location):
#
#     global BATSMAN_EMBEDDING_MODEL_CACHE
#     global COUNTRY_ENC_MAP_CACHE
#     global BATSMAN_ENC_MAP_CACHE
#     global LOC_ENC_MAP_FOR_BATSMAN_CACHE
#
#     if BATSMAN_EMBEDDING_MODEL_CACHE is None:
#         BATSMAN_EMBEDDING_MODEL_CACHE = outil.load_keras_model(outil.MODEL_DIR \
#                                                    + os.sep
#                                                    + outil.BATSMAN_EMBEDDING_MODEL)
#     batsman_embedding = BATSMAN_EMBEDDING_MODEL_CACHE
#
#     if COUNTRY_ENC_MAP_CACHE is None:
#         COUNTRY_ENC_MAP_CACHE = pickle.load(open(outil.MODEL_DIR \
#                                    + os.sep \
#                                    + outil.COUNTRY_ENCODING_MAP,'rb'))
#     country_enc_map = COUNTRY_ENC_MAP_CACHE
#
#     if BATSMAN_ENC_MAP_CACHE is None:
#         BATSMAN_ENC_MAP_CACHE = pickle.load(open(outil.MODEL_DIR \
#                                    + os.sep \
#                                    + outil.BATSMAN_ENCODING_MAP, 'rb'))
#     batsman_enc_map = BATSMAN_ENC_MAP_CACHE
#
#     if LOC_ENC_MAP_FOR_BATSMAN_CACHE is None:
#         LOC_ENC_MAP_FOR_BATSMAN_CACHE = pickle.load(open(outil.MODEL_DIR \
#                                    + os.sep \
#                                    + outil.LOC_ENCODING_MAP_FOR_BATSMAN, 'rb'))
#     loc_enc_map_for_batsman = LOC_ENC_MAP_FOR_BATSMAN_CACHE
#
#     loc_oh = loc_enc_map_for_batsman[location].reshape(1,-1)
#     opposition_oh = country_enc_map[opponent].reshape(1,-1)
#     batsman_oh = batsman_enc_map[team.strip() + ' ' + batsman.strip()].reshape(1,-1)
#     position_oh = get_oh_pos(position).reshape(1,-1)
#
#     batsman_group_emb_vec = batsman_embedding.predict([batsman_oh, position_oh, loc_oh, opposition_oh])[0]
#
#     return batsman_group_emb_vec
#
#
# def get_batsman_score(country,batsman,ref_date=None):
#     batsman_rank_file = rank.get_latest_rank_file('batsman',ref_date=ref_date)
#     batsman_rank_df = pd.read_csv(batsman_rank_file)
#     selected_batsman_df = batsman_rank_df[(batsman_rank_df['batsman']==batsman.strip())\
#                                    & (batsman_rank_df['country']==country)]
#     if selected_batsman_df.shape[0]==0:
#         raise Exception('No batsman score is available for '+country)
#
#     score = selected_batsman_df['batsman_score'].values[0]
#
#     return score
#
#
# def get_batsman_features_with_embedding(batsman,position,opponent_player_list,team,opponent,location,ref_date=None):
#     batsman_group_emb_vec = get_single_batsman_embedding(batsman,position,team,opponent,location)
#     batsman_score = get_batsman_score(team,batsman,ref_date=ref_date)
#     opponent_score = get_country_score(opponent, ref_date=ref_date)
#     bowler_mean, bowler_max, bowler_sum = get_bowler_mean_max(opponent, opponent_player_list, ref_date=ref_date)
#
#     combined_vector = list(batsman_group_emb_vec) + [batsman_score,opponent_score,bowler_mean]
#
#     return combined_vector


def get_batsman_adversarial_vector(team,team_player_list):
    global BATSMAN_ENC_MAP_CACHE
    global ADVERSARIAL_BATSMAN_MODEL_CACHE

    if BATSMAN_ENC_MAP_CACHE is None:
        BATSMAN_ENC_MAP_CACHE = pickle.load(open(outil.MODEL_DIR \
                                   + os.sep \
                                   + outil.BATSMAN_ENCODING_MAP, 'rb'))
    batsman_enc_map = BATSMAN_ENC_MAP_CACHE

    if ADVERSARIAL_BATSMAN_MODEL_CACHE is None:
        ADVERSARIAL_BATSMAN_MODEL_CACHE = outil.load_keras_model(outil.MODEL_DIR \
                                                               + os.sep
                                                               + outil.ADVERSARIAL_BATSMAN_MODEL)
    batsman_embedding_model = ADVERSARIAL_BATSMAN_MODEL_CACHE
    batsman_oh_list = list()
    for batsman in team_player_list:
        if batsman.strip() not in batsman_enc_map:
            continue
        batsman_oh = batsman_enc_map[batsman.strip()]
        batsman_oh_list.append(batsman_oh)

    if len(batsman_oh_list)==0:
        raise Exception("Not enough batsman")
    batsman_oh_matrix = np.stack(batsman_oh_list)
    batsman_emb_vec = batsman_embedding_model.predict(batsman_oh_matrix)

    return np.sum(batsman_emb_vec,axis=0)

def get_bowler_adversarial_vector(opponent,opponent_player_list):
    global BOWLER_ENC_MAP_CACHE
    global ADVERSARIAL_BOWLER_MODEL_CACHE

    if BOWLER_ENC_MAP_CACHE is None:
        BOWLER_ENC_MAP_CACHE = pickle.load(open(outil.MODEL_DIR \
                                                 + os.sep \
                                                 + outil.BOWLER_ENCODING_MAP, 'rb'))
    bowler_enc_map = BOWLER_ENC_MAP_CACHE

    if ADVERSARIAL_BOWLER_MODEL_CACHE is None:
        ADVERSARIAL_BOWLER_MODEL_CACHE = outil.load_keras_model(outil.MODEL_DIR \
                                                                 + os.sep
                                                                 + outil.ADVERSARIAL_BOWLER_MODEL)
    bowler_embedding_model = ADVERSARIAL_BOWLER_MODEL_CACHE
    bowler_oh_list = list()
    for bowler in opponent_player_list:
        if bowler.strip() not in bowler_enc_map:
            continue
        bowler_oh = bowler_enc_map[bowler.strip()]
        bowler_oh_list.append(bowler_oh)

    if len(bowler_oh_list)==0:
        raise Exception("Not enough bowler")

    bowler_oh_matrix = np.stack(bowler_oh_list)
    bowler_emb_vec = bowler_embedding_model.predict(bowler_oh_matrix)

    return np.sum(bowler_emb_vec, axis=0)


def get_location_adversarial_vector(location):
    global LOC_ENC_MAP_FOR_BATSMAN_CACHE
    global ADVERSARIAL_LOCATION_MODEL_CACHE

    if LOC_ENC_MAP_FOR_BATSMAN_CACHE is None:
        LOC_ENC_MAP_FOR_BATSMAN_CACHE = pickle.load(open(outil.MODEL_DIR \
                                   + os.sep \
                                   + outil.LOC_ENCODING_MAP, 'rb'))
    loc_enc_map_for_batsman = LOC_ENC_MAP_FOR_BATSMAN_CACHE

    if ADVERSARIAL_LOCATION_MODEL_CACHE is None:
        ADVERSARIAL_LOCATION_MODEL_CACHE = outil.load_keras_model(outil.MODEL_DIR \
                                                                 + os.sep
                                                                 + outil.ADVERSARIAL_LOCATION_MODEL)
    location_emb_model = ADVERSARIAL_LOCATION_MODEL_CACHE

    if location in loc_enc_map_for_batsman:
        loc_oh = loc_enc_map_for_batsman[location].reshape(1, -1)
    else:
        loc_oh = loc_enc_map_for_batsman["unknown"].reshape(1, -1)
    location_emb = location_emb_model.predict(loc_oh)[0]
    return location_emb

def get_adversarial_first_innings_feature_vector(team, opponent, location, team_player_list, opponent_player_list,
                                               ref_date=None, no_of_years=None):

    feature_vector = get_first_innings_feature_vector(team, opponent, location,
                                                      team_player_list, opponent_player_list,
                                                      ref_date=ref_date, no_of_years=no_of_years)

    #country_embedding_vector = get_team_opponent_location_embedding(team,opponent,location)

    #batsman_embedding_vector = get_batsman_embedding(team_player_list, team, opponent, location)

    batsman_embedding_vector = get_batsman_adversarial_vector(team,team_player_list)
    bowler_embedding_vector = get_bowler_adversarial_vector(opponent,opponent_player_list)
    location_embedding_vector = get_location_adversarial_vector(location)

    final_vector = np.concatenate([feature_vector,batsman_embedding_vector, bowler_embedding_vector, location_embedding_vector])

    return final_vector


def get_adversarial_second_innings_feature_vector(target,team, opponent, location, team_player_list, opponent_player_list,
                                               ref_date=None, no_of_years=None):

    feature_vector = get_second_innings_feature_vector(target,team, opponent, location,
                                                      team_player_list, opponent_player_list,
                                                      ref_date=ref_date, no_of_years=no_of_years)

    #country_embedding_vector = get_team_opponent_location_embedding(team,opponent,location)

    #batsman_embedding_vector = get_batsman_embedding(team_player_list, team, opponent, location)

    batsman_embedding_vector = get_batsman_adversarial_vector(team,team_player_list)
    bowler_embedding_vector = get_bowler_adversarial_vector(opponent,opponent_player_list)
    location_embedding_vector = get_location_adversarial_vector(location)

    final_vector = np.concatenate([feature_vector,batsman_embedding_vector, bowler_embedding_vector, location_embedding_vector])

    return final_vector

