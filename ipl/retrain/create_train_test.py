import pandas as pd
from ipl.feature_engg import util as cricutil
from ipl.data_loader import  data_loader as dl
from ipl.model_util import odi_util as outil
from ipl.feature_engg import feature_extractor as fe

import os
import numpy as np
import pickle
from tqdm import tqdm
import click
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

TRAIN_TEST_DIR = 'retrain_ipl_xy'
# country_emb_feature_team_oh_train_x = 'country_emb_team_oh_train_x.pkl'
# country_emb_feature_opponent_oh_train_x = 'country_emb_opponent_oh_train_x.pkl'
# country_emb_feature_location_oh_train_x = 'country_emb_location_oh_train_x.pkl'
# country_emb_feature_runs_scored_train_y = 'country_emb_runs_scored_train_y.pkl'
#
# country_emb_feature_team_oh_test_x = 'country_emb_team_oh_test_x.pkl'
# country_emb_feature_opponent_oh_test_x = 'country_emb_opponent_oh_test_x.pkl'
# country_emb_feature_location_oh_test_x = 'country_emb_location_oh_test_x.pkl'
# country_emb_feature_runs_scored_test_y = 'country_emb_runs_scored_test_y'
#
#
# batsman_emb_feature_batsman_oh_train_x = 'batsman_emb_batsman_oh_train_x.pkl'
# batsman_emb_feature_position_oh_train_x = 'batsman_emb_position_oh_train_x.pkl'
# batsman_emb_feature_location_oh_train_x = 'batsman_emb_location_oh_train_x.pkl'
# batsman_emb_feature_opponent_oh_train_x = 'batsman_emb_opponent_oh_train_x.pkl'
# batsman_emb_feature_runs_scored_train_y = 'batsman_emb_runs_scored_train_y.pkl'
#
# batsman_emb_feature_batsman_oh_test_x = 'batsman_emb_batsman_oh_test_x.pkl'
# batsman_emb_feature_position_oh_test_x = 'batsman_emb_position_oh_test_x.pkl'
# batsman_emb_feature_location_oh_test_x = 'batsman_emb_location_oh_test_x.pkl'
# batsman_emb_feature_opponent_oh_test_x = 'batsman_emb_opponent_oh_test_x.pkl'
# batsman_emb_feature_runs_scored_test_y = 'batsman_emb_runs_scored_test_y.pkl'
#
first_innings_base_train_x = 'first_innings_base_train_x.pkl'
first_innings_base_train_y = 'first_innings_base_train_y.pkl'
first_innings_base_test_x = 'first_innings_base_test_x.pkl'
first_innings_base_test_y = 'first_innings_base_test_y.pkl'
first_innings_base_columns = 'first_innings_base_columns.pkl'

second_innings_base_train_x = 'second_innings_base_train_x.pkl'
second_innings_base_train_y = 'second_innings_base_train_y.pkl'
second_innings_base_test_x = 'second_innings_base_test_x.pkl'
second_innings_base_test_y = 'second_innings_base_test_y.pkl'
second_innings_base_columns = 'second_innings_base_columns.pkl'

# first_innings_train_x = 'first_innings_train_x.pkl'
# first_innings_train_y = 'first_innings_train_y.pkl'
# first_innings_test_x = 'first_innings_test_x.pkl'
# first_innings_test_y = 'first_innings_test_y.pkl'
#
# second_innings_train_x = 'second_innings_train_x.pkl'
# second_innings_train_y = 'second_innings_train_y.pkl'
# second_innings_test_x = 'second_innings_test_x.pkl'
# second_innings_test_y = 'second_innings_test_y.pkl'
#
# batsman_runs_train_x = 'batsman_runs_train_x.pkl'
# batsman_runs_train_y = 'batsman_runs_train_y.pkl'
# batsman_runs_test_x = 'batsman_runs_test_x.pkl'
# batsman_runs_test_y = 'batsman_runs_test_y.pkl'

adversarial_feature_batsman_oh_train_x = 'adversarial_feature_batsman_oh_train_x.pkl'
adversarial_feature_position_oh_train_x = 'adversarial_feature_position_oh_train_x.pkl'
adversarial_feature_location_oh_train_x = 'adversarial_feature_location_oh_train_x.pkl'
adversarial_feature_bowler_oh_train_x = 'adversarial_feature_bowler_oh_train_x.pkl'
adversarial_feature_runs_scored_train_y = 'adversarial_feature_runs_scored_train_y.pkl'

adversarial_feature_batsman_oh_test_x = 'adversarial_feature_batsman_oh_test_x.pkl'
adversarial_feature_position_oh_test_x = 'adversarial_feature_position_oh_test_x.pkl'
adversarial_feature_location_oh_test_x = 'adversarial_feature_location_oh_test_x.pkl'
adversarial_feature_bowler_oh_test_x = 'adversarial_feature_bowler_oh_test_x.pkl'
adversarial_feature_runs_scored_test_y = 'adversarial_feature_runs_scored_test_y.pkl'

adversarial_first_innings_train_x = 'adversarial_first_innings_train_x.pkl'
adversarial_first_innings_train_y = 'adversarial_first_innings_train_y.pkl'
adversarial_first_innings_test_x = 'adversarial_first_innings_test_x'
adversarial_first_innings_test_y = 'adversarial_first_innings_test_y'

adversarial_second_innings_train_x = 'adversarial_second_innings_train_x.pkl'
adversarial_second_innings_train_y = 'adversarial_second_innings_train_y.pkl'
adversarial_second_innings_test_x = 'adversarial_second_innings_test_x'
adversarial_second_innings_test_y = 'adversarial_second_innings_test_y'



# def create_country_embedding_train_test(train_start,test_start,test_end=None,encoding_source='dev'):
#
#     if not os.path.isdir(TRAIN_TEST_DIR):
#         os.makedirs(TRAIN_TEST_DIR)
#
#     if encoding_source == 'production':
#         encoding_dir = outil.PROD_DIR
#     else:
#         encoding_dir = outil.DEV_DIR
#
#     train_start_dt = cricutil.str_to_date_time(train_start)
#     test_start_dt = cricutil.str_to_date_time(test_start)
#     if test_end is None:
#         test_end_dt = cricutil.today_as_date_time()
#     else:
#         test_end_dt = cricutil.str_to_date_time(test_end)
#
#     overall_start = train_start_dt
#     overall_end = test_end_dt
#     match_list_df = cricutil.read_csv_with_date(dl.CSV_LOAD_LOCATION + os.sep + 'match_list.csv')
#     match_list_df = match_list_df[(match_list_df['date'] >= overall_start) & \
#                                   (match_list_df['date'] <= overall_end)]
#     match_stats_df = pd.read_csv(dl.CSV_LOAD_LOCATION + os.sep + 'match_stats.csv')
#     match_list_df = match_list_df.merge(match_stats_df,how='inner',on='match_id')
#     match_list_df = match_list_df[match_list_df['first_innings']==match_list_df['team_statistics']]
#
#     no_of_matches = match_list_df.shape[0]
#     country_enc_map = pickle.load(open(os.path.join(encoding_dir,outil.COUNTRY_ENCODING_MAP),'rb'))
#     location_enc_map = pickle.load(open(os.path.join(encoding_dir, outil.LOC_ENCODING_MAP), 'rb'))
#
#     team_oh_list_train = []
#     opponent_oh_list_train = []
#     location_oh_list_train =[]
#     runs_scored_list_train =[]
#
#     team_oh_list_test = []
#     opponent_oh_list_test = []
#     location_oh_list_test = []
#     runs_scored_list_test = []
#
#     for index in tqdm(range(no_of_matches)):
#
#         team = match_list_df.iloc[index]['first_innings'].strip()
#         opponent = match_list_df.iloc[index]['second_innings'].strip()
#         location = match_list_df.iloc[index]['location'].strip()
#         runs_scored = match_list_df.iloc[index]['total_run']
#         date = match_list_df.iloc[index]['date']
#         try:
#             team_oh = np.array(country_enc_map[team])
#             opponent_oh = np.array(country_enc_map[opponent])
#             location_oh = np.array(location_enc_map[location])
#             if date<test_start_dt:
#                 team_oh_list_train.append(team_oh)
#                 opponent_oh_list_train.append(opponent_oh)
#                 location_oh_list_train.append(location_oh)
#                 runs_scored_list_train.append(runs_scored)
#             else:
#                 team_oh_list_test.append(team_oh)
#                 opponent_oh_list_test.append(opponent_oh)
#                 location_oh_list_test.append(location_oh)
#                 runs_scored_list_test.append(runs_scored)
#
#
#         except Exception as ex:
#             print(ex,' for ',team,opponent,location,' on ',date)
#
#     team_oh_train_x = np.stack(team_oh_list_train)
#     opponent_oh_train_x = np.stack(opponent_oh_list_train)
#     location_oh_train_x = np.stack(location_oh_list_train)
#     runs_scored_train_y = np.stack(runs_scored_list_train)
#
#     team_oh_test_x = np.stack(team_oh_list_test)
#     opponent_oh_test_x = np.stack(opponent_oh_list_test)
#     location_oh_test_x = np.stack(location_oh_list_test)
#     runs_scored_test_y = np.stack(runs_scored_list_test)
#
#     pickle.dump(team_oh_train_x, open(os.path.join(TRAIN_TEST_DIR, country_emb_feature_team_oh_train_x), 'wb'))
#     pickle.dump(opponent_oh_train_x, open(os.path.join(TRAIN_TEST_DIR, country_emb_feature_opponent_oh_train_x), 'wb'))
#     pickle.dump(location_oh_train_x, open(os.path.join(TRAIN_TEST_DIR, country_emb_feature_location_oh_train_x), 'wb'))
#     pickle.dump(runs_scored_train_y, open(os.path.join(TRAIN_TEST_DIR, country_emb_feature_runs_scored_train_y), 'wb'))
#
#     outil.create_meta_info_entry('country_embedding_train_xy', train_start,
#                                  str(cricutil.substract_day_as_datetime(test_start_dt,1).date()),
#                                  file_list=[country_emb_feature_team_oh_train_x,
#                                             country_emb_feature_opponent_oh_train_x,
#                                             country_emb_feature_location_oh_train_x,
#                                             country_emb_feature_runs_scored_train_y])
#
#     pickle.dump(team_oh_test_x, open(os.path.join(TRAIN_TEST_DIR, country_emb_feature_team_oh_test_x), 'wb'))
#     pickle.dump(opponent_oh_test_x, open(os.path.join(TRAIN_TEST_DIR, country_emb_feature_opponent_oh_test_x), 'wb'))
#     pickle.dump(location_oh_test_x, open(os.path.join(TRAIN_TEST_DIR, country_emb_feature_location_oh_test_x), 'wb'))
#     pickle.dump(runs_scored_test_y, open(os.path.join(TRAIN_TEST_DIR, country_emb_feature_runs_scored_test_y), 'wb'))
#
#     outil.create_meta_info_entry('country_embedding_test_xy', test_start, str(test_end_dt.date()),
#                                  file_list=[country_emb_feature_team_oh_test_x,
#                                             country_emb_feature_opponent_oh_test_x,
#                                             country_emb_feature_location_oh_test_x,
#                                             country_emb_feature_runs_scored_test_y])
#
#
# def create_batsman_embedding_train_test(train_start,test_start,test_end=None,encoding_source='dev',include_not_batted=False):
#     if not os.path.isdir(TRAIN_TEST_DIR):
#         os.makedirs(TRAIN_TEST_DIR)
#
#     if encoding_source == 'production':
#         encoding_dir = outil.PROD_DIR
#     else:
#         encoding_dir = outil.DEV_DIR
#
#     train_start_dt = cricutil.str_to_date_time(train_start)
#     test_start_dt = cricutil.str_to_date_time(test_start)
#     if test_end is None:
#         test_end_dt = cricutil.today_as_date_time()
#     else:
#         test_end_dt = cricutil.str_to_date_time(test_end)
#
#     overall_start = train_start_dt
#     overall_end = test_end_dt
#     match_list_df = cricutil.read_csv_with_date(dl.CSV_LOAD_LOCATION + os.sep + 'match_list.csv')
#     match_list_df = match_list_df[(match_list_df['date'] >= overall_start) & \
#                                   (match_list_df['date'] <= overall_end)]
#     match_stats_df = pd.read_csv(dl.CSV_LOAD_LOCATION + os.sep + 'match_stats.csv')
#     match_list_df = match_list_df.merge(match_stats_df,how='inner',on='match_id')
#
#     no_of_matches = match_list_df.shape[0]
#     country_enc_map = pickle.load(open(os.path.join(encoding_dir, outil.COUNTRY_ENCODING_MAP), 'rb'))
#     location_enc_map_for_batsman = pickle.load(open(os.path.join(encoding_dir, outil.LOC_ENCODING_MAP_FOR_BATSMAN), 'rb'))
#     batsman_enc_map = pickle.load(open(os.path.join(encoding_dir, outil.BATSMAN_ENCODING_MAP), 'rb'))
#
#     batsman_oh_list_train = []
#     position_oh_list_train = []
#     location_oh_list_train = []
#     opponent_oh_list_train = []
#     runs_scored_list_train = []
#
#     batsman_oh_list_test = []
#     position_oh_list_test = []
#     location_oh_list_test = []
#     opponent_oh_list_test = []
#     runs_scored_list_test = []
#
#     for index in tqdm(range(no_of_matches)):
#         if match_list_df.iloc[index]['first_innings']==match_list_df.iloc[index]['team_statistics']:
#             opponent = match_list_df.iloc[index]['second_innings'].strip()
#             country = match_list_df.iloc[index]['first_innings'].strip()
#         else:
#             opponent = match_list_df.iloc[index]['first_innings'].strip()
#             country = match_list_df.iloc[index]['second_innings'].strip()
#         location = match_list_df.iloc[index]['location'].strip()
#         match_id = match_list_df.iloc[index]['match_id']
#
#         date = match_list_df.iloc[index]['date']
#
#         if location not in location_enc_map_for_batsman:
#             print('location ',location,' not encoded for ',country,opponent,' on ',date)
#             continue
#         if opponent not in country_enc_map:
#             print('opponent ', opponent, ' not encoded for ', country, opponent, ' on ', date, ' at ',location)
#             continue
#
#         location_oh = location_enc_map_for_batsman[location]
#         opponent_oh = country_enc_map[opponent]
#
#         for bi in range(11):
#             batsman = match_list_df.iloc[index]['batsman_'+str(bi+1)].strip()
#             try:
#                 if batsman == 'not_batted':
#                     if not include_not_batted:
#                         break
#                     else:
#                         batsman_oh = np.array(batsman_enc_map[batsman])
#                 else:
#
#                     batsman_oh = np.array(batsman_enc_map[country+' '+batsman.strip()])
#                 position_oh = fe.get_oh_pos(bi+1)
#                 runs_scored = match_list_df.iloc[index]['batsman_'+str(bi+1)+'_runs']
#
#                 if date<test_start_dt:
#                     batsman_oh_list_train.append(batsman_oh)
#                     position_oh_list_train.append(position_oh)
#                     location_oh_list_train.append(location_oh)
#                     opponent_oh_list_train.append(opponent_oh)
#                     runs_scored_list_train.append(runs_scored)
#                 else:
#                     batsman_oh_list_test.append(batsman_oh)
#                     position_oh_list_test.append(position_oh)
#                     location_oh_list_test.append(location_oh)
#                     opponent_oh_list_test.append(opponent_oh)
#                     runs_scored_list_test.append(runs_scored)
#             except Exception as ex:
#                 print(ex,' for ',country,batsman,opponent,date,' -match_id: ',match_id)
#
#     batsman_oh_train_x = np.stack(batsman_oh_list_train)
#     position_oh_train_x = np.stack(position_oh_list_train)
#     location_oh_train_x = np.stack(location_oh_list_train)
#     opponent_oh_train_x = np.stack(opponent_oh_list_train)
#     runs_scored_train_y = np.stack(runs_scored_list_train)
#
#     batsman_oh_test_x = np.stack(batsman_oh_list_test)
#     position_oh_test_x = np.stack(position_oh_list_test)
#     location_oh_test_x = np.stack(location_oh_list_test)
#     opponent_oh_test_x = np.stack(opponent_oh_list_test)
#     runs_scored_test_y = np.stack(runs_scored_list_test)
#
#     pickle.dump(batsman_oh_train_x, open(os.path.join(TRAIN_TEST_DIR, batsman_emb_feature_batsman_oh_train_x), 'wb'))
#     pickle.dump(position_oh_train_x, open(os.path.join(TRAIN_TEST_DIR, batsman_emb_feature_position_oh_train_x), 'wb'))
#     pickle.dump(location_oh_train_x, open(os.path.join(TRAIN_TEST_DIR, batsman_emb_feature_location_oh_train_x), 'wb'))
#     pickle.dump(opponent_oh_train_x, open(os.path.join(TRAIN_TEST_DIR, batsman_emb_feature_opponent_oh_train_x), 'wb'))
#     pickle.dump(runs_scored_train_y, open(os.path.join(TRAIN_TEST_DIR, batsman_emb_feature_runs_scored_train_y), 'wb'))
#
#     outil.create_meta_info_entry('batsman_embedding_train_xy', train_start,
#                                  str(cricutil.substract_day_as_datetime(test_start_dt, 1).date()),
#                                  file_list=[batsman_emb_feature_batsman_oh_train_x,
#                                             batsman_emb_feature_position_oh_train_x,
#                                             batsman_emb_feature_location_oh_train_x,
#                                             batsman_emb_feature_opponent_oh_train_x,
#                                             batsman_emb_feature_runs_scored_train_y])
#
#     pickle.dump(batsman_oh_test_x, open(os.path.join(TRAIN_TEST_DIR, batsman_emb_feature_batsman_oh_test_x), 'wb'))
#     pickle.dump(position_oh_test_x, open(os.path.join(TRAIN_TEST_DIR, batsman_emb_feature_position_oh_test_x), 'wb'))
#     pickle.dump(location_oh_test_x, open(os.path.join(TRAIN_TEST_DIR, batsman_emb_feature_location_oh_test_x), 'wb'))
#     pickle.dump(opponent_oh_test_x, open(os.path.join(TRAIN_TEST_DIR, batsman_emb_feature_opponent_oh_test_x), 'wb'))
#     pickle.dump(runs_scored_test_y, open(os.path.join(TRAIN_TEST_DIR, batsman_emb_feature_runs_scored_test_y), 'wb'))
#
#     outil.create_meta_info_entry('batsman_embedding_test_xy', test_start, str(test_end_dt.date()),
#                                  file_list=[batsman_emb_feature_batsman_oh_test_x,
#                                             batsman_emb_feature_position_oh_test_x,
#                                             batsman_emb_feature_location_oh_test_x,
#                                             batsman_emb_feature_opponent_oh_test_x,
#                                             batsman_emb_feature_runs_scored_test_y])
#
#
def create_first_innings_base_train_test(train_start,test_start,test_end=None):
    if not os.path.isdir(TRAIN_TEST_DIR):
        os.makedirs(TRAIN_TEST_DIR)

    outil.use_model_from('dev')
    train_start_dt = cricutil.str_to_date_time(train_start)
    test_start_dt = cricutil.str_to_date_time(test_start)
    if test_end is None:
        test_end_dt = cricutil.today_as_date_time()
    else:
        test_end_dt = cricutil.str_to_date_time(test_end)

    overall_start = train_start_dt
    overall_end = test_end_dt
    match_list_df = cricutil.read_csv_with_date(dl.CSV_LOAD_LOCATION + os.sep + 'match_list.csv')
    match_list_df = match_list_df[(match_list_df['date'] >= overall_start) & \
                                  (match_list_df['date'] <= overall_end)]
    match_stats_df = pd.read_csv(dl.CSV_LOAD_LOCATION + os.sep + 'match_stats.csv')
    match_list_df = match_list_df.merge(match_stats_df, how='inner', on='match_id')
    match_id_list = list(match_list_df['match_id'].unique())

    feature_list_train = []
    target_list_train = []

    feature_list_test =[]
    target_list_test = []
    no_of_basman = 0
    for index,match_id in tqdm(enumerate(match_id_list)):
        match_info = match_list_df[match_list_df['match_id']==match_id]
        team_info = match_info[match_info['first_innings']==match_info['team_statistics']]
        opponent_info = match_info[match_info['second_innings']==match_info['team_statistics']]

        team = team_info['team_statistics'].values[0].strip()
        opponent = opponent_info['team_statistics'].values[0].strip()
        location = team_info['location'].values[0].strip()
        ref_dt_np = team_info['date'].values[0]
        ref_date = cricutil.npdate_to_datetime(ref_dt_np)
        runs_scored = team_info['total_run'].values[0]

        team_player_list = list()
        for bi in range(11):
            batsman = team_info['batsman_'+str(bi+1)].values[0].strip()
            if batsman == 'not_batted':
                break
            else:
                team_player_list.append(batsman)
        opponent_player_list = list()
        for boi in range(11):
            bowler = opponent_info['bowler_' + str(boi + 1)].values[0].strip()
            if bowler == 'not_bowled':
                break
            else:
                opponent_player_list.append(bowler)


        try:
            feature_dict = fe.get_instance_feature_dict(team, opponent, location,
                                                        team_player_list, opponent_player_list,
                                                        ref_date)
            no_of_basman = no_of_basman+len(team_player_list)

            if ref_date<test_start_dt:
                feature_list_train.append(feature_dict)
                target_list_train.append(runs_scored)
            else:
                feature_list_test.append(feature_dict)
                target_list_test.append(runs_scored)
        except Exception as ex:
            print(ex, ' for ',team, opponent, location, ' on ',ref_date.date() )

    print('mean no of batsman - ',no_of_basman/index)
    train_y = np.stack(target_list_train)
    test_y = np.stack(target_list_test)

    train_x = np.array(pd.DataFrame(feature_list_train).drop(columns=['team','opponent','location']))
    test_x  = np.array(pd.DataFrame(feature_list_test).drop(columns=['team','opponent','location']))
    cols = list(pd.DataFrame(feature_list_train).drop(columns=['team','opponent','location']).columns)

    pickle.dump(train_x,open(os.path.join(TRAIN_TEST_DIR,first_innings_base_train_x),'wb'))
    pickle.dump(train_y, open(os.path.join(TRAIN_TEST_DIR,first_innings_base_train_y), 'wb'))
    pickle.dump(cols, open(os.path.join(outil.DEV_DIR, first_innings_base_columns), 'wb'))

    outil.create_meta_info_entry('first_innings_base_train_xy', train_start,
                                 str(cricutil.substract_day_as_datetime(test_start_dt, 1).date()),
                                 file_list=[first_innings_base_train_x,
                                            first_innings_base_train_y,
                                            first_innings_base_columns])

    pickle.dump(test_x, open(os.path.join(TRAIN_TEST_DIR, first_innings_base_test_x), 'wb'))
    pickle.dump(test_y, open(os.path.join(TRAIN_TEST_DIR, first_innings_base_test_y), 'wb'))

    outil.create_meta_info_entry('first_innings_base_test_xy', str(test_start_dt.date()),
                                 str(test_end_dt.date()),
                                 file_list=[first_innings_base_test_x,
                                            first_innings_base_test_y])


def create_second_innings_base_train_test(train_start,test_start,test_end=None):
    if not os.path.isdir(TRAIN_TEST_DIR):
        os.makedirs(TRAIN_TEST_DIR)

    outil.use_model_from('dev')
    train_start_dt = cricutil.str_to_date_time(train_start)
    test_start_dt = cricutil.str_to_date_time(test_start)
    if test_end is None:
        test_end_dt = cricutil.today_as_date_time()
    else:
        test_end_dt = cricutil.str_to_date_time(test_end)

    overall_start = train_start_dt
    overall_end = test_end_dt
    match_list_df = cricutil.read_csv_with_date(dl.CSV_LOAD_LOCATION + os.sep + 'match_list.csv')
    match_list_df = match_list_df[(match_list_df['date'] >= overall_start) & \
                                  (match_list_df['date'] <= overall_end)]
    match_stats_df = pd.read_csv(dl.CSV_LOAD_LOCATION + os.sep + 'match_stats.csv')
    match_list_df = match_list_df.merge(match_stats_df, how='inner', on='match_id')
    match_id_list = list(match_list_df['match_id'].unique())

    feature_list_train = []
    result_list_train = []

    feature_list_test =[]
    result_list_test = []
    no_of_basman = 0
    for index,match_id in tqdm(enumerate(match_id_list)):
        match_info = match_list_df[match_list_df['match_id']==match_id]
        team_info = match_info[match_info['second_innings']==match_info['team_statistics']]
        opponent_info = match_info[match_info['first_innings']==match_info['team_statistics']]

        team = team_info['team_statistics'].values[0].strip()
        opponent = opponent_info['team_statistics'].values[0].strip()
        location = team_info['location'].values[0].strip()
        ref_dt_np = team_info['date'].values[0]
        ref_date = cricutil.npdate_to_datetime(ref_dt_np)
        target = opponent_info['total_run'].values[0]
        win=0
        if team_info['winner'].values[0]==team:
            win=1


        team_player_list = list()
        for bi in range(11):
            batsman = team_info['batsman_'+str(bi+1)].values[0].strip()
            if batsman == 'not_batted':
                break
            else:
                team_player_list.append(batsman)
        opponent_player_list = list()
        for boi in range(11):
            bowler = opponent_info['bowler_' + str(boi + 1)].values[0].strip()
            if bowler == 'not_bowled':
                break
            else:
                opponent_player_list.append(bowler)


        try:
            feature_dict = fe.get_instance_feature_dict(team, opponent, location,
                                                        team_player_list, opponent_player_list,
                                                        ref_date)
            feature_dict['target_score'] = target
            no_of_basman = no_of_basman+len(team_player_list)

            if ref_date<test_start_dt:
                feature_list_train.append(feature_dict)
                result_list_train.append(win)
            else:
                feature_list_test.append(feature_dict)
                result_list_test.append(win)
        except Exception as ex:
            print(ex, ' for ',team, opponent, location, ' on ',ref_date.date() )

    print('mean no of batsman - ',no_of_basman/index)
    train_y = np.stack(result_list_train)
    test_y = np.stack(result_list_test)

    train_x = np.array(pd.DataFrame(feature_list_train).drop(columns=['team','opponent','location']))
    test_x  = np.array(pd.DataFrame(feature_list_test).drop(columns=['team','opponent','location']))
    cols = list(pd.DataFrame(feature_list_train).drop(columns=['team','opponent','location']).columns)

    pickle.dump(train_x,open(os.path.join(TRAIN_TEST_DIR,second_innings_base_train_x),'wb'))
    pickle.dump(train_y, open(os.path.join(TRAIN_TEST_DIR,second_innings_base_train_y), 'wb'))
    pickle.dump(cols, open(os.path.join(outil.DEV_DIR, second_innings_base_columns), 'wb'))

    outil.create_meta_info_entry('second_innings_base_train_xy', train_start,
                                 str(cricutil.substract_day_as_datetime(test_start_dt, 1).date()),
                                 file_list=[second_innings_base_train_x,
                                            second_innings_base_train_y,
                                            second_innings_base_columns])

    pickle.dump(test_x, open(os.path.join(TRAIN_TEST_DIR, second_innings_base_test_x), 'wb'))
    pickle.dump(test_y, open(os.path.join(TRAIN_TEST_DIR, second_innings_base_test_y), 'wb'))

    outil.create_meta_info_entry('second_innings_base_test_xy', str(test_start_dt.date()),
                                 str(test_end_dt.date()),
                                 file_list=[second_innings_base_test_x,
                                            second_innings_base_test_y])

    # print(pd.DataFrame(feature_list_train))
#
#
# def create_first_innings_train_test(train_start,test_start,test_end=None):
#     if not os.path.isdir(TRAIN_TEST_DIR):
#         os.makedirs(TRAIN_TEST_DIR)
#
#     outil.use_model_from('dev')
#     train_start_dt = cricutil.str_to_date_time(train_start)
#     test_start_dt = cricutil.str_to_date_time(test_start)
#     if test_end is None:
#         test_end_dt = cricutil.today_as_date_time()
#     else:
#         test_end_dt = cricutil.str_to_date_time(test_end)
#
#     overall_start = train_start_dt
#     overall_end = test_end_dt
#     match_list_df = cricutil.read_csv_with_date(dl.CSV_LOAD_LOCATION + os.sep + 'match_list.csv')
#     match_list_df = match_list_df[(match_list_df['date'] >= overall_start) & \
#                                   (match_list_df['date'] <= overall_end)]
#     match_stats_df = pd.read_csv(dl.CSV_LOAD_LOCATION + os.sep + 'match_stats.csv')
#     match_list_df = match_list_df.merge(match_stats_df, how='inner', on='match_id')
#     match_id_list = list(match_list_df['match_id'].unique())
#
#     feature_list_train = []
#     target_list_train = []
#
#     feature_list_test =[]
#     target_list_test = []
#
#     for index,match_id in tqdm(enumerate(match_id_list)):
#         match_info = match_list_df[match_list_df['match_id']==match_id]
#         team_info = match_info[match_info['first_innings']==match_info['team_statistics']]
#         opponent_info = match_info[match_info['second_innings']==match_info['team_statistics']]
#
#         team = team_info['team_statistics'].values[0].strip()
#         opponent = opponent_info['team_statistics'].values[0].strip()
#         location = team_info['location'].values[0].strip()
#         ref_dt_np = team_info['date'].values[0]
#         ref_date = cricutil.npdate_to_datetime(ref_dt_np)
#         runs_scored = team_info['total_run'].values[0]
#
#         team_player_list = list()
#         for bi in range(11):
#             batsman = team_info['batsman_'+str(bi+1)].values[0].strip()
#             if batsman == 'not_batted':
#                 break
#             else:
#                 team_player_list.append(batsman)
#         opponent_player_list = list()
#         for boi in range(11):
#             bowler = opponent_info['bowler_' + str(boi + 1)].values[0].strip()
#             if bowler == 'not_bowled':
#                 break
#             else:
#                 opponent_player_list.append(bowler)
#
#         try:
#             feature_vector = fe.get_first_innings_feature_embedding_vector(team, opponent, location,
#                                                                             team_player_list, opponent_player_list,
#                                                                             ref_date=ref_date)
#
#             if ref_date<test_start_dt:
#                 feature_list_train.append(feature_vector)
#                 target_list_train.append(runs_scored)
#             else:
#                 feature_list_test.append(feature_vector)
#                 target_list_test.append(runs_scored)
#         except Exception as ex:
#             print(ex, ' for ',team, opponent, location, ' on ',ref_date.date() )
#
#     #print("pre-scaled values \n",np.stack(feature_list_train))
#     train_x = np.stack(feature_list_train)
#     train_y = np.stack(target_list_train)
#
#     test_x = np.stack(feature_list_test)
#     test_y = np.stack(target_list_test)
#
#     # pickle train_x, train_y,test_x,test_y,scaler
#     pickle.dump(train_x,open(os.path.join(TRAIN_TEST_DIR,first_innings_train_x),'wb'))
#     pickle.dump(train_y, open(os.path.join(TRAIN_TEST_DIR, first_innings_train_y), 'wb'))
#
#     pickle.dump(test_x, open(os.path.join(TRAIN_TEST_DIR, first_innings_test_x), 'wb'))
#     pickle.dump(test_y, open(os.path.join(TRAIN_TEST_DIR, first_innings_test_y), 'wb'))
#
#
#     outil.create_meta_info_entry('first_innings_train_xy', train_start,
#                                  str(cricutil.substract_day_as_datetime(test_start_dt, 1).date()),
#                                  file_list=[first_innings_train_x,
#                                             first_innings_train_y])
#
#     outil.create_meta_info_entry('first_innings_test_xy', str(test_start_dt.date()),
#                                  str(test_end_dt.date()),
#                                  file_list=[first_innings_test_x,
#                                             first_innings_test_y])
#
#
# def create_second_innings_train_test(train_start,test_start,test_end=None):
#     if not os.path.isdir(TRAIN_TEST_DIR):
#         os.makedirs(TRAIN_TEST_DIR)
#
#     outil.use_model_from('dev')
#     train_start_dt = cricutil.str_to_date_time(train_start)
#     test_start_dt = cricutil.str_to_date_time(test_start)
#     if test_end is None:
#         test_end_dt = cricutil.today_as_date_time()
#     else:
#         test_end_dt = cricutil.str_to_date_time(test_end)
#
#     overall_start = train_start_dt
#     overall_end = test_end_dt
#     match_list_df = cricutil.read_csv_with_date(dl.CSV_LOAD_LOCATION + os.sep + 'match_list.csv')
#     match_list_df = match_list_df[(match_list_df['date'] >= overall_start) & \
#                                   (match_list_df['date'] <= overall_end)]
#     match_stats_df = pd.read_csv(dl.CSV_LOAD_LOCATION + os.sep + 'match_stats.csv')
#     match_list_df = match_list_df.merge(match_stats_df, how='inner', on='match_id')
#     match_id_list = list(match_list_df['match_id'].unique())
#
#     feature_list_train = []
#     result_list_train = []
#
#     feature_list_test =[]
#     result_list_test = []
#
#     for index,match_id in tqdm(enumerate(match_id_list)):
#         match_info = match_list_df[match_list_df['match_id']==match_id]
#         team_info = match_info[match_info['second_innings']==match_info['team_statistics']]
#         opponent_info = match_info[match_info['first_innings']==match_info['team_statistics']]
#
#         team = team_info['team_statistics'].values[0].strip()
#         opponent = opponent_info['team_statistics'].values[0].strip()
#         location = team_info['location'].values[0].strip()
#         ref_dt_np = team_info['date'].values[0]
#         ref_date = cricutil.npdate_to_datetime(ref_dt_np)
#         target = opponent_info['total_run'].values[0]
#         win=0
#         if team_info['winner'].values[0]==team:
#             win=1
#
#
#         team_player_list = list()
#         for bi in range(11):
#             batsman = team_info['batsman_'+str(bi+1)].values[0].strip()
#             if batsman == 'not_batted':
#                 break
#             else:
#                 team_player_list.append(batsman)
#         opponent_player_list = list()
#         for boi in range(11):
#             bowler = opponent_info['bowler_' + str(boi + 1)].values[0].strip()
#             if bowler == 'not_bowled':
#                 break
#             else:
#                 opponent_player_list.append(bowler)
#
#
#         try:
#             feature_vec = fe.get_second_innings_feature_embedding_vector(target, team, opponent, location,
#                                                                          team_player_list, opponent_player_list,
#                                                                          ref_date=ref_date)
#
#             if ref_date<test_start_dt:
#                 feature_list_train.append(feature_vec)
#                 result_list_train.append(win)
#             else:
#                 feature_list_test.append(feature_vec)
#                 result_list_test.append(win)
#         except Exception as ex:
#             print(ex, ' for ',team, opponent, location, ' on ',ref_date.date() )
#
#     train_x = np.stack(feature_list_train)
#     train_y = np.stack(result_list_train)
#
#     test_x = np.stack(feature_list_test)
#     test_y = np.stack(result_list_test)
#
#     # pickle train_x, train_y,test_x,test_y,scaler
#     pickle.dump(train_x, open(os.path.join(TRAIN_TEST_DIR, second_innings_train_x), 'wb'))
#     pickle.dump(train_y, open(os.path.join(TRAIN_TEST_DIR, second_innings_train_y), 'wb'))
#
#     pickle.dump(test_x, open(os.path.join(TRAIN_TEST_DIR, second_innings_test_x), 'wb'))
#     pickle.dump(test_y, open(os.path.join(TRAIN_TEST_DIR, second_innings_test_y), 'wb'))
#
#     outil.create_meta_info_entry('second_innings_train_xy', train_start,
#                                  str(cricutil.substract_day_as_datetime(test_start_dt, 1).date()),
#                                  file_list=[second_innings_train_x,
#                                             second_innings_train_y])
#
#     outil.create_meta_info_entry('second_innings_test_xy', str(test_start_dt.date()),
#                                  str(test_end_dt.date()),
#                                  file_list=[second_innings_test_x,
#                                             second_innings_test_y])
#
#
# def create_batsman_runs_train_test(train_start,test_start,test_end=None):
#     if not os.path.isdir(TRAIN_TEST_DIR):
#         os.makedirs(TRAIN_TEST_DIR)
#
#     outil.use_model_from('dev')
#     train_start_dt = cricutil.str_to_date_time(train_start)
#     test_start_dt = cricutil.str_to_date_time(test_start)
#     if test_end is None:
#         test_end_dt = cricutil.today_as_date_time()
#     else:
#         test_end_dt = cricutil.str_to_date_time(test_end)
#
#     overall_start = train_start_dt
#     overall_end = test_end_dt
#     print('match_list ')
#     match_list_df = cricutil.read_csv_with_date(dl.CSV_LOAD_LOCATION + os.sep + 'match_list.csv')
#     match_list_df = match_list_df[(match_list_df['date'] >= overall_start) & \
#                                   (match_list_df['date'] <= overall_end)]
#     match_stats_df = pd.read_csv(dl.CSV_LOAD_LOCATION + os.sep + 'match_stats.csv')
#     match_list_df = match_list_df.merge(match_stats_df, how='inner', on='match_id')
#     match_id_list = list(match_list_df['match_id'].unique())
#
#     feature_list_train = []
#     target_list_train = []
#
#     feature_list_test =[]
#     target_list_test = []
#
#     for index,match_id in tqdm(enumerate(match_id_list)):
#
#         for team_innings,opponent_innings in zip(['first_innings','second_innings'],['second_innings','first_innings']):
#             match_info = match_list_df[match_list_df['match_id']==match_id]
#             team_info = match_info[match_info[team_innings]==match_info['team_statistics']]
#             opponent_info = match_info[match_info[opponent_innings]==match_info['team_statistics']]
#
#             team = team_info['team_statistics'].values[0].strip()
#             opponent = opponent_info['team_statistics'].values[0].strip()
#             location = team_info['location'].values[0].strip()
#             ref_dt_np = team_info['date'].values[0]
#             ref_date = cricutil.npdate_to_datetime(ref_dt_np)
#             opponent_player_list = list()
#             for boi in range(11):
#                 bowler = opponent_info['bowler_' + str(boi + 1)].values[0].strip()
#                 if bowler == 'not_bowled':
#                     break
#                 else:
#                     opponent_player_list.append(bowler)
#
#             for bi in range(11):
#
#                 batsman = team_info['batsman_'+str(bi+1)].values[0].strip()
#                 if batsman == 'not_batted':
#                     break
#                 else:
#                     try:
#                         feature_vector = fe.get_batsman_features_with_embedding(batsman,bi+1,opponent_player_list,team,opponent,location,ref_date=ref_date)
#                         runs_scored = team_info['batsman_'+str(bi+1)+'_runs'].values[0]
#                         if ref_date<test_start_dt:
#                             feature_list_train.append(feature_vector)
#                             target_list_train.append(runs_scored)
#                         else:
#                             feature_list_test.append(feature_vector)
#                             target_list_test.append(runs_scored)
#                     except Exception as ex:
#                         print(ex, ' for ',batsman, team, opponent, location, ' on ',ref_date.date() )
#                         #raise ex
#
#     #print("pre-scaled values \n",np.stack(feature_list_train))
#     train_x = np.stack(feature_list_train)
#     train_y = np.stack(target_list_train)
#
#     test_x = np.stack(feature_list_test)
#     test_y = np.stack(target_list_test)
#
#     # pickle train_x, train_y,test_x,test_y,scaler
#     pickle.dump(train_x,open(os.path.join(TRAIN_TEST_DIR, batsman_runs_train_x),'wb'))
#     pickle.dump(train_y, open(os.path.join(TRAIN_TEST_DIR, batsman_runs_train_y), 'wb'))
#
#     pickle.dump(test_x, open(os.path.join(TRAIN_TEST_DIR, batsman_runs_test_x), 'wb'))
#     pickle.dump(test_y, open(os.path.join(TRAIN_TEST_DIR, batsman_runs_test_y), 'wb'))
#
#     outil.create_meta_info_entry('batsman_runs_train_xy', train_start,
#                                  str(cricutil.substract_day_as_datetime(test_start_dt, 1).date()),
#                                  file_list=[batsman_runs_train_x,
#                                             batsman_runs_train_y])
#
#     outil.create_meta_info_entry('batsman_runs_test_xy', str(test_start_dt.date()),
#                                  str(test_end_dt.date()),
#                                  file_list=[batsman_runs_test_x,
#                                             batsman_runs_test_y])



def create_adversarial_train_test(train_start,test_start,test_end=None,encoding_source='dev',include_not_batted=False):
    if not os.path.isdir(TRAIN_TEST_DIR):
        os.makedirs(TRAIN_TEST_DIR)

    if encoding_source == 'production':
        encoding_dir = outil.PROD_DIR
    else:
        encoding_dir = outil.DEV_DIR

    train_start_dt = cricutil.str_to_date_time(train_start)
    test_start_dt = cricutil.str_to_date_time(test_start)
    if test_end is None:
        test_end_dt = cricutil.today_as_date_time()
    else:
        test_end_dt = cricutil.str_to_date_time(test_end)

    overall_start = train_start_dt
    overall_end = test_end_dt
    match_list_df = cricutil.read_csv_with_date(dl.CSV_LOAD_LOCATION + os.sep + 'match_list.csv')
    match_list_df = match_list_df[(match_list_df['date'] >= overall_start) & \
                                  (match_list_df['date'] <= overall_end)]
    match_stats_df = pd.read_csv(dl.CSV_LOAD_LOCATION + os.sep + 'match_stats.csv')
    match_list_df = match_list_df.merge(match_stats_df,how='inner',on='match_id')

    no_of_matches = match_list_df.shape[0]
    location_enc_map_for_batsman = pickle.load(open(os.path.join(encoding_dir, outil.LOC_ENCODING_MAP), 'rb'))
    batsman_enc_map = pickle.load(open(os.path.join(encoding_dir, outil.BATSMAN_ENCODING_MAP), 'rb'))
    bowler_enc_map = pickle.load(open(os.path.join(encoding_dir, outil.BOWLER_ENCODING_MAP), 'rb'))

    batsman_oh_list_train = []
    position_oh_list_train = []
    location_oh_list_train = []
    bowler_oh_list_train = []
    runs_scored_list_train = []

    batsman_oh_list_test = []
    position_oh_list_test = []
    location_oh_list_test = []
    bowler_oh_list_test = []
    runs_scored_list_test = []

    for index in tqdm(range(no_of_matches)):
        if match_list_df.iloc[index]['first_innings']==match_list_df.iloc[index]['team_statistics']:
            opponent = match_list_df.iloc[index]['second_innings'].strip()
            club = match_list_df.iloc[index]['first_innings'].strip()
        else:
            opponent = match_list_df.iloc[index]['first_innings'].strip()
            club = match_list_df.iloc[index]['second_innings'].strip()
        location = match_list_df.iloc[index]['location'].strip()
        match_id = match_list_df.iloc[index]['match_id']
        match_details = pd.read_csv(dl.CSV_LOAD_LOCATION + os.sep + str(match_id)+'.csv')

        date = match_list_df.iloc[index]['date']

        if location not in location_enc_map_for_batsman:
            location = 'unknown'

        location_oh = location_enc_map_for_batsman[location]

        team_batsman_list = list(match_details[(match_details['team']==club)]['batsman'].unique())
        opponent_bowler_list = list(match_details[(match_details['team']==club)]['bowler'].unique())

        for batsman in team_batsman_list:
            for bowler in opponent_bowler_list:
                try:
                    batsman_oh = batsman_enc_map[batsman.strip()]
                    bowler_oh = bowler_enc_map[bowler.strip()]
                    position_oh = fe.get_oh_pos(team_batsman_list.index(batsman)+1)

                    runs_scored = match_details[(match_details['team']==club) &\
                                                (match_details['batsman']==batsman) &\
                                                (match_details['bowler']==bowler)]['scored_runs'].sum()

                    if date<test_start_dt:
                        batsman_oh_list_train.append(batsman_oh)
                        position_oh_list_train.append(position_oh)
                        location_oh_list_train.append(location_oh)
                        bowler_oh_list_train.append(bowler_oh)
                        runs_scored_list_train.append(runs_scored)
                    else:
                        batsman_oh_list_test.append(batsman_oh)
                        position_oh_list_test.append(position_oh)
                        location_oh_list_test.append(location_oh)
                        bowler_oh_list_test.append(bowler_oh)
                        runs_scored_list_test.append(runs_scored)
                except Exception as ex:
                    print(ex,' for ',batsman,bowler,club,opponent,' at ',location,' on ',date)
                    #raise ex


    batsman_oh_train_x = np.stack(batsman_oh_list_train)
    position_oh_train_x = np.stack(position_oh_list_train)
    location_oh_train_x = np.stack(location_oh_list_train)
    bowler_oh_train_x = np.stack(bowler_oh_list_train)
    runs_scored_train_y = np.stack(runs_scored_list_train)

    batsman_oh_test_x = np.stack(batsman_oh_list_test)
    position_oh_test_x = np.stack(position_oh_list_test)
    location_oh_test_x = np.stack(location_oh_list_test)
    bowler_oh_test_x = np.stack(bowler_oh_list_test)
    runs_scored_test_y = np.stack(runs_scored_list_test)

    pickle.dump(batsman_oh_train_x, open(os.path.join(TRAIN_TEST_DIR, adversarial_feature_batsman_oh_train_x), 'wb'))
    pickle.dump(position_oh_train_x, open(os.path.join(TRAIN_TEST_DIR, adversarial_feature_position_oh_train_x), 'wb'))
    pickle.dump(location_oh_train_x, open(os.path.join(TRAIN_TEST_DIR, adversarial_feature_location_oh_train_x), 'wb'))
    pickle.dump(bowler_oh_train_x, open(os.path.join(TRAIN_TEST_DIR, adversarial_feature_bowler_oh_train_x), 'wb'))
    pickle.dump(runs_scored_train_y, open(os.path.join(TRAIN_TEST_DIR, adversarial_feature_runs_scored_train_y), 'wb'))

    outil.create_meta_info_entry('adversarial_train_xy', train_start,
                                 str(cricutil.substract_day_as_datetime(test_start_dt, 1).date()),
                                 file_list=[adversarial_feature_batsman_oh_train_x,
                                            adversarial_feature_position_oh_train_x,
                                            adversarial_feature_location_oh_train_x,
                                            adversarial_feature_bowler_oh_train_x,
                                            adversarial_feature_runs_scored_train_y])

    pickle.dump(batsman_oh_test_x, open(os.path.join(TRAIN_TEST_DIR, adversarial_feature_batsman_oh_test_x), 'wb'))
    pickle.dump(position_oh_test_x, open(os.path.join(TRAIN_TEST_DIR, adversarial_feature_position_oh_test_x), 'wb'))
    pickle.dump(location_oh_test_x, open(os.path.join(TRAIN_TEST_DIR, adversarial_feature_location_oh_test_x), 'wb'))
    pickle.dump(bowler_oh_test_x, open(os.path.join(TRAIN_TEST_DIR, adversarial_feature_bowler_oh_test_x), 'wb'))
    pickle.dump(runs_scored_test_y, open(os.path.join(TRAIN_TEST_DIR, adversarial_feature_runs_scored_test_y), 'wb'))

    outil.create_meta_info_entry('adversarial_test_xy', test_start, str(test_end_dt.date()),
                                 file_list=[adversarial_feature_batsman_oh_test_x,
                                            adversarial_feature_position_oh_test_x,
                                            adversarial_feature_location_oh_test_x,
                                            adversarial_feature_bowler_oh_test_x,
                                            adversarial_feature_runs_scored_test_y])


def create_adversarial_first_innings_train_test(train_start,test_start,test_end=None):
    if not os.path.isdir(TRAIN_TEST_DIR):
        os.makedirs(TRAIN_TEST_DIR)

    outil.use_model_from('dev')
    train_start_dt = cricutil.str_to_date_time(train_start)
    test_start_dt = cricutil.str_to_date_time(test_start)
    if test_end is None:
        test_end_dt = cricutil.today_as_date_time()
    else:
        test_end_dt = cricutil.str_to_date_time(test_end)

    overall_start = train_start_dt
    overall_end = test_end_dt
    match_list_df = cricutil.read_csv_with_date(dl.CSV_LOAD_LOCATION + os.sep + 'match_list.csv')
    match_list_df = match_list_df[(match_list_df['date'] >= overall_start) & \
                                  (match_list_df['date'] <= overall_end)]
    match_stats_df = pd.read_csv(dl.CSV_LOAD_LOCATION + os.sep + 'match_stats.csv')
    match_list_df = match_list_df.merge(match_stats_df, how='inner', on='match_id')
    match_id_list = list(match_list_df['match_id'].unique())

    feature_list_train = []
    target_list_train = []

    feature_list_test =[]
    target_list_test = []

    for index,match_id in tqdm(enumerate(match_id_list)):
        match_info = match_list_df[match_list_df['match_id']==match_id]
        team_info = match_info[match_info['first_innings']==match_info['team_statistics']]
        opponent_info = match_info[match_info['second_innings']==match_info['team_statistics']]

        team = team_info['team_statistics'].values[0].strip()
        opponent = opponent_info['team_statistics'].values[0].strip()
        location = team_info['location'].values[0].strip()
        ref_dt_np = team_info['date'].values[0]
        ref_date = cricutil.npdate_to_datetime(ref_dt_np)
        runs_scored = team_info['total_run'].values[0]

        team_player_list = list()
        for bi in range(11):
            batsman = team_info['batsman_'+str(bi+1)].values[0].strip()
            if batsman == 'not_batted':
                break
            else:
                team_player_list.append(batsman)
        opponent_player_list = list()
        for boi in range(11):
            bowler = opponent_info['bowler_' + str(boi + 1)].values[0].strip()
            if bowler == 'not_bowled':
                break
            else:
                opponent_player_list.append(bowler)

        try:
            feature_vector = fe.get_adversarial_first_innings_feature_vector(team, opponent, location,
                                                                            team_player_list, opponent_player_list,
                                                                            ref_date=ref_date)

            if ref_date<test_start_dt:
                feature_list_train.append(feature_vector)
                target_list_train.append(runs_scored)
            else:
                feature_list_test.append(feature_vector)
                target_list_test.append(runs_scored)
        except Exception as ex:
            print(ex, ' for ',team, opponent, location, ' on ',ref_date.date() )

    #print("pre-scaled values \n",np.stack(feature_list_train))
    train_x = np.stack(feature_list_train)
    train_y = np.stack(target_list_train)

    test_x = np.stack(feature_list_test)
    test_y = np.stack(target_list_test)

    # pickle train_x, train_y,test_x,test_y,scaler
    pickle.dump(train_x,open(os.path.join(TRAIN_TEST_DIR,adversarial_first_innings_train_x),'wb'))
    pickle.dump(train_y, open(os.path.join(TRAIN_TEST_DIR, adversarial_first_innings_train_y), 'wb'))

    pickle.dump(test_x, open(os.path.join(TRAIN_TEST_DIR, adversarial_first_innings_test_x), 'wb'))
    pickle.dump(test_y, open(os.path.join(TRAIN_TEST_DIR, adversarial_first_innings_test_y), 'wb'))


    outil.create_meta_info_entry('adversarial_first_innings_train_xy', train_start,
                                 str(cricutil.substract_day_as_datetime(test_start_dt, 1).date()),
                                 file_list=[adversarial_first_innings_train_x,
                                            adversarial_first_innings_train_y])

    outil.create_meta_info_entry('adversarial_first_innings_test_xy', str(test_start_dt.date()),
                                 str(test_end_dt.date()),
                                 file_list=[adversarial_first_innings_test_x,
                                            adversarial_first_innings_test_y])

def create_adversarial_second_innings_train_test(train_start,test_start,test_end=None):
    if not os.path.isdir(TRAIN_TEST_DIR):
        os.makedirs(TRAIN_TEST_DIR)

    outil.use_model_from('dev')
    train_start_dt = cricutil.str_to_date_time(train_start)
    test_start_dt = cricutil.str_to_date_time(test_start)
    if test_end is None:
        test_end_dt = cricutil.today_as_date_time()
    else:
        test_end_dt = cricutil.str_to_date_time(test_end)

    overall_start = train_start_dt
    overall_end = test_end_dt
    match_list_df = cricutil.read_csv_with_date(dl.CSV_LOAD_LOCATION + os.sep + 'match_list.csv')
    match_list_df = match_list_df[(match_list_df['date'] >= overall_start) & \
                                  (match_list_df['date'] <= overall_end)]
    match_stats_df = pd.read_csv(dl.CSV_LOAD_LOCATION + os.sep + 'match_stats.csv')
    match_list_df = match_list_df.merge(match_stats_df, how='inner', on='match_id')
    match_id_list = list(match_list_df['match_id'].unique())

    feature_list_train = []
    target_list_train = []

    feature_list_test =[]
    target_list_test = []

    for index,match_id in tqdm(enumerate(match_id_list)):
        match_info = match_list_df[match_list_df['match_id']==match_id]
        team_info = match_info[match_info['second_innings']==match_info['team_statistics']]
        opponent_info = match_info[match_info['first_innings']==match_info['team_statistics']]

        team = team_info['team_statistics'].values[0].strip()
        opponent = opponent_info['team_statistics'].values[0].strip()
        winner = team_info['winner'].values[0].strip()
        location = team_info['location'].values[0].strip()
        ref_dt_np = team_info['date'].values[0]
        ref_date = cricutil.npdate_to_datetime(ref_dt_np)
        runs_scored = opponent_info['total_run'].values[0]

        if team ==winner:
            win = 1
        else:
            win = 0

        team_player_list = list()
        for bi in range(11):
            batsman = team_info['batsman_'+str(bi+1)].values[0].strip()
            if batsman == 'not_batted':
                break
            else:
                team_player_list.append(batsman)
        opponent_player_list = list()
        for boi in range(11):
            bowler = opponent_info['bowler_' + str(boi + 1)].values[0].strip()
            if bowler == 'not_bowled':
                break
            else:
                opponent_player_list.append(bowler)

        try:
            feature_vector = fe.get_adversarial_second_innings_feature_vector(runs_scored,team, opponent, location,
                                                                            team_player_list, opponent_player_list,
                                                                            ref_date=ref_date)

            # updated_feature_vector = np.concatenate([feature_vector,[runs_scored]])

            if ref_date<test_start_dt:
                feature_list_train.append(feature_vector)
                target_list_train.append(win)
            else:
                feature_list_test.append(feature_vector)
                target_list_test.append(win)
        except Exception as ex:
            print(ex, ' for ',team, opponent, location, ' on ',ref_date.date() )
            #raise ex

    #print("pre-scaled values \n",np.stack(feature_list_train))
    train_x = np.stack(feature_list_train)
    train_y = np.stack(target_list_train)

    test_x = np.stack(feature_list_test)
    test_y = np.stack(target_list_test)

    # pickle train_x, train_y,test_x,test_y,scaler
    pickle.dump(train_x,open(os.path.join(TRAIN_TEST_DIR,adversarial_second_innings_train_x),'wb'))
    pickle.dump(train_y, open(os.path.join(TRAIN_TEST_DIR, adversarial_second_innings_train_y), 'wb'))

    pickle.dump(test_x, open(os.path.join(TRAIN_TEST_DIR, adversarial_second_innings_test_x), 'wb'))
    pickle.dump(test_y, open(os.path.join(TRAIN_TEST_DIR, adversarial_second_innings_test_y), 'wb'))


    outil.create_meta_info_entry('adversarial_second_innings_train_xy', train_start,
                                 str(cricutil.substract_day_as_datetime(test_start_dt, 1).date()),
                                 file_list=[adversarial_second_innings_train_x,
                                            adversarial_second_innings_train_y])

    outil.create_meta_info_entry('adversarial_second_innings_test_xy', str(test_start_dt.date()),
                                 str(test_end_dt.date()),
                                 file_list=[adversarial_second_innings_test_x,
                                            adversarial_second_innings_test_y])


@click.group()
def traintest():
    pass

# @traintest.command()
# @click.option('--train_start', help='start date for train data (YYYY-mm-dd)',required=True)
# @click.option('--test_start', help='start date for test data (YYYY-mm-dd)',required=True)
# @click.option('--test_end', help='end date for test (YYYY-mm-dd)')
# @click.option('--encoding_source', help='which enviornment to read from for one hot encoding(dev/production)',
#               default='dev')
# def country_embedding(train_start,test_start,test_end=None,encoding_source='dev'):
#     create_country_embedding_train_test(train_start,
#                                         test_start,
#                                         test_end=test_end,
#                                         encoding_source=encoding_source)
#
# @traintest.command()
# @click.option('--train_start', help='start date for train data (YYYY-mm-dd)',required=True)
# @click.option('--test_start', help='start date for test data (YYYY-mm-dd)',required=True)
# @click.option('--test_end', help='end date for test (YYYY-mm-dd)')
# @click.option('--encoding_source', help='which environment to read from for one hot encoding(dev/production)',
#               default='dev')
# @click.option('--include_not_batted', help='whether to create an encoding for not_batted)',
#               default=False)
# def batsman_embedding(train_start,test_start,test_end,encoding_source,include_not_batted):
#     create_batsman_embedding_train_test(train_start,
#                                         test_start,
#                                         test_end=test_end,
#                                         encoding_source=encoding_source,
#                                         include_not_batted=include_not_batted)

@traintest.command()
@click.option('--train_start', help='start date for train data (YYYY-mm-dd)',required=True)
@click.option('--test_start', help='start date for test data (YYYY-mm-dd)',required=True)
@click.option('--test_end', help='end date for test (YYYY-mm-dd)')
@click.option('--encoding_source', help='which environment to read from for one hot encoding(dev/production)',
              default='dev')
@click.option('--include_not_batted', help='whether to create an encoding for not_batted)',
              default=False)
def adversarial(train_start,test_start,test_end,encoding_source,include_not_batted):
    create_adversarial_train_test(train_start,
                                  test_start,
                                  test_end=test_end,
                                  encoding_source=encoding_source,
                                  include_not_batted=include_not_batted)


@traintest.command()
@click.option('--train_start', help='start date for train data (YYYY-mm-dd)',required=True)
@click.option('--test_start', help='start date for test data (YYYY-mm-dd)',required=True)
@click.option('--test_end', help='end date for test (YYYY-mm-dd)')
def first_innings_base(train_start, test_start, test_end):
    create_first_innings_base_train_test(train_start, test_start, test_end=test_end)


@traintest.command()
@click.option('--train_start', help='start date for train data (YYYY-mm-dd)',required=True)
@click.option('--test_start', help='start date for test data (YYYY-mm-dd)',required=True)
@click.option('--test_end', help='end date for test (YYYY-mm-dd)')
def second_innings_base(train_start, test_start, test_end):
    create_second_innings_base_train_test(train_start, test_start, test_end=test_end)

#
# @traintest.command()
# @click.option('--train_start', help='start date for train data (YYYY-mm-dd)',required=True)
# @click.option('--test_start', help='start date for test data (YYYY-mm-dd)',required=True)
# @click.option('--test_end', help='end date for test (YYYY-mm-dd)')
# def first_innings(train_start, test_start, test_end):
#     create_first_innings_train_test(train_start, test_start, test_end=test_end)
#
# @traintest.command()
# @click.option('--train_start', help='start date for train data (YYYY-mm-dd)',required=True)
# @click.option('--test_start', help='start date for test data (YYYY-mm-dd)',required=True)
# @click.option('--test_end', help='end date for test (YYYY-mm-dd)')
# def second_innings(train_start, test_start, test_end):
#     create_second_innings_train_test(train_start, test_start, test_end=test_end)
#
# @traintest.command()
# @click.option('--train_start', help='start date for train data (YYYY-mm-dd)',required=True)
# @click.option('--test_start', help='start date for test data (YYYY-mm-dd)',required=True)
# @click.option('--test_end', help='end date for test (YYYY-mm-dd)')
# def batsman_runs(train_start, test_start, test_end):
#     create_batsman_runs_train_test(train_start, test_start, test_end=test_end)

@traintest.command()
@click.option('--train_start', help='start date for train data (YYYY-mm-dd)',required=True)
@click.option('--test_start', help='start date for test data (YYYY-mm-dd)',required=True)
@click.option('--test_end', help='end date for test (YYYY-mm-dd)')
def adversarial_first_innings(train_start, test_start, test_end):
    create_adversarial_first_innings_train_test(train_start, test_start, test_end=test_end)

@traintest.command()
@click.option('--train_start', help='start date for train data (YYYY-mm-dd)',required=True)
@click.option('--test_start', help='start date for test data (YYYY-mm-dd)',required=True)
@click.option('--test_end', help='end date for test (YYYY-mm-dd)')
def adversarial_second_innings(train_start, test_start, test_end):
    create_adversarial_second_innings_train_test(train_start, test_start, test_end=test_end)



if __name__=="__main__":
    traintest()
