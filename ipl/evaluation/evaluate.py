from datetime import datetime
from ipl.data_loader import data_loader as dl
import pandas as pd
import os
from tqdm import tqdm
from ipl.feature_engg import util as cricutil,feature_extractor
from ipl.model_util import odi_util as outil
import numpy as np
import pickle
from sklearn.metrics import mean_absolute_error,mean_squared_error,accuracy_score
import click
from odi.retrain import create_train_test as ctt
import tensorflow as tf


tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def mape(y_true,y_predict):
    return np.sum((np.abs(y_true-y_predict)/y_true)*100)/len(y_true)


# def evaluate_first_innings(from_date, to_date, environment='production',use_emb=True,model='team'):
#     outil.use_model_from(environment)
#
#     predictor = None
#     if not use_emb:
#         try:
#
#             predictor = pickle.load(
#                 open(os.path.join(outil.MODEL_DIR, outil.FIRST_INNINGS_MODEL_BASE), 'rb'))
#         except:
#             raise Exception("use_emb: " + str(use_emb) + " option related files not available")
#     else:
#         try:
#             if model == 'team':
#                 predictor = pickle.load(open(outil.MODEL_DIR + os.sep + outil.FIRST_INNINGS_MODEL, 'rb'))
#             else:
#                 predictor = pickle.load(open(outil.MODEL_DIR + os.sep + outil.ADVERSARIAL_FIRST_INNINGS, 'rb'))
#         except:
#             print(' Provided First innings model in option not available')
#
#
#     start_date = datetime.strptime(from_date, '%Y-%m-%d')
#     end_date = datetime.strptime(to_date, '%Y-%m-%d')
#
#     custom_date_parser = lambda x: datetime.strptime(x, "%Y-%m-%d")
#     match_list_df = pd.read_csv(dl.CSV_LOAD_LOCATION+os.sep+'match_list.csv',parse_dates=['date'],date_parser=custom_date_parser)
#     match_list_df = match_list_df[(match_list_df['date']>=start_date) & (match_list_df['date']<=end_date)]
#     match_stats_df = pd.read_csv(dl.CSV_LOAD_LOCATION+os.sep+'match_stats.csv')
#
#     match_list_df=match_list_df.merge(match_stats_df,on='match_id',how='inner')
#
#     match_id_list = list(match_list_df['match_id'].unique())
#     feature_vector_list =[]
#     score_list = []
#
#     for match_id in tqdm(match_id_list):
#
#         first_innings_team = match_list_df[(match_list_df['match_id']==match_id) & \
#                                       (match_list_df['first_innings']==match_list_df['team_statistics'])]
#         second_innings_team = match_list_df[(match_list_df['match_id'] == match_id) & \
#                                       (match_list_df['second_innings'] == match_list_df['team_statistics'])]
#         team = first_innings_team['team_statistics'].values[0]
#         opponent = second_innings_team['team_statistics'].values[0]
#         location = first_innings_team['location'].values[0]
#         ref_date = cricutil.npdate_to_datetime(first_innings_team['date'].values[0])
#         try:
#             team_player_list = []
#             for i in range(11):
#                 player = first_innings_team['batsman_'+str(i+1)].values[0]
#                 if player == 'not_batted':
#                     break
#                 else:
#                     team_player_list.append(player)
#             opponent_player_list = []
#             for i in range(11):
#                 bowler = second_innings_team['bowler_'+str(i+1)].values[0]
#                 if bowler == 'not_bowled':
#                     break
#                 else:
#                     opponent_player_list.append(bowler)
#             #print('============',opponent_player_list)
#             if use_emb:
#                 if model == 'team':
#                     feature_vector = feature_extractor.get_first_innings_feature_embedding_vector(team, opponent, location,\
#                                                                                                  team_player_list,\
#                                                                                                  opponent_player_list,\
#                                                                                                  ref_date=ref_date)
#                 else:
#                     feature_vector = feature_extractor.get_adversarial_first_innings_feature_vector(team, opponent,
#                                                                                                     location, \
#                                                                                                     team_player_list, \
#                                                                                                     opponent_player_list, \
#                                                                                                     ref_date=ref_date)
#             else:
#
#                 feature_vector = feature_extractor.get_first_innings_feature_vector(team, opponent, location,\
#                                                                                     team_player_list, opponent_player_list,\
#                                                                                     ref_date=ref_date)
#
#
#             feature_vector_list.append(feature_vector)
#             score_list.append(first_innings_team['total_run'].values[0])
#         except Exception as ex:
#             print(match_id,': Exception for match between  ',team,' and ',opponent,' on ',ref_date)
#             print(ex)
#
#
#     feature_matrix = np.stack(feature_vector_list)
#     actual_runs = np.stack(score_list)
#
#     predicted_runs = predictor.predict(feature_matrix)
#
#     mape_val=mape(actual_runs, predicted_runs)
#     mae=mean_absolute_error(actual_runs, predicted_runs)
#     mse=mean_squared_error(actual_runs, predicted_runs)
#
#     print('mape :',mape_val)
#     print('mae :',mae)
#     print('mse :',mse)
#     print('total data ',len(actual_runs))
#
#     return mape_val,mae,mse,len(actual_runs)
#
#
# def evaluate_second_innings(from_date, to_date, environment='production',use_emb=True):
#     outil.use_model_from(environment)
#
#     if not use_emb:
#         try:
#             predictor = pickle.load(
#                 open(os.path.join(outil.MODEL_DIR, outil.SECOND_INNINGS_MODEL_BASE), 'rb'))
#         except:
#             raise Exception("use_emb: " + str(use_emb) + " option related files not available")
#     else:
#         predictor = pickle.load(open(outil.MODEL_DIR + os.sep + outil.SECOND_INNINGS_MODEL, 'rb'))
#
#     start_date = datetime.strptime(from_date, '%Y-%m-%d')
#     end_date = datetime.strptime(to_date, '%Y-%m-%d')
#
#     custom_date_parser = lambda x: datetime.strptime(x, "%Y-%m-%d")
#     match_list_df = pd.read_csv(dl.CSV_LOAD_LOCATION + os.sep + 'match_list.csv', parse_dates=['date'],
#                                 date_parser=custom_date_parser)
#     match_list_df = match_list_df[(match_list_df['date'] >= start_date) & (match_list_df['date'] <= end_date)]
#     match_stats_df = pd.read_csv(dl.CSV_LOAD_LOCATION + os.sep + 'match_stats.csv')
#
#     match_list_df = match_list_df.merge(match_stats_df, on='match_id', how='inner')
#
#     match_id_list = list(match_list_df['match_id'].unique())
#     feature_vector_list = []
#     result_list = []
#
#     for match_id in tqdm(match_id_list):
#         first_innings_team = match_list_df[(match_list_df['match_id'] == match_id) & \
#                                            (match_list_df['first_innings'] == match_list_df['team_statistics'])]
#         second_innings_team = match_list_df[(match_list_df['match_id'] == match_id) & \
#                                             (match_list_df['second_innings'] == match_list_df['team_statistics'])]
#         team = second_innings_team['team_statistics'].values[0]
#         opponent = first_innings_team['team_statistics'].values[0]
#         location = first_innings_team['location'].values[0]
#         ref_date = cricutil.npdate_to_datetime(first_innings_team['date'].values[0])
#
#         try:
#             team_player_list = []
#             for i in range(11):
#                 player = second_innings_team['batsman_'+str(i+1)].values[0]
#                 if player == 'not_batted':
#                     break
#                 else:
#                     team_player_list.append(player)
#             opponent_player_list = []
#             for i in range(11):
#                 bowler = first_innings_team['bowler_'+str(i+1)].values[0]
#                 if bowler == 'not_bowled':
#                     break
#                 else:
#                     opponent_player_list.append(bowler)
#             target = first_innings_team['total_run'].values[0]
#
#             if use_emb:
#                 feature_vector = feature_extractor.get_second_innings_feature_embedding_vector(target,team, opponent, location,\
#                                                                                                team_player_list,\
#                                                                                              opponent_player_list,\
#                                                                                              ref_date=ref_date)
#             else:
#                 feature_vector = feature_extractor.get_second_innings_feature_vector(target, team, opponent,\
#                                                                                     location, team_player_list,\
#                                                                                     opponent_player_list, ref_date=ref_date)
#             feature_vector_list.append(feature_vector)
#             if second_innings_team['second_innings'].values[0]==second_innings_team['winner'].values[0]:
#                 result_list.append(1)
#             else:
#                 result_list.append(0)
#
#         except Exception as ex:
#             print(match_id,': Exception for match between  ',team,' and ',opponent,' on ',ref_date)
#             print(ex)
#
#     feature_matrix = np.stack(feature_vector_list)
#     actual_results = np.stack(result_list)
#     predicted_results = predictor.predict(feature_matrix)
#
#     accuracy = accuracy_score(actual_results, predicted_results)
#
#     print('accuracy ',accuracy)
#     print('data size ', len(result_list))
#     return accuracy,len(result_list)
#
#
# def evaluate_combined_innings(from_date, to_date, environment='production',
#                               first_innings_emb = True,second_innings_emb=True,
#                               first_emb_model='team', second_emb_model='team'):
#     outil.use_model_from(environment)
#
#     predictor_second_innings = None
#     if not second_innings_emb:
#         try:
#             predictor_second_innings = pickle.load(
#                 open(os.path.join(outil.MODEL_DIR, outil.SECOND_INNINGS_MODEL_BASE), 'rb'))
#         except:
#             raise Exception("second_innings_emb: " + str(second_innings_emb) + " option related files not available")
#     else:
#         if second_emb_model == 'team':
#             predictor_second_innings = pickle.load(open(outil.MODEL_DIR + os.sep + outil.SECOND_INNINGS_MODEL, 'rb'))
#         else:
#             raise Exception("Provided option for second innings not available")
#
#     predictor_first_innings = None
#     if not first_innings_emb:
#         try:
#
#             predictor_first_innings = pickle.load(
#                 open(os.path.join(outil.MODEL_DIR, outil.FIRST_INNINGS_MODEL_BASE), 'rb'))
#         except:
#             raise Exception("first_innings_emb: " + str(first_innings_emb) + " option related files not available")
#     else:
#         try:
#             if first_emb_model == 'team':
#                 predictor_first_innings = pickle.load(open(outil.MODEL_DIR + os.sep + outil.FIRST_INNINGS_MODEL, 'rb'))
#             else:
#                 predictor_first_innings = pickle.load(open(outil.MODEL_DIR + os.sep + outil.ADVERSARIAL_FIRST_INNINGS, 'rb'))
#         except:
#             print(' Provided First innings model in option not available')
#
#     start_date = datetime.strptime(from_date, '%Y-%m-%d')
#     end_date = datetime.strptime(to_date, '%Y-%m-%d')
#
#     custom_date_parser = lambda x: datetime.strptime(x, "%Y-%m-%d")
#     match_list_df = pd.read_csv(dl.CSV_LOAD_LOCATION+os.sep+'match_list.csv',parse_dates=['date'],date_parser=custom_date_parser)
#     match_list_df = match_list_df[(match_list_df['date']>=start_date) & (match_list_df['date']<=end_date)]
#     match_stats_df = pd.read_csv(dl.CSV_LOAD_LOCATION+os.sep+'match_stats.csv')
#
#     match_list_df=match_list_df.merge(match_stats_df,on='match_id',how='inner')
#
#     match_id_list = list(match_list_df['match_id'].unique())
#     feature_vector_list =[]
#     score_list = []
#
#     first_innings_match_id_list = list()
#     for match_id in tqdm(match_id_list):
#
#         first_innings_team = match_list_df[(match_list_df['match_id']==match_id) & \
#                                       (match_list_df['first_innings']==match_list_df['team_statistics'])]
#         second_innings_team = match_list_df[(match_list_df['match_id'] == match_id) & \
#                                       (match_list_df['second_innings'] == match_list_df['team_statistics'])]
#         team = first_innings_team['team_statistics'].values[0]
#         opponent = second_innings_team['team_statistics'].values[0]
#         location = first_innings_team['location'].values[0]
#         ref_date = cricutil.npdate_to_datetime(first_innings_team['date'].values[0])
#         try:
#             team_player_list = []
#             for i in range(11):
#                 player = first_innings_team['batsman_'+str(i+1)].values[0]
#                 if player == 'not_batted':
#                     break
#                 else:
#                     team_player_list.append(player)
#             opponent_player_list = []
#             for i in range(11):
#                 bowler = second_innings_team['bowler_'+str(i+1)].values[0]
#                 if bowler == 'not_bowled':
#                     break
#                 else:
#                     opponent_player_list.append(bowler)
#
#             if first_innings_emb:
#                 if first_emb_model == 'team':
#                     feature_vector = feature_extractor.get_first_innings_feature_embedding_vector(team, opponent, location,\
#                                                                                                  team_player_list,\
#                                                                                                  opponent_player_list,\
#                                                                                                 ref_date=ref_date)
#                 else:
#                     feature_vector = feature_extractor.get_adversarial_first_innings_feature_vector(team, opponent,
#                                                                                                     location, \
#                                                                                                     team_player_list, \
#                                                                                                     opponent_player_list, \
#                                                                                                     ref_date=ref_date)
#             else:
#
#
#                     feature_vector = feature_extractor.get_first_innings_feature_vector(team, opponent, location,\
#                                                                                         team_player_list, opponent_player_list,\
#                                                                                         ref_date=ref_date)
#
#             feature_vector_list.append(feature_vector)
#             score_list.append(first_innings_team['total_run'].values[0])
#             first_innings_match_id_list.append(match_id)
#         except Exception as ex:
#             print(match_id,': Exception for match between  ',team,' and ',opponent,' on ',ref_date,' at ',location)
#             print(ex)
#
#
#     feature_matrix = np.stack(feature_vector_list)
#     actual_runs = np.stack(score_list)
#     predicted_runs = predictor_first_innings.predict(feature_matrix)
#
#     predicted_run_list = list(predicted_runs)
#
#     mape_val = mape(actual_runs, predicted_runs)
#     mae = mean_absolute_error(actual_runs, predicted_runs)
#     mse = mean_squared_error(actual_runs, predicted_runs)
#     print("=====FIRST INNNINGS METRICS=====")
#     print('mape :', mape_val)
#     print('mae :', mae)
#     print('mse :', mse)
#     print('total data ', len(actual_runs))
#
#     # Predict second Innings
#     result_list = list()
#     second_innings_feature_vector_list = list()
#     for match_id,predicted_first_innings_run in tqdm(zip(first_innings_match_id_list,predicted_run_list)):
#
#         first_innings_team = match_list_df[(match_list_df['match_id'] == match_id) & \
#                                            (match_list_df['first_innings'] == match_list_df['team_statistics'])]
#         second_innings_team = match_list_df[(match_list_df['match_id'] == match_id) & \
#                                             (match_list_df['second_innings'] == match_list_df['team_statistics'])]
#         team = second_innings_team['team_statistics'].values[0]
#         opponent = first_innings_team['team_statistics'].values[0]
#         location = first_innings_team['location'].values[0]
#         ref_date = cricutil.npdate_to_datetime(second_innings_team['date'].values[0])
#
#         try:
#             team_player_list = []
#             for i in range(11):
#                 player = second_innings_team['batsman_'+str(i+1)].values[0]
#                 if player == 'not_batted':
#                     break
#                 else:
#                     team_player_list.append(player)
#             opponent_player_list = []
#             for i in range(11):
#                 bowler = first_innings_team['bowler_'+str(i+1)].values[0]
#                 if bowler == 'not_bowled':
#                     break
#                 else:
#                     opponent_player_list.append(bowler)
#             # target = first_innings_team['total_run'].values[0]
#
#             if second_innings_emb:
#                 feature_vector = feature_extractor.get_second_innings_feature_embedding_vector(predicted_first_innings_run,team, opponent, location,
#                                                                                              team_player_list,
#                                                                                              opponent_player_list,
#                                                                                              ref_date=ref_date)
#             else:
#
#                 feature_vector = feature_extractor.get_second_innings_feature_vector(predicted_first_innings_run, team, opponent,\
#                                                                                     location, team_player_list,\
#                                                                                     opponent_player_list, ref_date=ref_date)
#
#
#
#             second_innings_feature_vector_list.append(feature_vector)
#             if second_innings_team['second_innings'].values[0]==second_innings_team['winner'].values[0]:
#                 result_list.append(1)
#             else:
#                 result_list.append(0)
#
#         except Exception as ex:
#             print(match_id,': 2nd innings Exception for match between  ',team,' and ',opponent,' on ',ref_date,' at ',location)
#             print(ex)
#
#     second_innings_feature_matrix = np.stack(second_innings_feature_vector_list)
#     actual_results = np.stack(result_list)
#
#     predicted_results = predictor_second_innings.predict(second_innings_feature_matrix)
#
#     accuracy = accuracy_score(actual_results, predicted_results)
#
#     print("=====First Innings====== ")
#     print('mape :', mape_val)
#     print('mae :', mae)
#     print('mse :', mse)
#     print('total data ', len(actual_runs))
#     print("=====Combined Innings====== ")
#     print('accuracy ',accuracy)
#     print('data size ', len(result_list))
#
#     if environment != "production":
#         outil.create_model_meta_info_entry("combined_validation_fie_"+
#                                            str(first_innings_emb)+"_sie_"+
#                                            str(second_innings_emb)+"_fim_"+
#                                            first_emb_model,
#                                            0,accuracy,info=from_date+"_to_"+to_date+
#                                            ": first innings embedding - "+str(first_innings_emb)+
#                                            ": second innings embedding - " + str(second_innings_emb))
#
#     # print("predictions ",predicted_results)
#     # print("actuals ", actual_results)
#     return mape_val,mae,mse,accuracy,len(result_list)
#
#
# @click.group()
# def evaluate():
#     pass
#
# @evaluate.command()
# @click.option('--from_date', help='start date in YYYY-mm-dd',required=True)
# @click.option('--to_date', help='end date in YYYY-mm-dd',required=True)
# @click.option('--env', help='end date in YYYY-mm-dd',default='production')
# @click.option('--use_emb', help='set False to use base model',default=True,type=bool)
# @click.option('--model', help='use team for team with batsman/advesarial for player wise',default='team')
# def first(from_date,to_date,env,use_emb,model):
#     evaluate_first_innings(from_date, to_date, environment=env,use_emb=use_emb,model=model)
#
# @evaluate.command()
# @click.option('--from_date', help='start date in YYYY-mm-dd',required=True)
# @click.option('--to_date', help='end date in YYYY-mm-dd',required=True)
# @click.option('--env', help='end date in YYYY-mm-dd',default='production')
# @click.option('--use_emb', help='set False to use base model',default=True,type=bool)
# def second(from_date,to_date,env,use_emb):
#     evaluate_second_innings(from_date, to_date, environment=env,use_emb=use_emb)
#
# @evaluate.command()
# @click.option('--from_date', help='start date in YYYY-mm-dd',required=True)
# @click.option('--to_date', help='end date in YYYY-mm-dd',required=True)
# @click.option('--env', help='end date in YYYY-mm-dd',default='production')
# @click.option('--first_innings_emb', help='set False to use base model',default=True,type=bool)
# @click.option('--second_innings_emb', help='set False to use base model',default=True,type=bool)
# @click.option('--first_emb_model', help='use team for team with batsman/advesarial for player wise',default='team')
# @click.option('--second_emb_model', help='use team for team with batsman/advesarial for player wise',default='team')
# def combined(from_date,to_date,env,first_innings_emb,second_innings_emb,first_emb_model,second_emb_model):
#     evaluate_combined_innings(from_date, to_date, environment=env,
#                               first_innings_emb = first_innings_emb,second_innings_emb=second_innings_emb,
#                               first_emb_model = first_emb_model, second_emb_model = second_emb_model)
#
#
# if __name__=='__main__':
#     evaluate()


