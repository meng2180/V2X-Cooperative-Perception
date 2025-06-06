#!/usr/bin/env bash

# Route_id: simulation/leaderboard/data/evaluation_routes/town05_short_r${route_id}.xml
# CP_model：v2xvit, late_fusion, ...
# Agent_config: simulation/leaderboard/team_code/agent_config/pnp_config_${Agent_config}.yaml
# Scenario_config: simulation/leaderboard/leaderboard/scenarios/scenario_parameter_${Scenario_config}.yaml

# Evaluation on one route
# bash RQ3/rq3.sh ${Route_id} ${Carla_port} ${CP_model} ${Agent_config} ${Scenario_config}

###start
# bash RQ3/rq3.sh 0 40000 v2xvit 1 v2xvit_5_10 1
# bash RQ3/rq3.sh 1 40000 v2xvit 1 v2xvit_5_10 1
# bash RQ3/rq3.sh 2 40000 v2xvit 1 v2xvit_5_10 1
# bash RQ3/rq3.sh 3 40000 v2xvit 1 v2xvit_5_10 1
# bash RQ3/rq3.sh 4 40000 v2xvit 1 v2xvit_5_10 1
# bash RQ3/rq3.sh 5 40000 v2xvit 1 v2xvit_5_10 1
# bash RQ3/rq3.sh 6 40000 v2xvit 1 v2xvit_5_10 1
# bash RQ3/rq3.sh 8 40000 v2xvit 1 v2xvit_5_10 1
# bash RQ3/rq3.sh 15 40000 v2xvit 1 v2xvit_5_10 1
# bash RQ3/rq3.sh 16 40000 v2xvit 1 v2xvit_5_10 1
# bash RQ3/rq3.sh 17 40000 v2xvit 1 v2xvit_5_10 1
# bash RQ3/rq3.sh 20 40000 v2xvit 1 v2xvit_5_10 1
# bash RQ3/rq3.sh 23 40000 v2xvit 1 v2xvit_5_10 1
# bash RQ3/rq3.sh 24 40000 v2xvit 1 v2xvit_5_10 1
# bash RQ3/rq3.sh 26 40000 v2xvit 1 v2xvit_5_10 1
# bash RQ3/rq3.sh 29 40000 v2xvit 1 v2xvit_5_10 1
# bash RQ3/rq3.sh 31 40000 v2xvit 1 v2xvit_5_10 1

# bash RQ3/rq3.sh 0 40000 fcooper 1 fcooper_5_10 1
# bash RQ3/rq3.sh 1 40000 fcooper 1 fcooper_5_10 1 
# bash RQ3/rq3.sh 2 40000 fcooper 1 fcooper_5_10 1
# bash RQ3/rq3.sh 3 40000 fcooper 1 fcooper_5_10 1
# bash RQ3/rq3.sh 4 40000 fcooper 1 fcooper_5_10 1
# bash RQ3/rq3.sh 5 40000 fcooper 1 fcooper_5_10 1
# bash RQ3/rq3.sh 6 40000 fcooper 1 fcooper_5_10 1
# bash RQ3/rq3.sh 8 40000 fcooper 1 fcooper_5_10 1
# bash RQ3/rq3.sh 9 40000 fcooper 1 fcooper_5_10 1
# bash RQ3/rq3.sh 11 40000 fcooper 1 fcooper_5_10 1
# bash RQ3/rq3.sh 15 40000 fcooper 1 fcooper_5_10 1
# bash RQ3/rq3.sh 20 40000 fcooper 1 fcooper_5_10 1
# bash RQ3/rq3.sh 23 40000 fcooper 1 fcooper_5_10 1
# bash RQ3/rq3.sh 24 40000 fcooper 1 fcooper_5_10 1
# bash RQ3/rq3.sh 26 40000 fcooper 1 fcooper_5_10 1
# bash RQ3/rq3.sh 28 40000 fcooper 1 fcooper_5_10 1
# bash RQ3/rq3.sh 29 40000 fcooper 1 fcooper_5_10 1
# bash RQ3/rq3.sh 30 40000 fcooper 1 fcooper_5_10 1
# bash RQ3/rq3.sh 31 40000 fcooper 1 fcooper_5_10 1


# bash RQ3/rq3.sh 0 40000 early_fusion 1 early_5_10 1
# bash RQ3/rq3.sh 1 40000 early_fusion 1 early_5_10 1
# bash RQ3/rq3.sh 2 40000 early_fusion 1 early_5_10 1
# bash RQ3/rq3.sh 3 40000 early_fusion 1 early_5_10 1
# bash RQ3/rq3.sh 4 40000 early_fusion 1 early_5_10 1
# bash RQ3/rq3.sh 5 40000 early_fusion 1 early_5_10 1
# bash RQ3/rq3.sh 15 40000 early_fusion 1 early_5_10 1
# bash RQ3/rq3.sh 16 40000 early_fusion 1 early_5_10 1
# bash RQ3/rq3.sh 17 40000 early_fusion 1 early_5_10 1
# bash RQ3/rq3.sh 20 40000 early_fusion 1 early_5_10 1
# bash RQ3/rq3.sh 21 40000 early_fusion 1 early_5_10 1
# bash RQ3/rq3.sh 22 40000 early_fusion 1 early_5_10 1
# bash RQ3/rq3.sh 23 40000 early_fusion 1 early_5_10 1
# bash RQ3/rq3.sh 24 40000 early_fusion 1 early_5_10 1
# bash RQ3/rq3.sh 26 40000 early_fusion 1 early_5_10 1
# bash RQ3/rq3.sh 28 40000 early_fusion 1 early_5_10 1
# bash RQ3/rq3.sh 29 40000 early_fusion 1 early_5_10 1
# bash RQ3/rq3.sh 30 40000 early_fusion 1 early_5_10 1
# bash RQ3/rq3.sh 31 40000 early_fusion 1 early_5_10 1


# bash RQ3/rq3.sh 0 40000 late_fusion 1 late_5_10 1
# bash RQ3/rq3.sh 1 40000 late_fusion 1 late_5_10 1
# bash RQ3/rq3.sh 2 40000 late_fusion 1 late_5_10 1
# bash RQ3/rq3.sh 3 40000 late_fusion 1 late_5_10 1
# bash RQ3/rq3.sh 4 40000 late_fusion 1 late_5_10 1
# bash RQ3/rq3.sh 5 40000 late_fusion 1 late_5_10 1
# bash RQ3/rq3.sh 8 40000 late_fusion 1 late_5_10 1
# bash RQ3/rq3.sh 14 40000 late_fusion 1 late_5_10 1
# bash RQ3/rq3.sh 15 40000 late_fusion 1 late_5_10 1
# bash RQ3/rq3.sh 16 40000 late_fusion 1 late_5_10 1
# bash RQ3/rq3.sh 20 40000 late_fusion 1 late_5_10 1
# bash RQ3/rq3.sh 21 40000 late_fusion 1 late_5_10 1
# bash RQ3/rq3.sh 23 40000 late_fusion 1 late_5_10 1
# bash RQ3/rq3.sh 24 40000 late_fusion 1 late_5_10 1
# bash RQ3/rq3.sh 26 40000 late_fusion 1 late_5_10 1
# bash RQ3/rq3.sh 28 40000 late_fusion 1 late_5_10 1
# bash RQ3/rq3.sh 29 40000 late_fusion 1 late_5_10 1
# bash RQ3/rq3.sh 30 40000 late_fusion 1 late_5_10 1
# bash RQ3/rq3.sh 31 40000 late_fusion 1 late_5_10 1