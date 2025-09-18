#!/usr/bin/env bash

# Route_id: simulation/leaderboard/data/evaluation_routes/town05_short_r${route_id}.xml
# CP_model：v2xvit, late_fusion, ...
# Agent_config: simulation/leaderboard/team_code/agent_config/pnp_config_${Agent_config}.yaml
# Scenario_config: simulation/leaderboard/leaderboard/scenarios/scenario_parameter_${Scenario_config}.yaml

# Evaluation on one route
# bash RQ3/rq3.sh ${Route_id} ${Carla_port} ${CP_model} ${Agent_config} ${Scenario_config}

###start
bash RQ3/rq3.sh 0 40000 early_fusion early_5_10 1
bash RQ3/rq3.sh 0 40000 late_fusion late_5_10 1
bash RQ3/rq3.sh 0 40000 fcooper fcooper_5_10 1
bash RQ3/rq3.sh 0 40000 v2xvit v2xvit_5_10 1
bash RQ3/rq3.sh 0 40000 single single_5_10 1