python $NUPLAN_DEVKIT_ROOT/nuplan-devkit/nuplan/planning/script/run_simulation.py \
+simulation=closed_loop_nonreactive_agents \
planner=pdm_hybrid_planner \
scenario_filter=val14_split \
scenario_builder=nuplan \
hydra.searchpath="[pkg://nuplan_garage.planning.script.config.common, pkg://nuplan_garage.planning.script.config.simulation, pkg://nuplan.planning.script.config.common, pkg://nuplan.planning.script.experiments]"