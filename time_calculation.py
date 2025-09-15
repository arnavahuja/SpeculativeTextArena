import json


step_info_log = "./chesstrajs_gpt-5-2025-08-07/ddaa9a24-0c8d-40f1-9c51-bebf7921c554/stepsinfo.json"

with open(step_info_log, "r") as f:
    step_info = json.load(f)

time_checker_regular = 0
time_checker_speculate = 0
temp_time_holder = 0
match = False
count = 0

for step in step_info:
    if step_info[step]["current_move"] == step_info[step]["current_pred"]:
        match = True
        time_checker_speculate += max(step_info[step]["time_taken_other_agent"], step_info[step]["time_taken_current_agent"])
        time_checker_regular += step_info[step]["time_taken_current_agent"]
        temp_time_holder = step_info[step]["time_taken_other_agent"] - step_info[step]["time_taken_prediction"]
    elif match:
        time_checker_regular += temp_time_holder
        match = False
    else:
        time_checker_speculate += step_info[step]["time_taken_current_agent"]
        time_checker_regular += step_info[step]["time_taken_current_agent"]
        match = False

print(f"Time taken to regular: {time_checker_regular}")
print(f"Time taken to speculate: {time_checker_speculate}")