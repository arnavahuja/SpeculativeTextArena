import os
import openai
from google import genai
from google.genai import types
from google.genai.errors import ClientError, ServerError
import requests
import json
import sys
import random
import time
import constants as Constants
from utils import Utils
import logging
import re
from metrics import Metrics
from os.path import join
import textarena as ta 
import constants as Constants
import uuid
from utils import Utils
from metrics import Metrics
from concurrent.futures import ThreadPoolExecutor

class ChessRun:
    def __init__(self, model_name="gemini"):
        openai.api_key = Constants.openai_api_key
        self.gemini_client = genai.Client(api_key=Constants.gemini_api_key)
        self.simulation_observations_dict = {}
        self.current_run_id = None
        self.model_name = model_name
        self.base_traj_path = "./chesstrajs_" + self.model_name 
        self.log("info", "Initializing env", save_log=False)
        self.env = self.get_env()
        self.log("info", "Env initialized", save_log=False)        

    def get_env(self):
        env = ta.make(env_id="Chess-v0")
        # wrap it for additional visualizations
        env = ta.wrappers.SimpleRenderWrapper(env=env)       
        return env
    
    def log(self, level, *args, save_log=True):
        text = level.upper() + " "
        for arg in args:
            text += str(arg)
            text += " "
        text = text.strip()
        text += "\n"
        print(text)

        if save_log:
            log_path = join(self.base_traj_path, str(self.current_run_id), "log.txt")
            Utils.append_file(text, log_path)


    def openai_llm(self, prompt, stop=["\n"]):
        response = openai.Completion.create(
        model=self.model_name,
        prompt=prompt,
        temperature=0,
        max_tokens=100,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=stop
        )
        return response["choices"][0]["text"]

    def gemini_llm(self, prompt, stop=["\n"]):
        config = types.GenerateContentConfig(stop_sequences=stop)
        response = self.gemini_client.models.generate_content(model=self.model_name, contents=prompt, config=config)
        return str(response.text)
    
    def call_llm(self, prompt, stop=["\n"]):
        if self.model_name.startswith("gemini"):
            return self.gemini_llm(prompt, stop)
        elif self.model_name.startswith("gpt"):
            return self.openai_llm(prompt, stop)
        else:
            raise ValueError("Model name not recognized")

    def step(self, env, action, simulate=False):
        if simulate:
            obs, r, done, info = env.step(action, step_type="simulate")
            return env.sim_obs, r, done, info
        attempts = 0
        while attempts < 10:
            try:
                return env.step(action)
            except requests.exceptions.Timeout:
                attempts += 1

    def separate_thought_and_action(self, thought_action):
        action_condition = f"Action: " in thought_action
        thought_condition = f"Thought: " in thought_action
        thought, action = None, None
        if thought_condition and action_condition:
            thought, action = thought_action.strip().split(f"Action: ")
            thought = thought.strip().split(f"Thought: ")[1]
        elif thought_condition:
            thought = thought_action.strip().split(f"Thought: ")[1]
            action = None
        elif action_condition:
            action = thought_action.strip().split(f"Action: ")[1]
            thought = "I think the opponent will do the action " + action
        else:
            thought = thought_action.strip()
            action = None
        
        if not action:
            action = "[x]"
        return thought, action
    
    def guess_action(self, action, observation):
        llm_prompt = Constants.react_instruction + observation + Constants.chess_guess_prompt
        output = self.call_llm(llm_prompt)
        self.log("SIMULATION GUESS OUTPUT:", output)
        thought, action = self.separate_thought_and_action(output)
        return thought, action

    def get_runid(self):
        return str(uuid.uuid4())    

    def clean_actions(self, action):
        # UCI moves are typically 4 or 5 characters (e.g., a2a4, e7e8q)
        # This regex matches [a-h][1-8][a-h][1-8] with optional promotion [a-h][1-8][a-h][1-8][qrbn]
        if action is None:
            return None
        pattern = r'\[([a-h][1-8][a-h][1-8][qrbn]?)\]'
        ans = None
        if re.findall(pattern, action):
            ans = '['+re.findall(pattern, action)[0]+']'
        else:
            self.log("ERROR", action, "did not match expected format.")
        return ans

    def time_task(self, name, func, *args, **kwargs):
        """Wrapper to time an individual task."""
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()

        time_taken = end - start
        return [result, time_taken]

    def chessthink(self, agents, to_print=True, simulate=False):

        def simulation_task(action, observation):
            new_observation = observation + "\n" + f"The next action performed is {action}\n"
            # Get b'
            guess_opponent_action_thought, guessed_opponent_action = self.guess_action(action, new_observation)

            simulated_observation = new_observation + "\n" + f"After this the next action performed is {guessed_opponent_action}\n. Thus the board state is updated accordingly.\n"
            # Get c
            sim_thought, sim_action = self.guess_action(guessed_opponent_action, simulated_observation)

            return {"simulated_observation": simulated_observation, "sim_action": sim_action, "guessed_opponent_action": guessed_opponent_action}

        def normal_task(observation, player_id):

            player_id, observation = self.env.get_observation()
            # Get b
            action = agents[player_id](observation)

            done, _ = self.env.step(action=action)

            if done: 
                return "DONE"

            return {"player_id": player_id, "observation": observation, "action": action}

        # Initialize the environment
        self.env.reset(num_players=Constants.num_chess_players)

        sim_action = None
        step_num = 0
        steps_info = {}
        tempi = 0

        # Get initial player and game state
        player_id, observation = self.env.get_observation()
        # Get a
        action = agents[player_id](observation) 
        # log initial state
        steps_info[step_num] = {"player_id": player_id, "observation": observation, "action": action, "simulated_action": None}
        self.log("INFO", f"STEP {step_num}:", Utils.dict_to_str(steps_info[step_num]))
        self.log('-'*100)
        
        # perform a
        done, _ = self.env.step(action=action)
        step_num += 1

        # Run the sim while not done
        while not done:
            #perform action 
            if simulate:
                with ThreadPoolExecutor() as executor:
                    future1 = executor.submit(self.time_task, "Simulation Task", simulation_task, action, observation)
                    future2 = executor.submit(self.time_task, "Normal Task", normal_task, observation, player_id)
                    step_num += 1

                    sim_result, sim_time = future1.result()
                    normal_result, normal_time = future2.result()

                    if normal_result == "DONE":
                        done = True
                        continue

                    guessed_opponent_action = sim_result["guessed_opponent_action"]
                    sim_action = sim_result["sim_action"]
                    simulated_observation = sim_result["simulated_observation"]


                    action = normal_result["action"]
                    observation = normal_result["observation"]
                    player_id = normal_result["player_id"]

                    steps_info[step_num] = {"player_id": player_id, "observation": observation, "action": action, "simulated_action": guessed_opponent_action}
                    self.log("INFO", f"STEP {step_num}:", Utils.dict_to_str(steps_info[step_num]))
                    self.log('-'*100)

                if guessed_opponent_action == action:
                    # if the guess was correct, c is already found
                    action = sim_action
                else:
                    # if the guess was wrong, need to get a new c
                    action = agents[player_id](observation) # otherwise, get a new c


                player_id, observation = self.env.get_observation()
                done, _ = self.env.step(action=action) # execute c
                step_num += 1                    
                steps_info[step_num] = {"player_id": player_id, "observation": observation, "action": action, "simulated_action": None}
                self.log("INFO", f"STEP {step_num}:", Utils.dict_to_str(steps_info[step_num]))
                self.log('-'*100)

            else:
                raise NotImplementedError("Non-simulation mode not implemented yet")
                # action = normal_task(action, observation)
                

            done, _ = self.env.step(action=action)

            if tempi > 3:
                break
            tempi += 1

        rewards, game_info = self.env.close()
        return rewards, game_info, steps_info

    def run(self, webthink_simulate=False, skip_done=False):
        self.current_run_id = self.get_runid()
        current_dir_path = join(self.base_traj_path, self.current_run_id)
        rewards = []
        infos = []
        agents = {
            0: ta.agents.GeminiAgent(model_name=Constants.gemini_model_name),
            1: ta.agents.OpenAIAgent(model_name=Constants.openai_model_name),
        }
        self.log("info", f"Starting run {self.current_run_id} with guess model: {self.model_name}")
        
        try:
            rewards, game_info, steps_info = self.chessthink(agents, to_print=True, simulate=webthink_simulate)
        except ClientError as e:
            self.log("info", f"Client Error!! Sleeping for {Constants.client_error_sleep_time} seconds...", save_log=False)
            self.log("error",e,save_log=False)
            # Utils.delete_dir(current_dir_path, nested=True)
            return
        
        except ServerError as e:
            self.log("info", f"Server Error!! Sleeping for {Constants.server_error_sleep_time} seconds...", save_log=False)
            # Utils.delete_dir(current_dir_path, nested=True)
            return

        Utils.save_json(steps_info, join(current_dir_path, "stepsinfo.json"))
        

if __name__=="__main__":
    chess_runner = ChessRun(model_name=Constants.gemini_guess_model_name)
    chess_runner.run(webthink_simulate=True, skip_done=True)

    # avg_metrics_dict, n_samples = Metrics.get_action_specific_avg_metric(chess_runner.base_traj_path)
    # print("AVERAGE METRIC:\n", json.dumps(avg_metrics_dict,indent=4), f"\nfor {n_samples} observations")