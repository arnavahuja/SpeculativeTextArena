import os
import openai
from openai import OpenAI
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
STANDARD_GAME_PROMPT = "You are a competitive game player. Make sure you read the game instructions carefully, and always follow the required format."

class ChessRun:
    def __init__(self, model_name="gemini"):
        openai.api_key = Constants.openai_api_key
        self.openai_client = OpenAI(api_key=Constants.openai_api_key)
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

        messages = [{"role": "system", "content": STANDARD_GAME_PROMPT }, {"role": "user", "content": prompt}]
        response = self.openai_client.chat.completions.create(
        model=self.model_name,
        messages=messages,
        temperature=0,
        n=1,
        stop=stop
        )
        return response.choices[0].message.content.strip()

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
    
    def guess_action(self, observation):
        llm_prompt = observation + Constants.chess_guess_prompt
        output = self.call_llm(llm_prompt)
        self.log("SIMULATION GUESS OUTPUT:", output)
        action = self.clean_actions(output)
        return action

    def get_runid(self):
        return str(uuid.uuid4())    

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

    def current_agent_task(self, agent, observation, is_correct_pred):
        start = time.perf_counter()
        if is_correct_pred:
            current_move = None
        else:
            current_move = agent(observation)
        end = time.perf_counter()

        time_taken = end - start
        return [current_move, time_taken]

    def other_agent_task(self, agent, observation):
        start = time.perf_counter()
        current_prediction = self.guess_action(observation)
        # TODO: Use correct format for the 
        new_observation = observation + "\n" + f"The next action performed is {current_prediction}\n" + "Thus the board state is updated accordingly. Based on this what is the next action?\n"
        current_speculation = agent(new_observation)
        end = time.perf_counter()

        time_taken = end - start
        return [current_prediction, current_speculation, time_taken]

    def chessthink(self, agents, to_print=True, simulate=False, stop_after=None):
        # Initialize the environment
        self.env.reset(num_players=Constants.num_chess_players)

        steps_info = {}
        initial = True
        done = False
        # initial_player_id, current_observation = self.env.get_observation()
        current_agent = agents[0]
        other_agent = agents[1]
        pid, current_observation = self.env.get_observation()
        is_correct_pred = False
        step_count = 0

        while not done:

            if not initial:
                prev_pred = current_pred
                prev_spec = current_spec
                prev_move = current_move
            pid, current_observation = self.env.get_observation()
            if not initial and (prev_pred == prev_move):
                current_move = prev_spec
                current_pred = None
                is_correct_pred = True
            else:
                is_correct_pred = False   
                with ThreadPoolExecutor() as executor:
                    future1 = executor.submit(self.current_agent_task, current_agent, current_observation, is_correct_pred)
                    future2 = executor.submit(self.other_agent_task, other_agent, current_observation)  
                    current_move, time_taken1 = future1.result()
                    current_pred, current_spec, time_taken2 = future2.result()
                    
                    if initial:
                        initial = False               
            
            steps_info[step_count] = {"player_id": pid, "current_observation": current_observation, "current_move": current_move, "current_pred": current_pred, "current_spec": current_spec, "time_taken_current_agent": time_taken1, "time_taken_other_agent": time_taken2}
            self.log("INFO", f"STEP {step_count}:", Utils.dict_to_str(steps_info[step_count]))
            self.log('-'*100)

            done, info = self.env.step(current_move)
            step_count += 1
            if stop_after and step_count >= stop_after:
                break
            temp = current_agent
            current_agent = other_agent
            other_agent = temp

        return steps_info

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
            steps_info = self.chessthink(agents, to_print=True, simulate=webthink_simulate, stop_after=10)
        except Exception as e:
            self.log("ERROR",e)
            return
        self.log("INFO", f"Run completed for {self.current_run_id}")
        Utils.save_json(steps_info, join(current_dir_path, "stepsinfo.json"))
        

if __name__=="__main__":
    chess_runner = ChessRun(model_name=Constants.openai_guess_model_name)
    chess_runner.run(webthink_simulate=True, skip_done=True)

    # avg_metrics_dict, n_samples = Metrics.get_action_specific_avg_metric(chess_runner.base_traj_path)
    # print("AVERAGE METRIC:\n", json.dumps(avg_metrics_ict,indent=4), f"\nfor {n_samples} observations")