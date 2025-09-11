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
import chess
import uuid
from metrics import Metrics
from concurrent.futures import ThreadPoolExecutor
STANDARD_GAME_PROMPT = "You are a competitive game player. Make sure you read the game instructions carefully, and always follow the required format."

class ChessRun:
    def __init__(self, model_name="gemini"):
        openai.api_key = Constants.openai_api_key
        self.openai_client = OpenAI(api_key=Constants.openai_api_key)
        #self.gemini_client = genai.Client(api_key=Constants.gemini_api_key)
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
        #env = ta.wrappers.SimpleRenderWrapper(env=env)       
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


    def openai_llm(self, prompt):

        messages = [{"role": "system", "content": STANDARD_GAME_PROMPT }, {"role": "user", "content": prompt}]
        response = self.openai_client.chat.completions.create(
        model=self.model_name,
        messages=messages,
        temperature=0,
        n=1
        )
        content = response.choices[0].message.content
        return content.strip() if content is not None else ""

    def gemini_llm(self, prompt, stop=None):
        config = types.GenerateContentConfig(stop_sequences=stop)
        response = self.gemini_client.models.generate_content(model=self.model_name, contents=prompt, config=config)
        return str(response.text)
    
    def call_llm(self, prompt, retries=3):
        valid_moves = ", ".join([f'[{move.uci()}]' for move in self.env.state.game_state["board"].legal_moves])
        # convert valid_moves to a list
        valid_moves = valid_moves.split(", ")
        if self.model_name.startswith("gemini"):
            for i in range(retries):
                action = self.gemini_llm(prompt)
                if action:
                    if action in valid_moves:
                        return action
                self.log("GUESS ACTION RETRY", f"Attempt {i+1} failed")
                #time.sleep(1)
            return None
        elif self.model_name.startswith("gpt"):
            for i in range(retries):
                action = self.openai_llm(prompt)
                if action:
                    if action in valid_moves:
                        return action
                self.log("GUESS ACTION RETRY", f"Attempt {i+1} failed")
                #time.sleep(1)
            return None
        else:
            raise ValueError("Model name not recognized")

    # def step(self, env, action, simulate=False):
    #     if simulate:
    #         obs, r, done, info = env.step(action, step_type="simulate")
    #         return env.sim_obs, r, done, info
    #     attempts = 0
    #     while attempts < 10:
    #         try:
    #             return env.step(action)
    #         except requests.exceptions.Timeout:
    #             attempts += 1

    # def separate_thought_and_action(self, thought_action):
    #     if thought_action is None:
    #         return "No thought provided", "[x]"
            
    #     action_condition = f"Action: " in thought_action
    #     thought_condition = f"Thought: " in thought_action
    #     thought, action = None, None
    #     if thought_condition and action_condition:
    #         thought, action = thought_action.strip().split(f"Action: ")
    #         thought = thought.strip().split(f"Thought: ")[1]
    #     elif thought_condition:
    #         thought = thought_action.strip().split(f"Thought: ")[1]
    #         action = None
    #     elif action_condition:
    #         action = thought_action.strip().split(f"Action: ")[1]
    #         thought = "I think the opponent will do the action " + action
    #     else:
    #         thought = thought_action.strip()
    #         action = None
        
    #     if not action:
    #         action = "[x]"
    #     return thought, action
    
    def guess_action(self, observation, player_id, retries=3):
        role = "White" if player_id == 0 else "Black"
        #llm_prompt = observation + f"You are acting as {role} in the chess game." + Constants.chess_guess_prompt
        # FIXME: trying without the role and guess prompt for now to see if it works
        llm_prompt = observation + "Reason succintly about the next move, return fast and short, in the format [UCI_MOVE], e.g. [e2e4] (exact syntax!), and make sure the move is in the list of valid moves."
        output = self.call_llm(llm_prompt)
        self.log("SIMULATION GUESS OUTPUT:", output)
        action = self.clean_actions(output)
        return action

    def get_runid(self):
        return str(uuid.uuid4())    
    
    def agent_call_with_retry(self, agent, observation, player_id, retries=3):
        valid_moves = ", ".join([f'[{move.uci()}]' for move in self.env.state.game_state["board"].legal_moves])
        # convert valid_moves to a list
        valid_moves = valid_moves.split(", ")
        for i in range(retries):
            action = agent(observation)
            print("="*100)
            print("Raw action:", action)
            print("="*100)
            action = self.clean_actions(action)
            if action is not None:
                if action in valid_moves:
                    return action
            self.log("RETRY", f"Attempt {i+1} failed")
            role = "White" if player_id == 0 else "Black"
            observation = observation + f"\nAttempt {i+1} failed, please remember that you are acting as {role} in the chess game, and you need to make a valid move from the last valid moves list. Return the move in the format [UCI_MOVE], for example [e2e4]."
            i += 1
            #time.sleep(1)
        return None
            

    def clean_actions(self, action):
        # UCI moves are typically 4 or 5 characters (e.g., a2a4, e7e8q)
        # This regex matches [a-h][1-8][a-h][1-8] with optional promotion [a-h][1-8][a-h][1-8][qrbn]
        # Now handles optional spaces inside brackets like [ d8f6 ] or [d8f6]
        if action is None:
            return None
        pattern = r'\[\s*([a-h][1-8][a-h][1-8][qrbn]?)\s*\]'
        ans = None
        if re.findall(pattern, action):
            ans = '['+re.findall(pattern, action)[-1]+']'
        else:
            self.log("ERROR", action, "did not match expected format.")
        return ans

    def current_agent_task(self, agent, observation, player_id):
        start = time.perf_counter()
        current_move = self.agent_call_with_retry(agent, observation, player_id)
        current_move = self.clean_actions(current_move)
        end = time.perf_counter()

        time_taken = end - start
        return [current_move, time_taken]

    def other_agent_task(self, agent, observation, player_id):

        # print("="*100)
        # print("Observation before prediction:", observation)
        # print("="*100)
        start = time.perf_counter()
        current_prediction = self.guess_action(observation, player_id)
        # print("="*100)
        # print("Current prediction:", current_prediction)
        # print("="*100)
        #Format the observation message with the correct format
        new_observation_message = f"\n[GAME] Player {player_id} made the following move: {current_prediction}"
        move_uci = current_prediction.lower().replace("[", "").replace("]", "")
        current_pred_move = chess.Move.from_uci(move_uci) # Attempt to make the move

        # print("="*100)
        # print("Current board before push:\n", Utils.board_with_coords(self.env.state.game_state["board"]))
        # print("\nCurrent valid moves before push:", ", ".join([f'[{move.uci()}]' for move in self.env.state.game_state["board"].legal_moves]))
        # print("="*100)
        if current_pred_move not in self.env.state.game_state["board"].legal_moves:
            # print(f"current pred move is not a legal move: {current_pred_move}")
            end = time.perf_counter()
            time_taken = end - start
            return None, None, time_taken
        # print(f"current pred move is a legal move: {current_pred_move}")
        # Push the move to the board
        self.env.state.game_state["board"].push(current_pred_move)
        # print("="*100)
        # print("Current board after push:\n", Utils.board_with_coords(self.env.state.game_state["board"]))
        # print("\nCurrent valid moves after push:", ", ".join([f'[{move.uci()}]' for move in self.env.state.game_state["board"].legal_moves]))
        # print("="*100)
        # Add the current board to the observation message, with the correct board format
        new_observation_message = new_observation_message + "\n[GAME] Current board: \n" + Utils.board_with_coords(self.env.state.game_state["board"]) 

        # Add the valid moves to the observation message
        valid_moves = ', '.join([f'[{move.uci()}]' for move in self.env.state.game_state["board"].legal_moves])
        new_observation_message = new_observation_message + "\nValid moves:" + valid_moves + "\n"
        # print(f"appending new observation message: {new_observation_message[0:300]}\n...\n")


        # Add the new observation message to the observation
        new_observation = observation + new_observation_message
        # print("="*100)
        # print("New observation:", new_observation)
        # print("="*100)

        # Hack: In original observation, the first line is [GAME] You are playing White/Black in a game of Chess. So we need to flip the role of the player
        new_observation = new_observation.replace("You are playing White","TEMP_WHITE").replace("You are playing Black", "You are playing White").replace("TEMP_WHITE", "You are playing Black")
        # Pop the move from the board -- restore the board to the original state

        # print("="*100)
        # print("New new observation after flip:", new_observation)
        # print("="*100)
        

        current_speculation = self.agent_call_with_retry(agent, new_observation, player_id)
        current_speculation = self.clean_actions(current_speculation)
        # print("="*100)
        # print("Current speculation:", current_speculation)
        # print("="*100)

        end = time.perf_counter()
        self.env.state.game_state["board"].pop()
        self.log("SIMULATION SPECULATION OUTPUT:", current_speculation)

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
        step_count = 0

        while not done:
            #print(f"{"@"*20} step {step_count} {"@"*20}")
            #time.sleep(5)
            if not initial:
                prev_pred = current_pred
                prev_spec = current_spec
                prev_move = current_move
            pid, current_observation = self.env.get_observation()

            if not initial and (prev_pred == prev_move):
                # print("Successfully predicted the opponent's move!")
                current_move = prev_spec
                current_pred = None
            else:
                with ThreadPoolExecutor() as executor:
                    future1 = executor.submit(self.current_agent_task, current_agent, current_observation, pid)
                    future2 = executor.submit(self.other_agent_task, other_agent, current_observation, pid)  
                    current_move, time_taken1 = future1.result()
                    current_pred, current_spec, time_taken2 = future2.result()
                    
                    if initial:
                        initial = False
                
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
        
        rewards, game_info = self.env.close()

        return steps_info, rewards, game_info

    def run(self, webthink_simulate=False, skip_done=False):
        self.current_run_id = self.get_runid()
        current_dir_path = join(self.base_traj_path, self.current_run_id)
        rewards = []
        infos = []
        agents = {
            0: ta.agents.OpenAIAgent(model_name=Constants.openai_model_name),
            1: ta.agents.OpenAIAgent(model_name=Constants.openai_model_name),
        }
        self.log("info", f"Starting run {self.current_run_id} with guess model: {self.model_name}")
        
        try:
            steps_info, rewards, game_info = self.chessthink(agents, to_print=True, simulate=webthink_simulate, stop_after=20)
        except Exception as e:
            self.log("ERROR",e)
            return
        self.log("INFO", f"Run completed for {self.current_run_id}")
        Utils.save_json(steps_info, join(current_dir_path, "stepsinfo.json"))
        Utils.save_json(rewards, join(current_dir_path, "rewards.json"))
        Utils.save_json(game_info, join(current_dir_path, "game_info.json"))

if __name__=="__main__":
    chess_runner = ChessRun(model_name=Constants.openai_guess_model_name)
    chess_runner.run(webthink_simulate=True, skip_done=True)

    # avg_metrics_dict, n_samples = Metrics.get_action_specific_avg_metric(chess_runner.base_traj_path)
    # print("AVERAGE METRIC:\n", json.dumps(avg_metrics_ict,indent=4), f"\nfor {n_samples} observations")