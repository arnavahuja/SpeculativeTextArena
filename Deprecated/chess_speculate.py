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
STANDARD_GAME_PROMPT = "You are a competitive game player. Make sure you read the game instructions carefully, and always follow the required format. Reason extensively and deeply, and then return your move. Important: always return a valid move and a valid move only, even if you are uncertain."
GUESS_PROMPT = "Reason very very succintly about the next move, return only the move, in the format [UCI_MOVE], e.g. [e2e4] (exact syntax!), and make sure the move is in the list of valid moves. Even if you are uncertain, still return a valid move. Important: reason very very quickly and always return a valid move and a valid move only, even if you are uncertain."

class ChessRun:
    def __init__(self, agent0_name="OpenRouter", agent1_name="OpenAI", model_name="gemini"):
        openai.api_key = Constants.openai_api_key
        self.openai_client = OpenAI(api_key=Constants.openai_api_key)
        #self.gemini_client = genai.Client(api_key=Constants.gemini_api_key)
        self.openrouter_client = OpenAI(base_url="https://openrouter.ai/api/v1",api_key=Constants.openrouter_api_key)
        self.agent0_name = agent0_name
        self.agent1_name = agent1_name
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
        reasoning_effort="low",
        n=1
        )
        content = response.choices[0].message.content
        print("OPENAI RESPONSE Raw:", content.strip())
        return content.strip() if content is not None else ""

    def gemini_llm(self, prompt, stop=None):
        config = types.GenerateContentConfig(stop_sequences=stop)
        response = self.gemini_client.models.generate_content(model=self.model_name, contents=prompt, config=config)
        return str(response.text)

    def openrouter_llm(self, prompt):
        response = self.openrouter_client.chat.completions.create(model=self.model_name, messages=[{"role": "system", "content": STANDARD_GAME_PROMPT}, {"role": "user", "content": prompt}], reasoning_effort="low")
        print("OPENROUTER RESPONSE Raw:", response.choices[0].message.content.strip())
        return response.choices[0].message.content.strip()
    
    def call_llm(self, prompt, retries=3):
        valid_moves = ", ".join([f'[{move.uci()}]' for move in self.env.state.game_state["board"].legal_moves])
        # convert valid_moves to a list
        valid_moves = valid_moves.split(", ")
        if self.model_name.startswith("gemini"):
            for i in range(retries):
                action = self.gemini_llm(prompt)
                if action:
                    action = self.clean_actions(action)
                    if action in valid_moves:
                        return action
                    else:
                        prompt = prompt + "\nAttempt " + str(i+1) + " failed because the action is not in the list of valid moves. Remember the valid moves are: " + valid_moves
                print("VALID MOVES:", valid_moves)
                self.log("GUESS ACTION RETRY", f"Attempt {i+1} failed")
                #time.sleep(1)
            return None
        elif self.model_name.startswith("gpt") or self.model_name.startswith("o"):
            for i in range(retries):
                action = self.openai_llm(prompt)
                if action:
                    action = self.clean_actions(action)
                    if action in valid_moves:
                        print("OPENAI RESPONSE action after cleaning:", action)
                        return action
                print("VALID MOVES:", valid_moves)
                self.log("GUESS ACTION RETRY", f"Attempt {i+1} failed")
                #time.sleep(1)
            return None
        elif "/" in self.model_name:
            for i in range(retries):
                action = self.openrouter_llm(prompt)
                if action:
                    action = self.clean_actions(action)
                    if action in valid_moves:
                        return action
                print("VALID MOVES:", valid_moves)
                self.log("GUESS ACTION RETRY", f"Attempt {i+1} failed")
                #time.sleep(1)
            return None
        else:
            raise ValueError("Model name not recognized")
    
    def guess_action(self, observation, retries=3):
        llm_prompt = observation + GUESS_PROMPT
        output = self.call_llm(llm_prompt, retries)
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
            action = self.clean_actions(action)
            if action is not None:
                if action in valid_moves:
                    return action
                else:
                    print("Attempt", i+1, "failed, action", action, "is not in the list of valid moves")
            self.log("RETRY", f"Attempt {i+1} failed")
            role = "White" if player_id == 0 else "Black"
            observation = observation + f"\nAttempt {i+1} failed, please remember that you are acting as {role} in the chess game, and you need to make a valid move from the last valid moves list. Return the move in the format [UCI_MOVE], for example [e2e4]."
            i += 1
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
        start = time.perf_counter()

        # get current prediction
        current_prediction = self.guess_action(observation, player_id)

        #Format the observation message with the correct format
        if current_prediction is None:
            end = time.perf_counter()
            time_taken = end - start
            return [None, None, time_taken, time_taken]
        end_prediction = time.perf_counter()
        time_taken_prediction = end_prediction - start

        new_observation_message = f"\n[GAME] Player {player_id} made the following move: {current_prediction}"

        # Push the move to the board
        move_uci = current_prediction.lower().replace("[", "").replace("]", "")
        current_pred_move = chess.Move.from_uci(move_uci)    
        self.env.state.game_state["board"].push(current_pred_move)

        # Add the current board to the observation message, with the correct board format
        new_observation_message = new_observation_message + "\n[GAME] Current board: \n" + Utils.board_with_coords(self.env.state.game_state["board"]) 

        # Add the valid moves to the observation message
        valid_moves = ', '.join([f'[{move.uci()}]' for move in self.env.state.game_state["board"].legal_moves])
        new_observation_message = new_observation_message + "\nValid moves:" + valid_moves + "\n"

        # Add the original observation message to the new observation
        new_observation = observation + new_observation_message

        # Hack: In original observation, the first line is [GAME] You are playing White/Black in a game of Chess. So we need to flip the role of the player
        new_observation = new_observation.replace("You are playing White","TEMP_WHITE").replace("You are playing Black", "You are playing White").replace("TEMP_WHITE", "You are playing Black")

        # Get the speculation with the new observation
        current_speculation = self.agent_call_with_retry(agent, new_observation, player_id)
        current_speculation = self.clean_actions(current_speculation)

        end = time.perf_counter()
        time_taken = end - start

        # Pop the move from the board -- restore the board to the original state
        self.env.state.game_state["board"].pop()
        self.log("SIMULATION SPECULATION OUTPUT:", current_speculation)

        return [current_prediction, current_speculation, time_taken_prediction, time_taken]

    def chessthink(self, agents, stop_after=None):
        # Initialize the environment
        self.env.reset(num_players=Constants.num_chess_players)

        steps_info = {}
        initial = True
        done = False
        current_agent = agents[0]
        other_agent = agents[1]
        pid, current_observation = self.env.get_observation()
        step_count = 0

        time_checker_regular = 0
        time_checker_speculate = 0
        temp_time_holder = 0

        while not done:
            if not initial:
                prev_pred = current_pred
                prev_spec = current_spec
                prev_move = current_move
            pid, current_observation = self.env.get_observation()

            if not initial and (prev_pred == prev_move):
                current_move = prev_spec
                current_pred = None
                time_checker_regular += temp_time_holder
            else:
                with ThreadPoolExecutor() as executor:
                    future1 = executor.submit(self.current_agent_task, current_agent, current_observation, pid)
                    future2 = executor.submit(self.other_agent_task, other_agent, current_observation, pid)  
                    current_move, time_taken1 = future1.result()
                    current_pred, current_spec, time_taken_prediction, time_taken2 = future2.result()
                    
                    if initial:
                        initial = False
                
                if initial:
                    initial = False

                # Add in the time taken this round
                time_checker_regular += time_taken1
                if current_move == current_pred:
                    time_checker_speculate += max(time_taken1, time_taken2)
                else:
                    time_checker_speculate += time_taken1
                temp_time_holder = time_taken2 - time_taken_prediction
            
            steps_info[step_count] = {"player_id": pid, "current_observation": current_observation, "current_move": current_move, "current_pred": current_pred, "current_spec": current_spec, "time_taken_current_agent": time_taken1, "time_taken_other_agent": time_taken2, "time_taken_prediction": time_taken_prediction}
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

        return steps_info, rewards, game_info, time_checker_regular, time_checker_speculate

    def run(self, stop_after=20):
        self.current_run_id = self.get_runid()
        current_dir_path = join(self.base_traj_path, self.current_run_id)
        rewards = []
        agents = {
            0: ta.agents.OpenRouterAgent(model_name=Constants.openrouter_model_name, system_prompt=STANDARD_GAME_PROMPT, verbose = False) if self.agent0_name == "OpenRouter" else ta.agents.OpenAIAgent(model_name=Constants.openai_model_name, system_prompt=STANDARD_GAME_PROMPT, verbose = False),
            1: ta.agents.OpenAIAgent(model_name=Constants.openai_model_name, system_prompt=STANDARD_GAME_PROMPT, verbose = False) if self.agent1_name == "OpenAI" else ta.agents.OpenRouterAgent(model_name=Constants.openrouter_model_name, system_prompt=STANDARD_GAME_PROMPT, verbose = False),
        }
        self.log("info", f"Starting run {self.current_run_id} with guess model: {self.model_name}")
        
        try:
            steps_info, rewards, game_info, time_checker_regular, time_checker_speculate = self.chessthink(agents, stop_after=stop_after)
        except Exception as e:
            self.log("ERROR",e)
            return
        self.log("INFO", f"Run completed for {self.current_run_id}")
        Utils.save_json(steps_info, join(current_dir_path, "stepsinfo.json"))
        Utils.save_json(rewards, join(current_dir_path, "rewards.json"))
        Utils.save_json(game_info, join(current_dir_path, "game_info.json"))
        Utils.save_json(time_checker_regular, join(current_dir_path, "time_checker_regular.json"))
        Utils.save_json(time_checker_speculate, join(current_dir_path, "time_checker_speculate.json"))

if __name__=="__main__":
    chess_runner = ChessRun(agent0_name = "OpenRouter", agent1_name = "OpenAI", model_name=Constants.openai_guess_model_name)
    begin_time = time.time()
    chess_runner.run(stop_after=40)
    end_time = time.time()
    print(f"Time taken: {end_time - begin_time} seconds")