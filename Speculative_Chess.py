"""
Chess Speculative Game Runner

A chess game runner that implements speculative execution
where a guess model predicts opponent moves and prepare responses in parallel.
"""

import re
import time
import uuid
import logging
from concurrent.futures import ThreadPoolExecutor
from os.path import join
from typing import Dict, List, Optional, Tuple, Any

import chess
import openai
from openai import OpenAI

import textarena as ta
from utils import Utils
import yaml


class Config:
    """Configuration management with YAML support"""
    
    def __init__(self, config_path: Optional[str] = "./config.yml"):
        if config_path and config_path.endswith('.yml'):
            self._load_from_yaml(config_path)
    
    def _load_from_yaml(self, config_path: str):
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # API Configuration
            self.openai_api_key = config['api']['openai']['key']
            self.openrouter_api_key = config['api']['openrouter']['key']
            #self.gemini_api_key = config['api']['gemini']['key']
            
            # Model Configuration
            self.openai_model_name = config['models']['openai']['main']
            self.openai_guess_model_name = config['models']['openai']['guess']
            self.openrouter_model_name = config['models']['openrouter']['main']
            self.openrouter_guess_model_name = config['models']['openrouter']['guess']
            #self.gemini_model_name = config['models']['gemini']['main']
            
            # Game Configuration
            self.num_chess_players = config['game']['num_players']
            self.client_error_sleep_time = config['game']['error_sleep_time']
            self.server_error_sleep_time = config['game']['error_sleep_time']
            self.stop_after = config['game']['stop_after']
            self.agent_name0 = config['game']['agent_name0']
            self.agent_name1 = config['game']['agent_name1']
            self.guess_model_name = config['game']['guess_model_name']
            self.num_guesses = config['game']['num_guesses']
            
        except Exception as e:
            print(f"Error loading YAML config: {e}")

class PromptTemplates:

    STANDARD_GAME_PROMPT = (
        "You are a competitive game player. Make sure you read the game instructions "
        "carefully, and always follow the required format. Reason extensively and deeply, "
        "and then return your move. Important: always return a valid move and a valid move "
        "only, even if you are uncertain."
    )
    
    GUESS_PROMPT = (
        "Reason very very succinctly about the next move, return {num_guesses} candidates for the next move. IMPORTANT FORMAT: "
        "Your response must be a comma-separated list of moves in [UCI_MOVE] format, like: [e2e4], [e2e3], [d2d4] "
        "(exact syntax with square brackets and commas between moves!). For example: [e2e4], [d2d4], [g1f3]. "
        "Ensure all moves are from the list of valid moves. Even if uncertain, return exactly {num_guesses} candidates. "
        "Reason very very quickly, no need to truly think about each of the candidates, just return the most likely candidates and return precisely {num_guesses} valid moves - no more, no less."
    )
    
    RETRY_PROMPT = (
        "\nAttempt {attempt} failed, please remember that you are acting as {role} in "
        "the chess game, and you need to make a valid move from the last valid moves list. "
        "Return the move in the format [UCI_MOVE], for example [e2e4]."
    )


class ChessActionCleaner:
    """Utility class for cleaning and validating chess actions"""
    
    UCI_PATTERN = re.compile(r'\[\s*([a-h][1-8][a-h][1-8][qrbn]?)\s*\]')
    
    @classmethod
    def clean_action(cls, action: Optional[str]) -> Optional[str]:
        """
        Clean and validate a chess action string.
        
        Args:
            action: Raw action string from agent
            
        Returns:
            Cleaned UCI move in format [move] or None if invalid
        """
        if action is None:
            return None
            
        match = cls.UCI_PATTERN.search(action)
        if match:
            return f'[{match.group(1)}]'
        
        return None
    
    @classmethod
    def clean_actions(cls, action: Optional[str]) -> List[str]:
        """
        Clean and validate multiple chess action strings.
        
        Args:
            action: Raw action string from agent that may contain multiple moves
            
        Returns:
            List of cleaned UCI moves in format [move]
        """
        if action is None:
            return []
            
        matches = cls.UCI_PATTERN.findall(action)
        return [f'[{move}]' for move in matches]


class GameLogger:
    
    def __init__(self, base_path: str, run_id: str):
        self.base_path = base_path
        self.run_id = run_id
        self.log_path = join(base_path, str(run_id), "log.txt")
    
    def log(self, level: str, *args, save_log: bool = True) -> None:
        """Log message with specified level"""
        message = f"{level.upper()} {' '.join(str(arg) for arg in args)}\n"
        print(message, end='')
        
        if save_log:
            Utils.append_file(message, self.log_path)


class AgentManager:
    
    def __init__(self, config: Config):
        self.config = config
        self.openai_client = OpenAI(api_key=config.openai_api_key)
        self.openrouter_client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=config.openrouter_api_key
        )
    
    def create_agents(self, agent0_name: str, agent1_name: str) -> Dict[int, Any]:

        agents = {}
        
        for i, agent_name in enumerate([agent0_name, agent1_name]):
            if agent_name == "OpenRouter":
                agents[i] = ta.agents.OpenRouterAgent(
                    model_name=self.config.openrouter_model_name,
                    system_prompt=PromptTemplates.STANDARD_GAME_PROMPT,
                    api_key=self.config.openrouter_api_key,
                    base_url="https://openrouter.ai/api/v1",
                    verbose=False
                )
            elif agent_name == "OpenAI":
                agents[i] = ta.agents.OpenAIAgent(
                    model_name=self.config.openai_model_name,
                    system_prompt=PromptTemplates.STANDARD_GAME_PROMPT,
                    api_key=self.config.openai_api_key,
                    base_url="https://api.openai.com/v1",
                    verbose=False
                )
            else:
                raise ValueError(f"Unknown agent type: {agent_name}")
        
        return agents
    
    def call_guess_llm(self, prompt: str, model_name: str, retries: int = 3) -> Tuple[Optional[str], Optional[int], Optional[int], Optional[int]]:
        for attempt in range(retries):
            try:
                if model_name.startswith("gpt") or model_name.startswith("o"):
                    response = self.openai_client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": PromptTemplates.STANDARD_GAME_PROMPT},
                            {"role": "user", "content": prompt}
                        ],
                        reasoning_effort="low",
                        n=1
                    )
                    usage = response.usage
                    if usage:
                        input_tokens = usage.prompt_tokens
                        output_tokens = usage.completion_tokens
                        total_tokens = usage.total_tokens
                    return response.choices[0].message.content.strip(), input_tokens, output_tokens, total_tokens
                
                elif "/" in model_name:  # OpenRouter model
                    response = self.openrouter_client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": PromptTemplates.STANDARD_GAME_PROMPT},
                            {"role": "user", "content": prompt}
                        ],
                        reasoning_effort="low"
                    )
                    usage = response.usage
                    if usage:
                        input_tokens = usage.prompt_tokens
                        output_tokens = usage.completion_tokens
                        total_tokens = usage.total_tokens
                    return response.choices[0].message.content.strip(), input_tokens, output_tokens, total_tokens
                
            except Exception as e:
                print(f"LLM call attempt {attempt + 1} failed: {e}")
                if attempt == retries - 1:
                    return None, None, None, None
        
        return None


class SpeculativeChessRunner:
    """
    Main class for running speculative chess games.
    
    This runner implements parallel speculation where a guess model predicts
    the opponent's move while the actual agent is thinking, allowing for
    faster gameplay through speculative execution.
    """
    
    def __init__(
        self,
        agent0_name: str = "OpenRouter",
        agent1_name: str = "OpenAI",
        guess_model_name: str = "gpt-5-2025-08-07",
        num_guesses: int = 1,
        config_path: Optional[str] = "./config.yml"
    ):
        """Initialize the chess runner with configuration"""
        self.config = Config(config_path)
        self.agent_manager = AgentManager(self.config)
        
        self.agent0_name = agent0_name
        self.agent1_name = agent1_name
        self.guess_model_name = guess_model_name
        self.num_guesses = num_guesses
        
        self.current_run_id: Optional[str] = None
        self.base_traj_path = f"./Spec_Chess_Trajs/{agent0_name}_vs_{agent1_name}_guess{guess_model_name}"
        self.logger: Optional[GameLogger] = None
        
        # Initialize environment
        self.env = self._create_environment()
        print("Chess environment initialized successfully")
    
    def _create_environment(self) -> ta.Env:
        env = ta.make(env_id="Chess-v0")
        return env
    
    def _get_valid_moves(self) -> List[str]:
        """Get list of valid moves in UCI format"""
        return [f'[{move.uci()}]' for move in self.env.state.game_state["board"].legal_moves]
    
    def _guess_action(self, observation: str, retries: int = 3) -> Tuple[Optional[str], float, Optional[int], Optional[int], Optional[int]]:
        start_pred_time = time.perf_counter()
        prompt = observation + PromptTemplates.GUESS_PROMPT.format(num_guesses=1)
        raw_output, input_tokens, output_tokens, total_tokens = self.agent_manager.call_guess_llm(prompt, self.guess_model_name, retries)
        end_pred_time = time.perf_counter()
        prediction_time = end_pred_time - start_pred_time
        
        if self.logger:
            self.logger.log("SIMULATION GUESS OUTPUT", raw_output)
        
        return ChessActionCleaner.clean_action(raw_output), prediction_time, input_tokens, output_tokens, total_tokens

    def _guess_actions(self, observation: str, retries: int = 3) -> Tuple[Optional[List[str]], float, Optional[int], Optional[int], Optional[int]]:
        start_pred_time = time.perf_counter()
        prompt = observation + PromptTemplates.GUESS_PROMPT.format(num_guesses=self.num_guesses)
        raw_output, input_tokens, output_tokens, total_tokens = self.agent_manager.call_guess_llm(prompt, self.guess_model_name, retries)
        end_pred_time = time.perf_counter()
        prediction_time = end_pred_time - start_pred_time
        
        if self.logger:
            self.logger.log("SIMULATION GUESS OUTPUT", raw_output)
        
        return ChessActionCleaner.clean_actions(raw_output), prediction_time, input_tokens, output_tokens, total_tokens
    
    def _agent_call_with_retry(
        self, 
        agent: Any, 
        observation: str, 
        player_id: int, 
        valid_moves: List[str],
        retries: int = 3
    ) -> Tuple[Optional[str], int, int, int]:
        role = "White" if player_id == 0 else "Black"
        
        for attempt in range(retries):
            raw_action, input_tokens, output_tokens, total_tokens = agent(observation)
            cleaned_action = ChessActionCleaner.clean_action(raw_action)
            
            if cleaned_action and cleaned_action in valid_moves:
                return cleaned_action, input_tokens, output_tokens, total_tokens
            
            if self.logger:
                self.logger.log("RETRY", f"Attempt {attempt + 1} failed for {role}")
            
            observation += PromptTemplates.RETRY_PROMPT.format(
                attempt=attempt + 1, 
                role=role
            )
        
        return None, 0, 0, 0
    
    def _current_agent_task(self, agent: Any, observation: str, player_id: int) -> Tuple[Optional[str], float, int, int, int]:
        """Execute the current agent's move selection"""
        start_time = time.perf_counter()
        valid_moves = self._get_valid_moves()
        move, input_tokens, output_tokens, total_tokens = self._agent_call_with_retry(agent, observation, player_id, valid_moves)
        end_time = time.perf_counter()
        
        return move, end_time - start_time, input_tokens, output_tokens, total_tokens
    
    def _speculation_task(self, agent: Any, observation: str, player_id: int) -> Tuple[List[Optional[str]], List[Optional[str]], List[Optional[float]], List[Optional[float]], List[Optional[float]], List[Optional[int]], List[Optional[int]], List[Optional[int]], List[Optional[int]], List[Optional[int]], List[Optional[int]]]:
        """Execute speculation: predict opponent move and prepare response"""
        start_time = time.perf_counter()

        # # Generate multiple predictions in parallel
        # with ThreadPoolExecutor(max_workers=self.num_guesses) as executor:
        #     prediction_futures = [
        #         executor.submit(self._guess_action, observation, retries=3)
        #         for _ in range(self.num_guesses)
        #     ]
        #     prediction_results = [future.result() for future in prediction_futures]

        prediction_results = self._guess_actions(observation, retries=3)
        print(f"Prediction results: {prediction_results}")

        # Unpack predictions and prediction times
        # predictions : List[Optional[str]] = []
        # individual_prediction_times : List[Optional[float]] = []
        # for result in prediction_results:
        #     if result is not None and len(result) == 2:
        #         predictions.append(result[0])
        #         individual_prediction_times.append(result[1])
        #     else:
        #         predictions.append(None)
        #         individual_prediction_times.append(None)

        predictions = prediction_results[0]
        individual_prediction_times = [prediction_results[1]] * len(predictions) if predictions is not None else [None]
        input_prediction_tokens = [prediction_results[2]] * len(predictions) if predictions is not None else [None]
        output_prediction_tokens = [prediction_results[3]] * len(predictions) if predictions is not None else [None]
        total_prediction_tokens = [prediction_results[4]] * len(predictions) if predictions is not None else [None]

        # Remove None predictions and get unique predictions
        valid_predictions = [p for p in predictions if p is not None] if predictions is not None else []
        valid_prediction_times = [individual_prediction_times[i] for i in range(len(individual_prediction_times)) if predictions is not None and predictions[i] is not None]

        if not valid_predictions:
            return [], [], valid_prediction_times, [0] * len(valid_prediction_times), valid_prediction_times, [], [], [], [], [], []

        print(f"Predictions: {predictions}")
        print(f"Individual prediction times: {individual_prediction_times}")
        print(f"Input prediction tokens: {input_prediction_tokens}")
        print(f"Output prediction tokens: {output_prediction_tokens}")
        print(f"Total prediction tokens: {total_prediction_tokens}")

        print(f"Valid predictions: {valid_predictions}")
        print(f"Valid prediction times: {valid_prediction_times}")

        # Simulate the predicted moves in parallel
        with ThreadPoolExecutor(max_workers=len(valid_predictions)) as executor:
            speculation_futures = [
                executor.submit(self._simulate_and_speculate, agent, observation, player_id, prediction)
                for prediction in valid_predictions
            ]
            speculations_results = [future.result() for future in speculation_futures]

        print(f"Speculations: {speculations_results}")
        
        speculations: List[Optional[str]] = []
        individual_speculation_times: List[Optional[float]] = []
        input_speculation_tokens: List[Optional[int]] = []
        output_speculation_tokens: List[Optional[int]] = []
        total_speculation_tokens: List[Optional[int]] = []

        for speculation in speculations_results:
            if speculation is not None and len(speculation) == 5:
                speculations.append(speculation[0])
                individual_speculation_times.append(speculation[1])
                input_speculation_tokens.append(speculation[2])
                output_speculation_tokens.append(speculation[3])
                total_speculation_tokens.append(speculation[4])
            else:
                speculations.append(None)
                individual_speculation_times.append(None)
                input_speculation_tokens.append(None)
                output_speculation_tokens.append(None)
                total_speculation_tokens.append(None)
        

        total_times: List[Optional[float]] = []
        for i in range(len(valid_prediction_times)):
            pred_time = individual_prediction_times[i]
            spec_time = individual_speculation_times[i]
            if pred_time is None or spec_time is None:
                total_times.append(None)
            else:
                total_times.append(pred_time + spec_time)
        return valid_predictions, speculations, valid_prediction_times, individual_speculation_times, total_times, input_prediction_tokens, output_prediction_tokens, total_prediction_tokens, input_speculation_tokens, output_speculation_tokens, total_speculation_tokens
    
    def _simulate_and_speculate(
        self, 
        agent: Any, 
        observation: str, 
        player_id: int, 
        predicted_move: str
    ) -> Tuple[Optional[str], float, int, int, int]:
        """Simulate predicted move and generate speculative response"""

        start_time_speculate = time.perf_counter()

        # Create game state message
        move_message = f"\n[GAME] Player {player_id} made the following move: {predicted_move}"
        
        # Execute predicted move on board
        move_uci = predicted_move.lower().replace("[", "").replace("]", "")
        predicted_chess_move = chess.Move.from_uci(move_uci)
        # Make a copy of the board and push the move to the copy
        board_copy = self.env.state.game_state["board"].copy()
        board_copy.push(predicted_chess_move)
        
        # Build new observation with board state
        board_str = Utils.board_with_coords(board_copy)     
        valid_moves = [f'[{move.uci()}]' for move in board_copy.legal_moves]
        valid_moves_in_string = ", ".join(valid_moves)
        
        new_observation = (
            observation + move_message + 
            f"\n[GAME] Current board: \n{board_str}" +
            f"\nValid moves: {valid_moves_in_string}\n"
        )
        
        # Flip player roles for speculation
        new_observation = new_observation.replace(
            "You are playing White", "TEMP_WHITE"
        ).replace(
            "You are playing Black", "You are playing White"
        ).replace(
            "TEMP_WHITE", "You are playing Black"
        )
        
        # Get speculative move
        speculation, input_tokens, output_tokens, total_tokens = self._agent_call_with_retry(agent, new_observation, player_id, valid_moves)
        
        if self.logger:
            self.logger.log("SIMULATION SPECULATION OUTPUT", speculation)
        
        return speculation, time.perf_counter() - start_time_speculate, input_tokens, output_tokens, total_tokens
    
    def _execute_game_loop(
        self,
        agents: Dict[int, Any],
        stop_after: Optional[int] = None
    ) -> Tuple[Dict[int, Any], Any, Any, float, float]:
        """Main game execution loop"""
        self.env.reset(num_players=self.config.num_chess_players)
        
        steps_info = {}
        step_count = 0
        done = False
        is_initial_step = True
        
        current_agent = agents[0]
        other_agent = agents[1]
        
        regular_time = 0.0
        speculative_time = 0.0

        temp_time_holder = 0.0

        # Initialize game state
        player_id, observation = self.env.get_observation()
        
        while not done:
            # Handle prediction matching from previous step
            if not is_initial_step:
                prev_predictions = steps_info[step_count - 1]["current_pred"]
                prev_speculations = steps_info[step_count - 1]["current_spec"]
                prev_move = steps_info[step_count - 1]["current_move"]

            player_id, observation = self.env.get_observation()

            # Use the first speculation that matches the previous move
            speculation_hit = False
            current_move = None
            if not is_initial_step and prev_predictions:
                for i, pred in enumerate(prev_predictions):
                    if pred == prev_move and i < len(prev_speculations):
                        current_move = prev_speculations[i]
                        speculation_hit = True
                        break

            if speculation_hit:
                current_predictions : List[Optional[str]] = []
                current_speculations : List[Optional[str]] = []
                regular_time += temp_time_holder
                time_taken1 = 0
                times_taken2 = []
                prediction_times = []
                speculation_times = []
                input_prediction_tokens = []
                output_prediction_tokens = []
                total_prediction_tokens = []
                input_speculation_tokens = []
                output_speculation_tokens = []
                total_speculation_tokens = []
                input_tokens1 = []
                output_tokens1 = []
                total_tokens1 = []
            else:
                # Run parallel execution
                with ThreadPoolExecutor() as executor:
                    current_future = executor.submit(
                        self._current_agent_task, current_agent, observation, player_id
                    )
                    speculation_future = executor.submit(
                        self._speculation_task, other_agent, observation, player_id
                    )
                    
                    current_move, time_taken1, input_tokens1, output_tokens1, total_tokens1 = current_future.result()
                    current_predictions, current_speculations, prediction_times, speculation_times, times_taken2, input_prediction_tokens, output_prediction_tokens, total_prediction_tokens, input_speculation_tokens, output_speculation_tokens, total_speculation_tokens = speculation_future.result()
                    
                    if is_initial_step:
                        is_initial_step = False
                
                # Update timing counters
                regular_time += time_taken1

                if current_move in current_predictions:
                    speculation_hit_index = current_predictions.index(current_move)
                    speculative_time += max(time_taken1, times_taken2[speculation_hit_index] or 0)
                    temp_time_holder = speculation_times[speculation_hit_index] or 0
                else:
                    speculative_time += time_taken1
            
            # Record step information
            steps_info[step_count] = {
                "player_id": player_id,
                "current_observation": observation,
                "current_move": current_move,
                "current_pred": current_predictions,
                "current_spec": current_speculations,
                "time_taken_current_agent": time_taken1,
                "time_taken_other_agent": times_taken2,
                "time_taken_prediction": prediction_times,
                "time_taken_speculation": speculation_times,
                "speculation_hit": speculation_hit,
                "num_predictions": len(current_predictions) if current_predictions else 0,
                "input_tokens_current_agent": input_tokens1,
                "output_tokens_current_agent": output_tokens1,
                "total_tokens_current_agent": total_tokens1,
                "input_tokens_prediction": input_prediction_tokens,
                "output_tokens_prediction": output_prediction_tokens,
                "total_tokens_prediction": total_prediction_tokens,
                "input_tokens_speculation": input_speculation_tokens,
                "output_tokens_speculation": output_speculation_tokens,
                "total_tokens_speculation": total_speculation_tokens,
            }
            
            if self.logger:
                self.logger.log("INFO", f"STEP {step_count}:", Utils.dict_to_str(steps_info[step_count]))
                self.logger.log('-' * 100)
            
            # Execute move
            done, info = self.env.step(current_move)
            step_count += 1
            
            if stop_after and step_count >= stop_after:
                break
            
            # Swap agents for next turn
            current_agent, other_agent = other_agent, current_agent
        
        rewards, game_info = self.env.close()
        return steps_info, rewards, game_info, regular_time, speculative_time
    
    def run(self, stop_after: int = 20) -> None:
        """Run a complete chess game with speculative execution"""
        # Setup run
        self.current_run_id = str(uuid.uuid4())
        self.logger = GameLogger(self.base_traj_path, self.current_run_id)
        
        current_dir_path = join(self.base_traj_path, self.current_run_id)
        
        # Create agents
        agents = self.agent_manager.create_agents(self.agent0_name, self.agent1_name)
        
        self.logger.log("INFO", f"Starting run {self.current_run_id} with guess model: {self.guess_model_name}")
        
        try:
            # Execute game
            steps_info, rewards, game_info, regular_time, speculative_time = self._execute_game_loop(
                agents, stop_after
            )
            
            # Save results
            Utils.save_json(steps_info, join(current_dir_path, "stepsinfo.json"))
            Utils.save_json(rewards, join(current_dir_path, "rewards.json"))
            Utils.save_json(game_info, join(current_dir_path, "game_info.json"))
            Utils.save_json(regular_time, join(current_dir_path, "time_checker_regular.json"))
            Utils.save_json(speculative_time, join(current_dir_path, "time_checker_speculate.json"))
            
            self.logger.log("INFO", f"Run completed for {self.current_run_id}")
            
        except Exception as e:
            if self.logger:
                self.logger.log("ERROR", str(e))
            raise


def main():
    """Main execution function"""
    config_path = "./config.yml"
    config = Config(config_path)
    agent_name0 = config.agent_name0
    agent_name1 = config.agent_name1
    guess_model_name = config.guess_model_name
    num_guesses = config.num_guesses
    runner = SpeculativeChessRunner(
        agent0_name=agent_name0,
        agent1_name=agent_name1,
        guess_model_name=guess_model_name,
        num_guesses=num_guesses
    )
    
    start_time = time.time()
    runner.run(stop_after=config.stop_after)
    end_time = time.time()
    
    print(f"Total execution time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()