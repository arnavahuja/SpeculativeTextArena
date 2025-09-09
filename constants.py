num_chess_players = 2
client_error_sleep_time = 60
server_error_sleep_time = 60
gemini_api_key = ""
openai_api_key = ""
gemini_model_name = "gemini-2.5-flash"
gemini_guess_model_name = "gemini-2.0-flash"
openai_model_name = "gpt-4.1-turbo"
openai_guess_model_name = "gpt-4.1-mini"
chess_guess_prompt = "Based on these observations and moves given, guess the next move in the format mentioned. Action must always be present in the output and in a legal UCI format chess move like [e2e4]. Do not under any circumstance give a response without an action in legal uci format"
react_instruction = """
Solve a chess playing task with interleaving Thought and Action steps.
Thought can reason about the current chess board situation, possible strategies, threats, or tactics.
Action must always be a legal UCI format chess move like [e2e4].
Always reply in this format:
Thought: <your reasoning about the position>
Action: <your chosen UCI move>

Example:

Thought: I want to control the center early, so I will push the king's pawn.
Action: [e2e4]
"""