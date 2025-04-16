import copy
from copy import deepcopy
from typing import List, Dict, Tuple
from string import ascii_lowercase as letters

import numpy as np

import clemcore.clemgame.metrics as ms
from clemcore.clemgame import GameMaster, DialogueGameMaster, GameBenchmark, GameSpec, GameScorer
from clemcore.backends import Model
from clemcore.clemgame import Player

from player import Speaker


class FirstLast(DialogueGameMaster):
    """Implement mechanisms for playing FirstLast."""
    def __init__(self, game_name: str, game_path: str, experiment: Dict, player_models: List[Model]):
        super().__init__(game_name, game_path, experiment, player_models)
        # assign experiment attributes that will be necessary later
        self.topic = experiment['name']

    def _on_setup(self, **game_instance) -> None:
        """
        Set up the episode (mandatory).
        Args:
            game_instance: The game instance dict.
        """
        self.game_instance = game_instance

        self.n_turns = game_instance['n_turns']

        # instantiate both players:
        self.player_a = Speaker(self.player_models[0], 'A', game_instance['first_letter'])
        self.player_b = Speaker(self.player_models[1], 'B', game_instance['first_letter'])

        # add players, including assigning their initial prompts:
        # self.add_player(self.player_a, initial_prompt=game_instance['prompt_player_a'])
        self.add_player(self.player_a, initial_context=game_instance['prompt_player_a'])
        self.add_player(self.player_b, initial_prompt=game_instance['prompt_player_b'])

        # initialise game variables:
        self.current_turn: int = 0
        self.current_letter: str = game_instance['first_letter']

        # initialise common metrics:
        self.request_count: int = 0
        self.parsed_request_count: int = 0
        self.violated_request_count: int = 0

        # initialise attributes that will be used for the evaluation scores
        self.aborted: bool = False
        self.lose: bool = False
        self.complete_turns: int = 0

        self.correct_response = False
        self.turn_scores = [0] * (self.n_turns + 1)

        # log any additional keys that will be relevant for evaluation
        self.log_key('n_turns', game_instance['n_turns'])

    def _validate_player_response(self, player: Player, response: str) -> bool:
        """
        Check if the response follows the move format rule.
        Args:
            player: The player that produced the response.
            response: The response string.
        Returns:
            True if the move format rule was followed, else False.
        """
        # increase the number of API requests:
        self.request_count += 1
        # check move format rule:
        if not response.startswith('I SAY:'):
            self.aborted = True
            # log the abortion event
            action = {'type': 'invalid format', 'content': 'abort'}
            self.log_event(from_='GM', to='GM', action=action)
            # increase the counter of requests that violate form rules
            self.violated_request_count += 1
            return False
        else:
            # increase the counter of requests that conform to form rules
            self.parsed_request_count += 1
            # log the event that the string was valid (no strange characters)
            action = {'type': 'metadata', 'content': 'valid string'}
            self.log_event(from_='GM', to='GM', action=action)
        return True

    def _parse_response(self, player: Player, response: str) -> Tuple[str, str]:
        """
        Add the response to the other player's message history and split the response and return the first and last word.
        """
        if player == self.player_a:
            self.set_context_for(self.player_b, response)
        if player == self.player_b:
            self.set_context_for(self.player_a, response)
        # remove the move format tag and split on whitespace:
        words = response[7:].split()
        return words[0].lower(), words[-1].lower()

    def _on_valid_player_response(self, player: Player, parsed_response: Tuple[str, str]):
        """
        Check player response for game rule adherence.
        Args:
            player: The current player.
            parsed_response: The parsed response, a tuple of the first and last word of the response.
        """
        first_word_correct_letter = parsed_response[0][0] == self.current_letter  # True if the first letter of the first word in the response is correct
        last_word_correct_letter = parsed_response[0][0] == parsed_response[1][0]  # True if the first letters of the first and last word match
        self.correct_response = first_word_correct_letter and last_word_correct_letter
        if not self.correct_response:
            self.lose = True
            # log the fact that the game is now lost
            action = {'type': 'parse',
                      'content': f'{parsed_response[0]}/{parsed_response[1]} violates rules'}
            self.log_event(from_='GM', to='GM', action=action)
        else:
            # log the fact that the answer was correct
            action = {'type': 'parse',
                      'content': f'{parsed_response[0]}/{parsed_response[1]} conforms to rules'}
            self.log_event(from_='GM', to='GM', action=action)

    def compute_response_score(self, response: str, context: Dict):
        """
        Compute a score for a player response.
        Args:
            response: The player response string to be scored.
            context: The context message that was added to the player message history to produce the response.
        Returns:
            1 if the firstlast game rules were followed, 0 otherwise.
        """
        response_correct: bool = False
        # move format rule:
        if response.startswith('I SAY:'):
            # remove the move format tag and split on whitespace:
            words: list = response[7:].split()
            # check for first letter rule:
            first_word_correct_letter = words[0][0] == self.current_letter
            last_word_correct_letter = words[0][0] == words[-1][0]
            response_correct = first_word_correct_letter and last_word_correct_letter

        self.turn_scores[self.current_turn] += 1

        return 1 if response_correct else 0

    def compute_episode_score(self):
        """
        Calculate a score for the episode.
        When this is reached and the last turn had a correct response, the episode gets full score.
        """
        if self.correct_response:
            return 100
        return 0

    def _does_game_proceed(self) -> bool:
        """Check if game should proceed."""
        return (self.current_turn < self.n_turns
                and not self.aborted
                and not self.lose)

    def _start_next_round(self) -> bool:
        """Start next round after each player's turn.
        Returns:
            True
        """
        return True

    def _on_after_round(self):
        """Updates the letter being played after a successful round."""
        # update the letter being played:
        current_index = letters.index(self.current_letter)
        self.current_letter = letters[current_index + 1]
        # increment firstlast turns:
        self.current_turn += 1

    def _on_after_game(self) -> None:
        """Log variables needed for scoring."""
        # log a message informing that the game was successfully played:
        if not self.aborted and not self.lose:
            action = {'type': 'info', 'content': 'game successful'}
            self.log_event(from_='GM', to='GM', action=action)
        # log a final message saying that the game did come to an end:
        action = {'type': 'info', 'content': 'end game'}
        self.log_event(from_='GM', to='GM', action=action)
        # log firstlast-specific values:
        self.log_key('Played turns', self.current_turn)
        self.log_key('Complete turns', self.complete_turns)
        self.log_key('Turn scores', self.turn_scores)
        # log standard metrics:
        self.log_key(ms.METRIC_ABORTED, self.aborted)
        self.log_key(ms.METRIC_LOSE, self.lose)
        self.log_key(ms.METRIC_REQUEST_COUNT, self.request_count)
        self.log_key(ms.METRIC_REQUEST_COUNT_PARSED, self.parsed_request_count)
        self.log_key(ms.METRIC_REQUEST_COUNT_VIOLATED, self.violated_request_count)


class FirstLastScorer(GameScorer):
    """Scorer for the firstlast game."""
    def __init__(self, game_name: str, experiment: Dict, game_instance: Dict):
        super().__init__(game_name, experiment, game_instance)

    def score_turns(self, episode_interactions: Dict) -> None:
        """Calculate and log turn-level scores."""
        played_turns = episode_interactions['Played turns']
        turn_scores = episode_interactions['Turn scores']
        for turn in range(0, played_turns):
            self.log_turn_score(turn, "turn score", turn_scores[turn])

    def log_main_score(self, episode_interactions: Dict):
        complete_turns = episode_interactions['Complete turns']
        n_turns = episode_interactions['n_turns']
        aborted = int(episode_interactions[ms.METRIC_ABORTED])
        # IMPORTANT: aborted episodes MUST have a bench score of NaN!
        bench_score = complete_turns / n_turns if not aborted else np.nan
        self.log_episode_score(ms.BENCH_SCORE, bench_score)


class FirstLastGameBenchmark(GameBenchmark):
    """Integrate the game into the benchmark."""
    def __init__(self, game_spec: GameSpec):
        super().__init__(game_spec)

    def create_game_master(self,
                           experiment: Dict,
                           player_models: List[Model]
                           ) -> GameMaster:
        return FirstLast(self.game_name, self.game_path, experiment, player_models)

    def create_game_scorer(self, experiment: Dict, game_instance: Dict) -> GameScorer:
        return FirstLastScorer(self.game_name, experiment, game_instance)