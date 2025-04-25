import os.path
from typing import Dict, Tuple, List, Union
from string import ascii_lowercase as letters
import logging

import numpy as np

import clemcore.metrics as ms
from clemcore.clemgame import GameSpec, GameMaster, GameBenchmark, Player, DialogueGameMaster, GameScorer, GameRecorder, \
    GameException, ParseError, ValidationError
from clemcore.backends import Model
from clemcore.utils import file_utils, string_utils

# import the Speaker player class:
from player import Speaker

# initialize logging:
logger = logging.getLogger(__name__)


class FirstLast(DialogueGameMaster):
    """Implement mechanisms for playing FirstLast."""

    def __init__(self, game_name: str, game_path: str, experiment: Dict, player_models: List[Model]):
        super().__init__(game_name, game_path, experiment, player_models)

    def _on_setup(self, **game_instance) -> None:
        """
        Set up the episode (mandatory).
        Args:
            game_instance: The game instance dict.
        """
        self.game_instance: dict = game_instance

        self.n_turns: int = game_instance['n_turns']

        # instantiate both players:
        self.player_a = Speaker(self.player_models[0], 'A', game_instance['first_letter'])
        self.player_b = Speaker(self.player_models[1], 'B', game_instance['first_letter'])

        # add players, including assigning their initial prompts:
        self.add_player(self.player_a, initial_context=game_instance['prompt_player_a'])
        self.add_player(self.player_b, initial_prompt=game_instance['prompt_player_b'])

        # initialise game variables:
        self.current_turn: int = 0
        self.current_letter: str = game_instance['first_letter']
        # log any additional keys that will be relevant for evaluation
        self.log_key('n_turns', game_instance['n_turns'])

        self.correct_response = False
        self.turn_scores = [0] * (self.n_turns + 1)
        self.complete_turns: int = 0

        # initialise common metrics:
        self.request_count: int = 0
        self.parsed_request_count: int = 0
        self.violated_request_count: int = 0

        # initialise attributes that will be used for the evaluation scores
        self.aborted: bool = False
        self.lose: bool = False

    def _parse_response(self, player: Player, response: str) -> Tuple[str, str]:
        """
        Add the response to the other player's message history and check if the response follows the move format rule,
        then split the response and return the first and last word.
        Args:
            player: The player that produced the response.
            response: The response string.
        Returns:
            Tuple of the first and last word of the response.
        Raises:
            ParseError: If the response is missing 'I SAY: '.
        """
        # increase the number of API requests:
        self.request_count += 1

        if player == self.player_a:
            self.set_context_for(self.player_b, response)
        if player == self.player_b:
            self.set_context_for(self.player_a, response)

        # check for move format tag:
        if not response.startswith("I SAY: "):
            raise ParseError()

        # increase the counter of requests that conform to form rules
        self.parsed_request_count += 1
        # log the event that the string was valid (no strange characters)
        action = {'type': 'metadata', 'content': 'move format followed'}
        self.log_event(from_='GM', to='GM', action=action)

        # remove the move format tag and split on whitespace:
        words = response[7:].split()
        return words[0].lower(), words[-1].lower()

    def _on_parse_error(self, error: GameException):
        """Abort the game due to failed parsing."""
        # set the game to be aborted:
        self.aborted = True
        # increase the counter of requests that violate the move format rule:
        self.violated_request_count += 1
        # log the abortion event:
        action = {'type': 'missing tag', 'content': 'abort'}
        self.log_event(from_='GM', to='GM', action=action)

    def _validate_player_response(self, player: Player, parsed_response: str):
        """
        Check if the parsed response follows the game rules.
        Args:
            player: The player that produced the response.
            response: The response string.
        Returns:
            True if the game rules were followed, else False.
        """
        first_word_correct_letter = parsed_response[0][
                                        0] == self.current_letter  # True if the first letter of the first word in the response is correct
        last_word_correct_letter = parsed_response[0][0] == parsed_response[1][
            0]  # True if the first letters of the first and last word match
        self.correct_response = first_word_correct_letter and last_word_correct_letter
        if not self.correct_response:
            # log the fact that the game is now lost:
            action = {'type': 'rule violation',
                      'content': f'{parsed_response[0]}/{parsed_response[1]} violates rules'}
            # Note: logged here and not in _on_validation_error() to record the actual words that violated the rules
            self.log_event(from_='GM', to='GM', action=action)
            raise ValidationError()
        else:
            # log the fact that the answer was correct:
            action = {'type': 'valid response',
                      'content': f'{parsed_response[0]}/{parsed_response[1]} conforms to rules'}
            self.log_event(from_='GM', to='GM', action=action)
            # set the current turn's score to 1:
            self.turn_scores[self.current_turn] = 1

    def _on_validation_error(self, error: GameException):
        """Lose the game due to violated rules."""
        self.lose = True

    def _on_valid_player_response(self, player: Player, parsed_response: Tuple[str, str]):
        """
        Advance the game state, preparing the next player's turn.
        Args:
            player: The current player.
            parsed_response: The parsed response, a tuple of the first and last word of the response.
        """
        # increment current turn:
        self.current_turn += 1
        # increment completed turns:
        self.complete_turns += 1
        # update the letter being played:
        current_index = letters.index(self.current_letter)
        self.current_letter = letters[current_index + 1]

    def compute_response_score(self, response: str, context: Dict):
        """
        Compute a score for a player response.
        Args:
            response: The player response string to be scored.
            context: The context message that was added to the player message history to produce the response.
        Returns:
            1 if the firstlast game rules were followed, 0 otherwise.
        """
        return self.turn_scores[self.current_turn - 1]

    def _does_game_proceed(self) -> bool:
        """Check if game should proceed."""
        return (self.current_turn < self.n_turns
                and not self.aborted
                and not self.lose)

    def compute_episode_score(self):
        """
        Calculate a score for the episode based on successful turns and target number of turns.
        Returns:
            Episode score value in range 0-100.
        """
        turn_score_sum = sum(self.turn_scores)
        success_ratio = turn_score_sum / self.n_turns
        return success_ratio * 100

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