import copy
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
        """Set up the episode (mandatory)."""
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
        self.request_counts = [0] * (self.n_turns + 1)
        self.parsed_request_counts = [0] * (self.n_turns + 1)
        self.violated_request_counts = [0] * (self.n_turns + 1)

        # initialise attributes that will be used for the evaluation scores
        self.aborted: bool = False
        self.lose: bool = False
        self.complete_turns: int = 0

        self.correct_response = False

        # log any additional keys that will be relevant for evaluation
        self.log_key('n_turns', game_instance['n_turns'])

    def _validate_player_response(self, player: Player, response: str) -> bool:
        """Check if the response follows the move format rule.
        Args:
            player: The player that produced the response.
            response: The response string.
        Returns:
            True if the move format rule was followed, else False.
        """
        # increase the number of API requests:
        self.request_counts[self.current_turn] += 1
        # check move format rule:
        if not response.startswith('I SAY:'):
            self.aborted = True
            # log the abortion event
            action = {'type': 'invalid format', 'content': 'abort'}
            self.log_event(from_='GM', to='GM', action=action)
            # increase the counter of requests that violate form rules
            self.violated_request_counts[self.current_turn] += 1
            return False
        else:
            # increase the counter of requests that conform to form rules
            self.parsed_request_counts[self.current_turn] += 1
            # log the event that the string was valid (no strange characters)
            action = {'type': 'metadata', 'content': 'valid string'}
            self.log_event(from_='GM', to='GM', action=action)
        return True

    def _parse_response(self, player: Player, response: str) -> Tuple[str, str]:
        """Split the response and return the first and last word."""
        if player == self.player_a:
            self.set_context_for(self.player_b, response)
        if player == self.guesser:
            self.set_context_for(self.describer, response)
        # remove the move format tag and split on whitespace:
        words = response[7:].split()
        return words[0], words[-1]

    def _on_valid_player_response(self, player: Player, parsed_response: Tuple[str, str]):
        """Check player response for game rule adherence."""
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
        """Compute a score for a player response.
        Args:
            response: The player response string to be scored.
            context: The context message that was added to the player message history to produce the response.
        Returns:
            1 if the firstlast game rules were followed, 0 otherwise.
        """
        # TODO: figure out a way to make this work properly ... or get the DGM.step() order fixed...
        """
        if self._validate_player_response(self.current_player, response):
            parsed_response = self._parse_response(self.current_player, response)
            self._on_valid_player_response(self.current_player, parsed_response) # this method sets self.correct_response to True or False
            return 1 if self.correct_response else 0
        else:
            return 0
        """
        return 1 if self.correct_response else 0  # this will be based on the turn/round before...

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

        self.log_key('Played turns', self.current_turn)
        self.log_key('Complete turns', self.complete_turns)
        self.log_key(ms.METRIC_ABORTED, self.aborted)
        self.log_key(ms.METRIC_LOSE, self.lose)
        self.log_key(ms.METRIC_REQUEST_COUNT, self.request_counts)
        self.log_key(ms.METRIC_REQUEST_COUNT_PARSED, self.parsed_request_counts)
        self.log_key(ms.METRIC_REQUEST_COUNT_VIOLATED, self.violated_request_counts)


class FirstLastScorer(GameScorer):
    """Scorer for the firstlast game."""
    def __init__(self, game_name: str, experiment: Dict, game_instance: Dict):
        super().__init__(game_name, experiment, game_instance)

    def compute_scores(self, episode_interactions: Dict) -> None:
        """Compute episode-level and turn-level scores (mandatory)."""
        played_turns = episode_interactions['Played turns']
        complete_turns = episode_interactions['Complete turns']
        # reqs = episode_interactions[ms.METRIC_REQUEST_COUNT][1:]
        reqs = episode_interactions[ms.METRIC_REQUEST_COUNT]
        p_reqs = episode_interactions[ms.METRIC_REQUEST_COUNT_PARSED]
        v_reqs = episode_interactions[ms.METRIC_REQUEST_COUNT_VIOLATED]
        n_turns = len(reqs)

        for turn in range(0, played_turns):
            self.log_turn_score(turn, ms.METRIC_REQUEST_COUNT, reqs[turn])
            self.log_turn_score(turn, ms.METRIC_REQUEST_COUNT_PARSED, p_reqs[turn])
            self.log_turn_score(turn, ms.METRIC_REQUEST_COUNT_VIOLATED, v_reqs[turn])

        aborted = int(episode_interactions[ms.METRIC_ABORTED])
        lose = int(episode_interactions[ms.METRIC_LOSE]) if not aborted else 0
        success =  1 - lose if not aborted else 0
        bench_score = complete_turns / n_turns if not aborted else np.nan

        self.log_episode_score(ms.METRIC_ABORTED, aborted)
        self.log_episode_score(ms.METRIC_LOSE, lose)
        self.log_episode_score(ms.METRIC_SUCCESS, success)
        self.log_episode_score(ms.METRIC_REQUEST_COUNT, sum(reqs))
        self.log_episode_score(ms.METRIC_REQUEST_COUNT_PARSED, sum(p_reqs))
        self.log_episode_score(ms.METRIC_REQUEST_COUNT_VIOLATED, sum(v_reqs))
        self.log_episode_score(ms.METRIC_REQUEST_SUCCESS, sum(p_reqs) / sum(reqs))
        self.log_episode_score(ms.BENCH_SCORE, bench_score)


class FirstLastGameBenchmark(GameBenchmark):
    """Integrate the game into the benchmark run."""
    def __init__(self, game_spec: GameSpec):
        super().__init__(game_spec)
        print(f"FirstLastGameBenchmark initilzed, self.game_name: {self.game_name}")

    def create_game_master(self,
                           experiment: Dict,
                           player_models: List[Model]
                           ) -> GameMaster:
        return FirstLast(self.game_name, self.game_path, experiment, player_models)

    def create_game_scorer(self, experiment: Dict, game_instance: Dict) -> GameScorer:
        return FirstLastScorer(self.game_name, experiment, game_instance)