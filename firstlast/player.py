# save the contents of this cell as firstlast/players.py

import random
import logging
from string import ascii_lowercase as letters
from typing import List, Dict, Union

from clemcore.clemgame import Player
from clemcore.backends import Model

# initialize logging:
logger = logging.getLogger(__name__)

class Speaker(Player):
    def __init__(self, model: Model, letter: str, firstlast_player: str, name: str = None):
        # if the player is a program and you don't want to make API calls to
        # LLMS, use model_name="programmatic"
        # TODO: check how programmatic is passed with ModelSpec
        super().__init__(model, name)
        self.player: str = firstlast_player
        self.initial_letter: str = letter

    # implement this method as you prefer, with these same arguments
    def _custom_response(self, context: Dict) -> str:
        """Return a mock message with the suitable letter and format.
        Args:
            context: The dialogue context to which the player should respond. Base class method, not used in this example.
        Returns:
            Mock message with the suitable letter and format.
        """
        # get the first letter of the content of the last message
        # messages is a list of dictionaries with messages in openai API format
        turn_idx = len(self._messages)  # will be 1 if only initial prompt message is in message history

        if turn_idx == 1 and self.player == 'A':
            letter = 'I SAY: ' + self.initial_letter
        else:
            previous_letter = self._messages[-1]['content'][7].lower()
            # introduce a small probability that the player fails
            letter = self._sample_letter(previous_letter)
        # return a string whose first and last tokens start with the next letter
        return f"{letter}xxx from {self.player}, turn {turn_idx} {letter.replace('I SAY: ', '')}xxx."

    # an additional method specific for this game
    # for testing, we want the utterances to be invalid or incorrect sometimes
    def _sample_letter(self, letter: str) -> str:
        """Randomly decide which letter to use in a custom response message."""
        prob = random.random()
        index = letters.index(letter)
        if prob < 0.05:
            # correct but invalid (no tag)
            return letters[index + 1]
        if prob < 0.1:
            # valid tag but wrong letter
            return 'I SAY: ' + letter
        # valid and correct
        return 'I SAY: ' + letters[index + 1]