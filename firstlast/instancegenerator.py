import os
import random
import string
import logging

from clemcore.clemgame import GameInstanceGenerator

# initialize logging:
logger = logging.getLogger(__name__)

# set the name of the game in the script, as you named the directory
# this name will be used everywhere, including in the table of results
GAME_NAME = 'firstlast'
# we will create 10 instances for each experiment; vary this as you wish
N_INSTANCES = 10
# if the generation involves randomness, remember to set a random seed
SEED = 123

class FirstLastGameInstanceGenerator(GameInstanceGenerator):
    def __init__(self):
        # always do this to initialise GameInstanceGenerator
        super().__init__(os.path.dirname(__file__))

    # define on_generate, a mandatory method
    def on_generate(self):
        # get the list of topics, which will be our experiments
        topics = self.load_file('resources/topics.txt').strip('\n').split('\n')
        # get the prompts for player a and player b
        # we'll keep the prompts fixed in all instances, replacing only the
        # necessary slots (but you can do it differently)
        prompt_a = self.load_template('resources/initial_prompts/initial_prompt_a')
        prompt_b = self.load_template('resources/initial_prompts/initial_prompt_b')

        # building the file, one experiment at a time
        for topic in topics:
            # create an experiment (for us, named after a topic)
            experiment = self.add_experiment(topic)
            # build N_INSTANCES instances for each experiment
            for game_id in range(N_INSTANCES):
                # set the parameters
                # here we do it randomly, but that can also be read from a file
                # one of the first 5 letters in the alphabet
                letter = random.choice(string.ascii_lowercase[:5])
                # up to 8 turns, so that we don't run out of letters
                n_turns = random.randint(3, 8)
                # create a game instance, using a game_id counter/index
                instance = self.add_game_instance(experiment, game_id)
                # populate the game instance with its parameters
                instance['first_letter'] = letter
                instance['n_turns'] = n_turns
                instance['prompt_player_a'] = self.create_prompt(
                    topic, prompt_a, letter, n_turns)
                instance['prompt_player_b'] = self.create_prompt(
                    topic, prompt_b, letter, n_turns)

    # an additional method, specific for our example
    def create_prompt(self,
                      topic: str,
                      prompt: str,
                      letter: str,
                      n_turns: int) -> str:
        """Replace a prompt template with slot values."""
        text = string.Template(prompt).substitute(topic=topic, letter=letter,
                                                  n_turns=n_turns)
        return text


if __name__ == '__main__':
    random.seed(SEED)
    # always call this, which will actually generate and save the JSON file
    FirstLastGameInstanceGenerator().generate()