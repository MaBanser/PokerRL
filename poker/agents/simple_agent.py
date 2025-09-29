import clubs
from . import BaseAgent

import random


class SimpleAgent(BaseAgent):
    """A simple agent that randomly folds, calls, or raises based on predefined probabilities.
    Parameters:
        fold_probability (float): Probability of folding, default is 0.3.
        call_probability (float): Probability of calling, default is 0.4.
        raise_probability (float): Probability of raising, default is 0.3.
    The probabilities should sum to 1. If they do not, they will be normalized.
    """
    def __init__(self, fold_probability = 0.3, call_probability = 0.4, raise_probability = 0.3) -> None:
        super().__init__()
        # Normalize the probabilities
        sum_probabilities = fold_probability + call_probability + raise_probability
        self.fold_probability = (1/ sum_probabilities) * fold_probability
        self.call_probability = (1/ sum_probabilities) * call_probability
        self.raise_probability = (1/ sum_probabilities) * raise_probability
        
        self.name = f'Simple Agent\
            ({int(self.fold_probability*100)}/\
            {int(self.call_probability*100)}/\
            {int(self.raise_probability*100)})'
        self.short_name = 'Simple'

    def act(self, obs: clubs.poker.engine.ObservationDict) -> int:
        call = obs['call']
        min_raise = obs['min_raise']
        max_raise = obs['max_raise']
        rand = random.random()
        if rand < self.fold_probability:
            bet = 0
        elif rand < self.fold_probability + self.call_probability:
            bet = call
        else:
            bet = random.randint(min_raise, max_raise)
        return bet
