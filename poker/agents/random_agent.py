import clubs
from . import BaseAgent

import random


class RandomAgent(BaseAgent):
    """A random agent that randomly folds, calls, or raises."""
    def __init__(self) -> None:
        super().__init__()
        self.name = "Random Agent"
        self.short_name = "Random"

    def act(self, obs: clubs.poker.engine.ObservationDict) -> int:
        call = obs['call']
        max_raise = obs['max_raise']

        bet = random.randint(0, max(max_raise, call))
        return bet
