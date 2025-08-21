from re import L
from typing import Any, List, Optional, Tuple, Type, Union, Literal, TypedDict

from clubs.poker import Dealer
from clubs.configs import PokerConfig
from clubs.poker.engine import ObservationDict
from poker import agents
from poker.agents import BaseAgent

class ClubsEnv(Dealer):
    """
    Wrapper for gym like Clubs poker environment.
    This class extends the Dealer class from clubs.poker to register agents
    that can automatically interact with the poker environment.
    Agents have to inherit from poker.agents.BaseAgent and implement the act method.

    Args:
    ----------
    config : PokerConfig
        configuration for the poker game, see poker/clubs/clubs/configs.py for details
    agents : List[Type[BaseAgent]]
        list of agents that can interact with the environment, agents
        have to inherit from poker.agents.BaseAgent and implement the act method
    """

    def __init__(self, config: PokerConfig, agents: List[Type[BaseAgent]]) -> None:
        super().__init__(**config)
        self.agents = agents

    def act(self, obs: ObservationDict) -> int:
        """
        Method to let the currently active agent act based on the current observation.
        Returns the bet amount that the agent decides to make.

        Parameters
        ----------
        obs : ObservationDict
            current observation of the game, see clubs.poker.engine.ObservationDict for details

        Returns
        -------
        bet : int
            the bet amount that the agent decides to make
        """
        current_agent = self.agents[obs['action']]
        bet = current_agent.act(obs)
        return bet

    def render(self, mode = "human", sleep = 0, **kwargs):
        return super().render(mode, sleep, agents=self.agents, **kwargs)
