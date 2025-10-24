import clubs
from typing import Union


class BaseAgent:
    """Base class for all agents in the poker game.
    This class defines the basic structure and interface for agents.
    Agents should inherit from this class and implement the `act` method.
    
    Attributes:
        name (str): The name of the agent.
        is_training (bool): Indicates whether the agent is in training mode.
        action_agg (dict): A dictionary to aggregate actions taken by the agent.
    """
    def __init__(self) -> None:
        self.name = 'Base Agent'
        self.short_name = 'Base'
        self.is_training = False        
        self.action_agg = {}

    def act(self, obs: clubs.poker.engine.ObservationDict) -> int:
        raise NotImplementedError()
    
    def reset(self) -> None:
        """Resets the agent's internal state. This method should be overridden by agents that maintain state."""
        pass

    def finalize_hand(self, obs: clubs.poker.engine.ObservationDict, reward: float) -> None:
        """Finalizes the agent's state at the end of a hand. This method should be overridden by agents that require this functionality."""
        pass
    
    def train(self) -> Union[None, dict]:
        """Trains the agent. This method should be overridden by agents that require training."""
        pass

    def set_eval(self) -> None:
        """Sets the agent to evaluation mode. This method should be overridden by agents that require this functionality."""
        self.is_training = False

    def set_train(self) -> None:
        """Sets the agent to training mode. This method should be overridden by agents that require this functionality."""
        self.is_training = True

    def warm_up(self, is_warm_up: bool = False):
        """Sets the agent to training mode. This method should be overridden by agents that require this functionality."""
        pass