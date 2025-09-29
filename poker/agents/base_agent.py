import clubs


class BaseAgent:
    """Base class for all agents in the poker game.
    This class defines the basic structure and interface for agents.
    Agents should inherit from this class and implement the `act` method.
    Attributes:
        name (str): The name of the agent.
        is_training (bool): Indicates whether the agent is in training mode.
    """
    def __init__(self) -> None:
        self.name = 'Base Agent'
        self.short_name = 'Base'
        self.is_training = False

    def act(self, obs: clubs.poker.engine.ObservationDict) -> int:
        raise NotImplementedError()