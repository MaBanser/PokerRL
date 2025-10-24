from typing import List, Type, Tuple

from clubs import Dealer, Card
from clubs.configs import PokerConfig
from clubs.poker.engine import ObservationDict
from poker.agents import BaseAgent

class VerboseObservationDict(ObservationDict):
    action: int
    active: List[bool]
    button: int
    big_blind: int
    call: int
    community_cards: List[Card]
    hole_cards: List[Card]
    max_raise: int
    min_raise: int
    pot: int
    stacks: List[int]
    street_commits: List[int]
    pot_commits: List[int]
    street: int
    num_streets: int
    num_players: int
    num_hole_cards: int
    num_community_cards: int
    has_acted: List[bool]
    history: List[Tuple[int, int, bool]]
    street_raises: int

class ClubsEnv(Dealer):
    def __init__(self, config: PokerConfig, agents: List[Type[BaseAgent]]) -> None:
        """
        Wrapper for Clubs poker environment.
        This class extends the Dealer class from clubs.poker to register agents
        that can automatically interact with the poker environment.
        Agents have to inherit from poker.agents.BaseAgent and implement the act method.
        Agents can be set to training mode, which means that the environment can call
        the train method of the agent.

        Args:
            config (PokerConfig): Configuration for the poker game, see poker/clubs/clubs/configs.py for details.
            agents (List[Type[BaseAgent]]): List of agents that can interact with the environment, 
                                            agents have to inherit from poker.agents.BaseAgent and implement the act method.
        """
        super().__init__(**config)
        self.agents = agents

    def act(self, obs: ObservationDict, verbose=False) -> int:
        """
        Method to let the currently active agent act based on the current observation.
        Returns the bet amount that the agent decides to make.

        Args:
            obs (ObservationDict): Current observation of the game, see clubs.poker.engine.ObservationDict for details.

        Returns:
            bet (int): The bet amount that the agent decides to make.
        """
        current_agent = self.agents[obs['action']]
        bet = current_agent.act(self._expand_obs(obs))
        if verbose:
            print(f'{current_agent.name} bets: {bet}')
        return bet
    
    def step(self, bet) -> Tuple[VerboseObservationDict, List[float], List[bool], dict]:
        """
        Takes a step in the environment based on the bet made by the current agent.
        Normalizes the reward based on the big blind and finalizes the hand for all agents if the round is done.

        Args:
            bet (int): The bet amount made by the current agent.

        Returns:
            observation (VerboseObservationDict): The next observation after taking the step.
            reward (List[float]): List of rewards for each agent after taking the step.
            done (List[bool]): List indicating whether each agent's episode has ended.
            info (dict): Additional information about the step.
        """
        obs, reward, done, info = super().step(bet)

        # Normalize reward
        if self.big_blind != 0:
            reward = [r/self.big_blind for r in reward]

        # If round is done, finalize hand for all agents
        if all(done):
            for i, agent in enumerate(self.agents):
                obs['hole_cards'] = self.hole_cards[i]

                agent.finalize_hand(self._expand_obs(obs), reward[i])

        return self._expand_obs(obs), reward, done, info            
    
    def reset(self, reset_button = False, reset_stacks = False) -> VerboseObservationDict:
        """
        Resets the environment to start a new hand.
        Also resets all agents.

        Args:
            reset_button (bool): Whether to reset the dealer button position.
            reset_stacks (bool): Whether to reset the players' stacks to initial values.
        
        Returns:
            observation (VerboseObservationDict): The initial observation after resetting the environment.
        """
        for agent in self.agents:
            agent.reset()
        return self._expand_obs(super().reset(reset_button, reset_stacks))
    
    def train(self, steps=1) -> dict:
        """
        Trains all agents that are in training mode for a specified number of steps.

        Args:
            steps (int): Number of training steps to perform for each training agent.

        Returns:
            losses (dict): A dictionary containing the training losses for each training agent.
        """
        losses = {}
        for i, agent in enumerate(self.agents):
            if agent.is_training:
                loss = agent.train(steps)
            if loss is not None:
                losses[f'{agent.name}_{i}'] = loss

        return losses

    def render(self, mode = "human", sleep = 0, **kwargs):
        """
        Calls the render method of the Dealer class with the agents registered in this environment.

        Args:
            mode (str): The rendering mode, e.g., "human" for graphical rendering or "ascii" for text-based rendering.
            sleep (int): Time in seconds to sleep between rendering frames.
            **kwargs: Additional keyword arguments to pass to the render method.
        """
        return super().render(mode, sleep, agents=self.agents, **kwargs)
    
    def _expand_obs(self, obs) -> VerboseObservationDict:
        """
        Expands the observation dictionary to include additional information.

        Args:
            obs (ObservationDict): The original observation dictionary.

        Returns:
            observation (VerboseObservationDict): The expanded observation dictionary.
        """
        observation: VerboseObservationDict = {
            **obs,
            "big_blind": self.big_blind,
            "pot_commits": self.pot_commits,
            "street": self.street,
            "num_streets": self.num_streets,
            "num_players": self.num_players,
            "num_hole_cards": self.num_hole_cards,
            "num_community_cards": sum(self.num_community_cards),
            "has_acted": self.street_option,
            "history": self.history,
            "street_raises": self.street_raises,
        }
        return observation
    
def get_template_observation(config: PokerConfig) -> VerboseObservationDict:
    """
    Generates a template observation dictionary based on the provided poker configuration.

    Args:
        config (PokerConfig): Configuration for the poker game, see poker/clubs/clubs/configs.py for details.

    Returns:
        observation (VerboseObservationDict): A template observation dictionary with default values.
    """
    obs: VerboseObservationDict = {
        "action": 0,
        "active": [False] * config["num_players"],
        "button": 0,
        "big_blind": 0,
        "call": 0,
        "community_cards": [],
        "hole_cards": [],
        "max_raise": 0,
        "min_raise": 0,
        "pot": 0,
        "stacks": [0] * config["num_players"],
        "street_commits": [0] * config["num_players"],
        "pot_commits": [0] * config["num_players"],
        "street": 0,
        "num_streets": config["num_streets"],
        "num_players": config["num_players"],
        "num_hole_cards": config["num_hole_cards"],
        "num_community_cards": sum(config["num_community_cards"]),
        "has_acted": [False] * config["num_players"],
        "history": [],
        "street_raises": 0,
    }
    return obs

