import clubs
from . import BaseAgent

class CallAgent(BaseAgent):
    """An agent that always calls the current bet."""
    def __init__(self) -> None:
        super().__init__()
        self.name = "Call Agent"
        self.short_name = "Call"
        
    def act(self, obs: clubs.poker.engine.ObservationDict) -> int:
        return obs['call']