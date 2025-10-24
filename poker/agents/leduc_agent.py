import random
from . import BaseAgent
from poker.envs.clubs_env import VerboseObservationDict

class LeducAgent(BaseAgent):
    """A rule based agent for Leduc Poker."""
    def __init__(self) -> None:
        super().__init__()
        self.name = 'Leduc Agent'
        self.short_name = 'Leduc'

    def act(self, obs: VerboseObservationDict) -> int:
        # Get own cards rank (suit does not matter)
        hole_card = obs['hole_cards'][0].rank
        call_value = obs['call']
        raise_value = obs['min_raise']
        
        rand = random.random()

        # Pre-Flop
        if obs['street'] == 0:
            match hole_card:
                case 'Q':
                    if rand < 0.4:
                        return call_value
                    return 0
                
                case 'K':
                    if rand < 0.1:
                        return max(raise_value, call_value)
                    if rand < 0.7:
                        return call_value
                    return 0
                
                case 'A':
                    if rand < 0.2:
                        return max(raise_value, call_value)
                    if rand < 0.9:
                        return call_value
                    return 0
                
        # Post-Flop 
        else:
            community_card = obs['community_cards'][0].rank
            if hole_card == community_card:
                return max(raise_value, call_value)
            else:
                match hole_card:
                    case 'Q':
                        return 0
                    
                    case 'K':
                        if rand < 0.6:
                            return call_value
                        return 0
                    
                    case 'A':
                        if rand < 0.9:
                            return call_value
                        return 0
                    
        raise Exception('Unexpected card rank. Make sure to use Leduc environment.')
        return 0