from clubs.configs import PokerConfig

import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, state_size: int, hidden_size: int = 64, num_hidden_layer: int = 1):
        """
        Deep Q-Network (DQN) model for approximating the Q-value function.
        
        Args:
            state_size (int): The size of the input state vector.
            hidden_size (int): The size of the hidden layers, default is 64.
            num_hidden_layer (int): The number of hidden layers, default is 1.
        """
        super(DQN, self).__init__()
        self.num_actions = 3
        self.fc1 = nn.Linear(state_size, hidden_size)

        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(num_hidden_layer)]
        )

        self.out = nn.Linear(hidden_size, self.num_actions)


    def forward(self, x) -> torch.Tensor:
        x = F.leaky_relu(self.fc1(x))
        for layer in self.hidden_layers:
            x = F.leaky_relu(layer(x))
        return self.out(x)
    
    def act(self, input_state, mask, epsilon = 0) -> torch.Tensor:
        """
        Selects an action based on the input state and mask using an epsilon-greedy policy.

        Args:
            input_state (torch.Tensor): The input state tensor.
            mask (torch.Tensor): The mask tensor indicating valid actions.
            epsilon (float): The probability of choosing a random action, default is 0.

        Returns:
            torch.Tensor: The selected action tensor.
        """
        with torch.no_grad():
            q_vals = self(input_state)
            q_vals.masked_fill_(mask==0, -float(1e9))
            p = torch.rand([])
            if torch.greater(p,epsilon):
                action = torch.argmax(q_vals,-1)
            else:
                action = torch.multinomial(mask.float(),1).squeeze(-1)
        return action
    
    def max_q(self, input_state, mask) -> torch.Tensor:
        """
        Computes the maximum Q-value for the given input state and mask.

        Args:
            input_state (torch.Tensor): The input state tensor.
            mask (torch.Tensor): The mask tensor indicating valid actions.

        Returns:
            torch.Tensor: The maximum Q-value tensor.
        """

        q_vals = self(input_state)
        q_vals.masked_fill_(mask==0, -float(1e9))
        return torch.amax(q_vals,-1).unsqueeze(-1)
    
    def q_val(self, input_state, actions) -> torch.Tensor:
        """
        Retrieves the Q-values for the specified actions given the input state.

        Args:
            input_state (torch.Tensor): The input state tensor.
            actions (torch.Tensor): The actions tensor.

        Returns:
            torch.Tensor: The Q-values tensor for the specified actions.
        """
        q_vals = self(input_state)
        return torch.gather(q_vals,1,actions)
    
    
class PI(nn.Module):
    def __init__(self, state_size: int, hidden_size: int = 64, num_hidden_layer: int = 1):
        """
        Policy Iteration (PI) model for approximating the policy function.

        Args:
            state_size (int): The size of the input state vector.
            hidden_size (int): The size of the hidden layers, default is 64.
            num_hidden_layer (int): The number of hidden layers, default is 1.
        """
        super(PI, self).__init__()
        self.num_actions = 3
        self.fc1 = nn.Linear(state_size, hidden_size)

        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(num_hidden_layer)]
        )

        self.out = nn.Linear(hidden_size, self.num_actions)


    def forward(self, x, mask) -> torch.Tensor:
        x = F.leaky_relu(self.fc1(x))
        for layer in self.hidden_layers:
            x = F.leaky_relu(layer(x))
        x = self.out(x)
        x.masked_fill_(mask==0, -float(1e9))
        return x

    def act(self, input_state, mask) -> torch.Tensor:
        """
        Selects an action based on the input state and mask using the policy model.

        Args:
            input_state (torch.Tensor): The input state tensor.
            mask (torch.Tensor): The mask tensor indicating valid actions.

        Returns:
            torch.Tensor: The selected action tensor.
        """
        with torch.no_grad():
            action_probs = F.softmax(self(input_state, mask), -1)
            action = torch.multinomial(action_probs, 1).squeeze(-1)
        return action
    
class NFSP(nn.Module):
    def __init__(self, config: PokerConfig, state_size: int, hidden_size: int = 64, num_hidden_layer: int = 1):
        """
        Neural Fictitious Self-Play (NFSP) model combining DQN and PI models.

        Args:
            config (PokerConfig): Configuration for the poker game.
            state_size (int): The size of the input state vector.
            hidden_size (int): The size of the hidden layers, default is 64.
            num_hidden_layer (int): The number of hidden layers, default is 1.
        """
        super(NFSP, self).__init__()
        self.kwargs = {
            'state_size': state_size,
            'hidden_size': hidden_size,
            'num_hidden_layer': num_hidden_layer
        }
        self.name = config["name"]
        self.num_players = config["num_players"]

        self.state_size = state_size
        self.best_response_model = DQN(state_size, hidden_size, num_hidden_layer)
        self.target_best_response_model = DQN(state_size, hidden_size, num_hidden_layer)
        self.update_target_network()

        self.average_policy_model = PI(state_size, hidden_size, num_hidden_layer)
    
    def act_best_response(self, input_state, mask, epsilon=0) -> torch.Tensor:
        """
        Selects an action based on the input state and mask using the best response model with epsilon-greedy policy.

        Args:
            input_state (torch.Tensor): The input state tensor.
            mask (torch.Tensor): The mask tensor indicating valid actions.
            epsilon (float): The probability of choosing a random action, default is 0.

        Returns:
            torch.Tensor: The selected action tensor.
        """
        return self.best_response_model.act(input_state, mask, epsilon)
    
    def act_average_policy(self, input_state, mask) -> torch.Tensor:
        """
        Selects an action based on the input state and mask using the average policy model.

        Args:
            input_state (torch.Tensor): The input state tensor.
            mask (torch.Tensor): The mask tensor indicating valid actions.

        Returns:
            torch.Tensor: The selected action tensor.
        """
        return self.average_policy_model.act(input_state, mask)
    
    def update_target_network(self):
        """
        Updates the target best response model with the weights of the best response model.
        """
        self.target_best_response_model.load_state_dict(self.best_response_model.state_dict())


if __name__ == "__main__":
    import clubs
    import os

    # Environment and agent setup
    config = clubs.configs.LEDUC_TWO_PLAYER.copy()


    model = NFSP(config, state_size=10)

    print(f'DQN layer 1 weights: {model.best_response_model.fc1.weight.cpu().detach().numpy()}')

    os.makedirs(f'model_weights/{model.name}_{model.num_players}', exist_ok=True)
    torch.save([model.kwargs, model.state_dict()], f'model_weights/{model.name}_{model.num_players}/NFSP.pth')


    kwargs, weights = torch.load(f'model_weights/{config["name"]}_{config["num_players"]}/NFSP.pth')
    loaded_model = NFSP(config=config,**kwargs)
    loaded_model.load_state_dict(weights)

    print(f'Loaded DQN layer 1 weights: {loaded_model.best_response_model.fc1.weight.cpu().detach().numpy()}')