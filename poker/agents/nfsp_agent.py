import math
from clubs.configs import PokerConfig
from . import BaseAgent
from poker import models, envs
from poker.models import rl_buffer, sl_buffer
from poker.utils import cards_to_ints

import torch

class NFSPAgent(BaseAgent):
    """An agent that uses neural ficticious self play to decide actions."""
    def __init__(
            self,
            config: PokerConfig,
            model: models.NFSP = None,
            hand_strength_predictor: models.HandStrengthModel = None,
            is_training: bool = False,
            hidden_size: int = 64,
            num_hidden_layers: int = 1,
            type_state: str = 'short',
            rl_buffer_size: int = 200_000,
            sl_buffer_size: int = 2_000_000,
            batch_size: int = 128,
            rl_learning_rate: float = 0.1,
            sl_learning_rate: float = 0.005,
            target_update: int = 300,
            epsilon_start: float = 0.06,
            eta: float = 0.1,
            device: torch.device = torch.device('cpu')
        ) -> None:
        """
        Initializes the NFSP agent.

        Args:
            config (PokerConfig): Configuration for the poker game.
            model (models.NFSP, optional): Pretrained NFSP model. If None, a new model is created. Defaults to None.
            hand_strength_predictor (models.HandStrengthModel, optional): Pretrained hand strength predictor model. Defaults to None.
            is_training (bool, optional): Whether the agent is in training mode. Defaults to False.
            hidden_size (int, optional): Size of the hidden layers in the NFSP model. Defaults to 64.
            num_hidden_layers (int, optional): Number of hidden layers in the NFSP model. Defaults to 1.
            type_state (str, optional): Type of state representation ('short' or 'long'). Defaults to 'short'.
            rl_buffer_size (int, optional): Capacity of the RL buffer. Defaults to 200,000.
            sl_buffer_size (int, optional): Capacity of the SL buffer. Defaults to 2,000,000.
            batch_size (int, optional): Batch size for training. Defaults to 128.
            rl_learning_rate (float, optional): Learning rate for RL optimizer. Defaults to 0.1.
            sl_learning_rate (float, optional): Learning rate for SL optimizer. Defaults to 0.005.
            target_update (int, optional): Number of steps between target network updates. Defaults to 300.
            epsilon_start (float, optional): Initial epsilon value for epsilon-greedy policy. Defaults to 0.06.
            eta (float, optional): Probability of using best response policy during training. Defaults to 0.1.
            device (torch.device, optional): Device to run the model on. Defaults to CPU.
        """
        super().__init__()
        self.config = config
        self.name = 'NFSP Agent'
        self.short_name = 'NFSP'
        self.is_training = is_training
        self.has_hand_strength_predictor=False
        self.hand_strength_predictor = hand_strength_predictor
        if self.hand_strength_predictor:
            self.name += ' + Equity'
            self.short_name += ' + EQ'
            self.has_hand_strength_predictor=True
            self.hand_strength_predictor.eval()
        self.type_state=type_state
        self.action_agg = {'0':0, '1':0, '2':0}
        self.eta_orig = eta
        self.epsilon_orig = epsilon_start
        self.active_hand_players = None
        self.template_obs = envs.clubs_env.get_template_observation(self.config)
        self.state_size = len(self.get_feature_state(self.template_obs))
        if model:
            self.model = model
        else:
            self.model = models.NFSP(config=self.config, state_size=self.state_size, hidden_size=hidden_size, num_hidden_layer=num_hidden_layers)
        assert self.model.state_size == self.state_size, f"Model state size does not match expected state size. Make sure the model was trained with the same config."
        self.model.to(device)
        self.device = device
        if not self.is_training:
            self.model.average_policy_model.eval()
        else:
            self.name += ' (Training)'
            self.short_name += ' (T)'
            self.model.best_response_model.train()
            self.model.target_best_response_model.train()
            self.model.average_policy_model.train()
            self.rl_optimizer = torch.optim.SGD(self.model.best_response_model.parameters(), lr=rl_learning_rate)
            self.sl_optimizer = torch.optim.SGD(self.model.average_policy_model.parameters(), lr=sl_learning_rate)
            self.target_update = target_update
            self.rl_criterion = torch.nn.HuberLoss()
            self.sl_criterion = torch.nn.CrossEntropyLoss()
            self.rl_buffer = rl_buffer(rl_buffer_size)
            self.sl_buffer = sl_buffer(sl_buffer_size)
            self.batch_size = batch_size
            self.epsilon_start = epsilon_start
            self.epsilon = epsilon_start
            self.eta = eta
            self.last_state = None
            self.last_action = None
            self.train_steps = 0
            self.t = 0


    def act(self, obs: envs.clubs_env.VerboseObservationDict) -> int:
        """
        Selects an action based on the current observation.

        Args:
            obs (VerboseObservationDict): The current observation from the environment.

        Returns:
            bet: int: The action to take (bet amount).
        """
        mask = torch.tensor(self.get_action_mask(obs), device=self.device).float().unsqueeze(0)
        state = torch.tensor(self.get_feature_state(obs), device=self.device).float().unsqueeze(0)
        if not self.is_training:
            action = self.model.act_average_policy(state, mask)
            self.action_agg[f'{action.item()}'] += 1
        else:
            if torch.greater(torch.rand([]), self.eta):
                action = self.model.act_average_policy(state, mask)
            else:
                action = self.model.act_best_response(state, mask, self.epsilon)
                self.epsilon = self.epsilon_start/math.sqrt(1+self.t/100)
                self.t += 1
                self.sl_buffer.store(state.squeeze(0), action.long(), mask.squeeze(0))
            if self.last_state is not None:
                self.rl_buffer.store(self.last_state, self.last_action, 0, state.squeeze(0), 0, mask.squeeze(0))
            self.last_state = state.squeeze(0)
            self.last_action = action

        action = action.item()
        if action == 2:
            return obs['min_raise']
        if action == 1:
            return obs['call']
        
        return 0
    
    def finalize_hand(self, obs: envs.clubs_env.VerboseObservationDict, reward: float) -> None:
        """
        Finalizes the hand for the agent, storing the final experience in the RL buffer.

        Args:
            obs (VerboseObservationDict): The final observation from the environment.
            reward (float): The reward received at the end of the hand.
        """
        if not self.is_training:
            return
        mask = torch.tensor(self.get_action_mask(obs), device=self.device).float()
        state = torch.tensor(self.get_feature_state(self.template_obs), device=self.device).float()
        self.rl_buffer.store(self.last_state, self.last_action, reward, state, 1, mask)
        
    def reset(self) -> None:
        """
        Resets the agent's internal state at the beginning of a new hand.
        """
        if self.is_training:
            self.last_state = None
            self.last_action = None
        self.active_hand_players = None
        self.action_agg = {'0':0, '1':0, '2':0}

    def train(self, steps=1) -> dict:
        """ 
        Trains the agent's models for a specified number of steps.

        Args:
            steps (int, optional): Number of training steps to perform. Defaults to 1.

        Returns:
            dict: A dictionary containing the RL and SL losses.
        """
        rl_loss = self.train_RL(steps)
        sl_loss = self.train_SL(steps)
        
        loss = {'RL_Loss': rl_loss, 'SL_Loss': sl_loss}
        return loss
    
    def train_RL(self, steps=1) -> float:
        """
        Trains the best response model using reinforcement learning for a specified number of steps.
        
        Args:
            steps (int, optional): Number of training steps to perform. Defaults to 1.
            
        Returns:
            float: The average RL loss over the training steps.
        """
        running_average_loss = 0
        for _ in range(steps):
            state, action, reward, next_state, done, next_mask = next(self.rl_buffer.sample(self.batch_size))
            state = state.to(self.device, non_blocking=True)
            action = action.to(self.device, non_blocking=True)
            reward = reward.to(self.device, non_blocking=True)
            next_state = next_state.to(self.device, non_blocking=True)
            done = done.to(self.device, non_blocking=True)
            next_mask = next_mask.to(self.device, non_blocking=True)
            self.train_steps += 1

            # Calculate target Q
            with torch.no_grad():
                target = reward + self.model.target_best_response_model.max_q(next_state, next_mask)*(1-done)

            pred = self.model.best_response_model.q_val(state, action)
            loss = self.rl_criterion(pred, target)

            self.rl_optimizer.zero_grad()
            loss.backward()
            self.rl_optimizer.step()
    
            running_average_loss += loss.item()

            if self.train_steps >= self.target_update:
                self.model.update_target_network()
                self.train_steps = 0

        return running_average_loss/steps

    def train_SL(self, steps=1) -> float:
        """
        Trains the average policy model using supervised learning for a specified number of steps.

        Args:
            steps (int, optional): Number of training steps to perform. Defaults to 1.

        Returns:
            float: The average SL loss over the training steps.
        """
        running_average_loss = 0
        for _ in range(steps):
            state, action, mask = next(self.sl_buffer.sample(self.batch_size))
            state = state.to(self.device, non_blocking=True)
            action = action.to(self.device, non_blocking=True)
            mask = mask.to(self.device, non_blocking=True)

            pred = self.model.average_policy_model(state, mask)
            loss = self.sl_criterion(pred, action.squeeze())

            self.sl_optimizer.zero_grad()
            loss.backward()
            self.sl_optimizer.step()
    
            running_average_loss += loss.item()

        return running_average_loss/steps

    def set_eval(self) -> None:
        """
        Sets the agent to evaluation mode.
        """
        self.model.average_policy_model.eval()
        return super().set_eval()

    def set_train(self) -> None:
        """
        Sets the agent to training mode.
        """
        self.model.average_policy_model.train()
        return super().set_train()
    
    def warm_up(self, is_warm_up = False) -> None:
        """
        Sets the agent to warm-up mode, adjusting eta and epsilon accordingly.

        Args:
            is_warm_up (bool, optional): Whether to enter warm-up mode. Defaults to False.
        """
        if is_warm_up:
            self.eta_orig = self.eta
            self.epsilon_orig = self.epsilon_start
            self.eta = 1
            self.epsilon_start = 0.7
        else:
            self.eta = self.eta_orig
            self.t = 0
            self.epsilon_start = self.epsilon_orig
            self.epsilon = self.epsilon_start
        return super().warm_up(is_warm_up)

    def get_action_mask(self, obs: envs.clubs_env.VerboseObservationDict) -> list[int]:
        """
        Generates an action mask based on the current observation.
        
        Args:
            obs (VerboseObservationDict): The current observation from the environment.
            
        Returns:
            list[int]: A list representing the action mask.
        """
        if obs['call'] == 0:
            return [0, 1, 1]    # Don't allow fold if call is 0
        elif obs['min_raise'] == 0:
            return [1, 1, 0]    # Don't allow raise if player can't raise
        return [1, 1, 1]
    
    def get_feature_state(self, obs: envs.clubs_env.VerboseObservationDict) -> list[float]:
        """
        Generates a feature state representation based on the current observation.

        Args:
            obs (VerboseObservationDict): The current observation from the environment.

        Returns:
            list[float]: A list representing the feature state.
        """
        state = []
        idx = obs['action']
        bb = obs['big_blind']
        if bb == 0:
            bb = 1 # For later normalization
        num_players = obs['num_players']

        # Card encoding
        hole_cards_idx = [0]*obs['num_hole_cards']
        hole_cards_idx[:len(obs['hole_cards'])] = cards_to_ints(obs['hole_cards'], self.config['num_ranks'])
        community_cards_idx = [0]*obs['num_community_cards']
        community_cards_idx[:len(obs['community_cards'])] = cards_to_ints(obs['community_cards'], self.config['num_ranks'])
        
        if self.has_hand_strength_predictor:
            # card_embeddings = self.hand_strength_predictor.card_embedding.weight.cpu().detach().numpy()

            # # Embedding of hole cards and community cards
            # hole_cards = card_embeddings[hole_cards_idx].flatten().tolist()
            # community_cards = card_embeddings[community_cards_idx].flatten().tolist()

            # One-hot encoding of hole cards and community cards
            hole_cards = [0] * (self.config['num_ranks'] * self.config['num_suits'])
            for card in hole_cards_idx:
                if card != 0:
                    hole_cards[card-1] = 1
            community_cards = [0] * (self.config['num_ranks'] * self.config['num_suits'])
            for card in community_cards_idx:
                if card != 0:
                    community_cards[card-1] = 1
            # Get hand strength as predicted by hand strength predictor
            hand_strength = self.hand_strength_predictor.win_prob(torch.tensor(hole_cards_idx + community_cards_idx).unsqueeze(0)).item()
            # Center hand_strength and scale to [-1,1]
            hand_strength = (hand_strength-0.5)*2

        else:
            # One-hot encoding of hole cards and community cards
            hole_cards = [0] * (self.config['num_ranks'] * self.config['num_suits'])
            for card in hole_cards_idx:
                if card != 0:
                    hole_cards[card-1] = 1
            community_cards = [0] * (self.config['num_ranks'] * self.config['num_suits'])
            for card in community_cards_idx:
                if card != 0:
                    community_cards[card-1] = 1

        # One-hot encoding of player position relative to button
        button = [0] * num_players
        button[obs['button']] = 1
        button = button[idx:]+button[:idx]

        # One-hot encoding of active players relative to current player
        active = [1 if a else 0 for a in obs['active'][idx:]+obs['active'][:idx]]

        # One-hot encoding if player has acted this street, relative to current player
        has_acted = [1 if acted or not active else 0 for acted, active in zip(obs['has_acted'][idx:]+obs['has_acted'][:idx],obs['active'][idx:]+obs['active'][:idx])]

        # One-hot encoding if player has raised or checked per street, relative to current player
        has_raised = [[0] * num_players for _ in range(self.config['num_streets'])]
        has_checked = [[0] * num_players for _ in range(self.config['num_streets'])]
        # First round big blind already bet
        last_bet = bb
        acted_since_raise = set()
        if self.active_hand_players == None:
            self.active_hand_players = {i for i, a in enumerate(obs['active']) if a}
        active_players = set(self.active_hand_players)
        s = 0
        for player, bet_amount, folded in obs['history']:
            if not folded:    # Not folded
                if bet_amount > last_bet:   # Has raised
                    last_bet = bet_amount
                    acted_since_raise = {player}
                    has_raised[s][player] = 1
                    continue
                elif sum(has_raised[s]) == 0: # Now one raised this street
                    has_checked[s][player] = 1
                
                if sum(has_checked[s][p] for p in active_players) == len(active_players):  # Everyone checked -> Next street
                    s += 1
                    last_bet = 0
                    acted_since_raise = set()
                    continue
                
                acted_since_raise.add(player)
                if len(acted_since_raise) == len(active_players):  # Last one to call
                    s += 1
                    last_bet = 0
                    acted_since_raise = set()
                    
            else:   # Folded
                active_players.discard(player)

        has_raised = [e for s in has_raised for e in s[idx:]+s[:idx]]
        has_checked = [e for s in has_checked for e in s[idx:]+s[:idx]]
        

        # One-hot encoding of current street
        street = [0] * obs['num_streets']
        street[obs['street']] = 1

        # Stacks and commits in big blinds relative to current player (so that the current player is always first)
        stacks = [chips/bb for chips in obs['stacks'][idx:]+obs['stacks'][:idx]]
        pot_commits = [chips/bb for chips in obs['pot_commits'][idx:]+obs['pot_commits'][:idx]]
        street_commits = [chips/bb for chips in obs['street_commits'][idx:]+obs['street_commits'][:idx]]

        # Normalized chip amounts
        pot = obs['pot']/bb
        call = obs['call']/bb
        min_raise = obs['min_raise']/bb
        max_raise = obs['max_raise']/bb
        stack = stacks[0]
        pot_commit = pot_commits[0]
        street_commit = street_commits[0]

        if self.has_hand_strength_predictor:
            if self.type_state == 'short':
                state = hole_cards + community_cards + street + button + active + has_acted + has_raised + has_checked + [pot, call, min_raise] + [hand_strength]
            else:
                state = hole_cards + community_cards + street + button + active + has_acted + has_raised + has_checked + [pot, call, min_raise, max_raise] + stacks + street_commits + pot_commits + [hand_strength]
        else:
            if self.type_state == 'short':
                state = hole_cards + community_cards + street + button + active + has_acted + has_raised + has_checked + [pot, call, min_raise]
            else:
                state = hole_cards + community_cards + street + button + active + has_acted + has_raised + has_checked + [pot, call, min_raise, max_raise] + stacks + street_commits + pot_commits
        return state