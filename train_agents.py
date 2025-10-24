import os
from time import process_time
import json

from typing import TypedDict

import torch

import clubs
from clubs.configs import PokerConfig

from poker import agents, models
from poker.envs import ClubsEnv

class ModelPerformance(TypedDict):
    train_losses: float
    reward: float

def train_agents(
        config: PokerConfig,
        train_agents_list: list[agents.BaseAgent],
        eval_agents_list: list[list[agents.BaseAgent]],
        num_epochs: int = 50_000,
        warm_up: int = 10000,
        steps_per_epoch: int = 128,
        train_steps_per_epoch: int = 2,
        eval_every: int = 500,
        num_eval_hands: int = 10000,
        save_progress_every: int = 1000,
        folder_to_save: str = "default"
        ) -> tuple[dict, dict]:
    """
    Trains the provided agents in the specified poker environment configuration.

    Args:
        config (PokerConfig): Configuration for the poker environment.
        train_agents_list (list[agents.BaseAgent]): List of agents to be trained.
        eval_agents_list (list[list[agents.BaseAgent]]): List of lists of agents for evaluation.
        num_epochs (int): Number of training epochs, default is 50,000.
        warm_up (int): Number of warm-up hands before training starts, default is 10,000.
        steps_per_epoch (int): Number of environment steps per epoch, default is 128.
        train_steps_per_epoch (int): Number of training steps per epoch, default is 2.
        eval_every (int): Frequency of evaluation in epochs, default is 500.
        num_eval_hands (int): Number of hands to play during evaluation, default is 10,000.
        save_progress_every (int): Frequency of saving progress in epochs, default is 1,000.
        folder_to_save (str): Folder name to save training progress, default is "default".

    Returns:
        tuple[dict, dict]: A tuple containing:
            - dict: Training losses history.
            - dict: Reward history during evaluations.
    """
    assert len(train_agents_list) == config['num_players'], "Number of training agents must match number of players in config."
    for eval_agents in eval_agents_list:
        assert len(eval_agents) == config['num_players'] - 1, "Number of evaluation agents + one training agent must match number of players in config."

    training_losses = {}    
    reward_history = {}
    for i, agent in enumerate(train_agents_list):
        if agent.is_training:
            training_losses[f'{agent.name}_{i}'] = {}
            for eval_agents in eval_agents_list:
                reward_history[f'{agent.name}_{i}'] = {
                    f'{[a.name for a in l]}': {
                        'Epoch':[], 
                        f'Chips/hand over {num_eval_hands} hands':[], 
                        f'Std over {num_eval_hands} hands':[], 
                        f'Wins/{num_eval_hands} hands':[], 
                        f'Ties/{num_eval_hands} hands':[], 
                        'Actions':[]
                    } for l in eval_agents_list}

    train_env = ClubsEnv(config, train_agents_list)
    
    t = process_time()
    for agent in train_agents_list:
        agent.warm_up(True)
    for _ in range(warm_up):            
        obs = train_env.reset(reset_button=False, reset_stacks=True)

        done = [False] * config['num_players']
        while not all(done):
            bet = train_env.act(obs)
            obs, reward, done, info = train_env.step(bet)

    for agent in train_agents_list:
        agent.warm_up(False)
    elapsed_time = process_time() - t
    print(f'Warm up completed: {elapsed_time:.2f} seconds.')

    for epoch in range(num_epochs):
        if (epoch) % eval_every == 0:
            t = process_time()
            for i, agent in enumerate(train_agents_list):
                if not agent.is_training:
                    continue
                agent.set_eval()
                
                print(f'RL_Buffer Size: {len(agent.rl_buffer)} | SL_Buffer Size: {len(agent.sl_buffer)}')

                for eval_agents in eval_agents_list:
                    eval_list = eval_agents.copy()
                    eval_list.insert(0, agent)
                    eval_env = ClubsEnv(config, eval_list)
                    mean_reward = 0
                    mean_squared = 0
                    wins = 0
                    ties = 0
                    action_agg = {'0':0, '1':0, '2':0}
                    for eval_hand in range(num_eval_hands):
                        obs = eval_env.reset(reset_button=False, reset_stacks=True)

                        done = [False] * config['num_players']
                        while not all(done):
                            bet = eval_env.act(obs)
                            obs, reward, done, info = eval_env.step(bet)

                        my_reward = reward[0]
                        if my_reward == 0:
                            ties += 1
                        elif my_reward > 0:
                            wins += 1

                        # Welford's Algorithm for standard deviation
                        delta_1 = my_reward - mean_reward
                        mean_reward += delta_1 / (eval_hand+1)
                        delta_2 = my_reward - mean_reward
                        mean_squared += delta_1 * delta_2

                        for action, count in agent.action_agg.items():
                            action_agg[action] += count

                    std_reward = (mean_squared/(num_eval_hands-1)) ** 0.5
                    reward_history[f'{agent.name}_{i}'][f'{[a.name for a in eval_agents]}']['Epoch'].append(epoch)
                    reward_history[f'{agent.name}_{i}'][f'{[a.name for a in eval_agents]}'][f'Chips/hand over {num_eval_hands} hands'].append(mean_reward)
                    reward_history[f'{agent.name}_{i}'][f'{[a.name for a in eval_agents]}'][f'Std over {num_eval_hands} hands'].append(std_reward)
                    reward_history[f'{agent.name}_{i}'][f'{[a.name for a in eval_agents]}'][f'Wins/{num_eval_hands} hands'].append(wins)
                    reward_history[f'{agent.name}_{i}'][f'{[a.name for a in eval_agents]}'][f'Ties/{num_eval_hands} hands'].append(ties)
                    reward_history[f'{agent.name}_{i}'][f'{[a.name for a in eval_agents]}'][f'Actions'].append(action_agg)
                    print(f'Epoch: {epoch:5.0f}, {agent.name}_{i} against {[a.name for a in eval_agents]}: {mean_reward:.4f} bb/hand')

                agent.set_train()

            elapsed_time = process_time() - t
            print(f'Evaluation completed: {elapsed_time:.2f} seconds.')

        t = process_time()
        step = 0
        while step < steps_per_epoch:
            obs = train_env.reset(reset_button=False, reset_stacks=True)

            done = [False] * config['num_players']
            while not all(done):
                bet = train_env.act(obs)
                obs, reward, done, info = train_env.step(bet)
                step += 1

        agent_loss = train_env.train(train_steps_per_epoch)
        for agent_name, loss in agent_loss.items():
            for key, value in loss.items():
                training_losses[agent_name].setdefault(key, []).append(value)
            # print(f'Epoch: {epoch+1:5.0f}, {agent_name}: {loss}')

        if ((epoch) % save_progress_every == 0) or (epoch+1) == num_epochs:
                os.makedirs(f'training_progress/player/{config['name']}_{config['num_players']}/{folder_to_save}', exist_ok=True)
                with open(f'training_progress/player/{config['name']}_{config['num_players']}/{folder_to_save}/{agent.name}_loss_hist.txt', 'w') as file:
                    json.dump(training_losses, file, indent=4)
                with open(f'training_progress/player/{config['name']}_{config['num_players']}/{folder_to_save}/{agent.name}_reward_hist.txt', 'w') as file:
                    json.dump(reward_history, file, indent=4)

                for i, agent in enumerate(train_agents_list):
                    os.makedirs(f'model_weights/{config['name']}_{config['num_players']}/{folder_to_save}', exist_ok=True)
                    torch.save([agent.model.kwargs, agent.model.state_dict()], f'model_weights/{config['name']}_{config['num_players']}/{folder_to_save}/{agent.name}_{i}.pth')

        elapsed_time = process_time() - t
        print(f'Time for epoch {epoch+1}: {elapsed_time:.2f} seconds.', end='\r')
    
    return training_losses, reward_history

if __name__ == "__main__":
    config = clubs.configs.LEDUC_TWO_PLAYER.copy()

    use_cuda = torch.cuda.is_available()
    device   = torch.device("cuda" if use_cuda else "cpu")

    kwargs, weights = torch.load(f'model_weights/{config["name"]}_{config["num_players"]}/hand_strength_predictor.pth')
    hand_strength_predictor = models.HandStrengthModel(
        config=config,
        **kwargs
    )
    hand_strength_predictor.load_state_dict(weights)

    nsfp_kwargs = {
        'config': config,
        'hand_strength_predictor': hand_strength_predictor,
        'is_training': True,
        'hidden_size': 64,
        'num_hidden_layers': 1,
        'rl_buffer_size': 200_000,
        'sl_buffer_size': 500_000,
        'batch_size': 128,
        'rl_learning_rate': 0.01,
        'sl_learning_rate': 0.005,
        'target_update': 300,
        'epsilon_start': 0.06,
        'eta': 0.2,
        'device': device
    }
    
    nfsp_1 = agents.NFSPAgent(**nsfp_kwargs)
    nfsp_2 = agents.NFSPAgent(**nsfp_kwargs)

    train_agents_list = [nfsp_1, nfsp_2]
    eval_agents_list = [[agents.RandomAgent()],[agents.CallAgent()],[agents.LeducAgent()]]
    
    training_kwargs = {
        'config': config,
        'train_agents_list': train_agents_list,
        'eval_agents_list': eval_agents_list,
        'num_epochs': 50_000,
        'warm_up': 1000,
        'steps_per_epoch':128,
        'train_steps_per_epoch': 2,
        'eval_every': 500,
        'num_eval_hands': 10000,
        'save_progress_every': 500,
        'folder_to_save': "short_state"
    }

    t = process_time()
    training_losses, reward_history = train_agents(**training_kwargs)
    elapsed_time = process_time() - t
    print(f"Training completed in {elapsed_time:.2f} seconds.")