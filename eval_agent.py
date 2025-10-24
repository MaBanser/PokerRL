import torch
import clubs
from poker import agents
from poker import models
from poker.envs import ClubsEnv


def main():
    simple_agent = agents.SimpleAgent()
    random_agent = agents.RandomAgent()
    call_agent = agents.CallAgent()
    leduc_agent = agents.LeducAgent()
    config = clubs.configs.LEDUC_TWO_PLAYER.copy()

    # Load NFSP model
    try:
        kwargs, weights = torch.load(f'model_weights/{config["name"]}_{config["num_players"]}/short_state/NFSP Agent (Training)_0.pth')
    except FileNotFoundError:
        raise FileNotFoundError(f'No model weights found for NFSP Agent at model_weights/{config["name"]}_{config["num_players"]}/short_state/NFSP Agent (Training)_0.pth. Please train the model first by running train_agents.py.')
    nfsp_0 = models.NFSP(
        config=config,
        **kwargs
    )
    nfsp_0.load_state_dict(weights)

    nfsp_agent_0 = agents.NFSPAgent(
        config=config,
        model = nfsp_0,
    )

    try:
        kwargs, weights = torch.load(f'model_weights/{config["name"]}_{config["num_players"]}/short_state/NFSP Agent (Training)_1.pth')
    except FileNotFoundError:
        raise FileNotFoundError(f'No model weights found for NFSP Agent at model_weights/{config["name"]}_{config["num_players"]}/short_state/NFSP Agent (Training)_1.pth. Please train the model first by running train_agents.py.')
    nfsp_1 = models.NFSP(
        config=config,
        **kwargs
    )
    nfsp_1.load_state_dict(weights)

    nfsp_agent_1 = agents.NFSPAgent(
        config=config,
        model = nfsp_1,
    )

    # Load hand strength predictor
    try:
        kwargs, weights = torch.load(f'model_weights/{config["name"]}_{config["num_players"]}/hand_strength_predictor.pth')
    except FileNotFoundError:
        raise FileNotFoundError(f'No model weights found for Hand Strength Predictor at model_weights/{config["name"]}_{config["num_players"]}/hand_strength_predictor.pth. Please train the model first by running train_handstrength.py.')
    hand_strength_predictor = models.HandStrengthModel(
        config=config,
        **kwargs
    )
    hand_strength_predictor.load_state_dict(weights)

    # Load NFSP + Equity model
    try:
        kwargs, weights = torch.load(f'model_weights/{config["name"]}_{config["num_players"]}/short_state/NFSP Agent + Equity (Training)_0.pth')
    except FileNotFoundError:
        raise FileNotFoundError(f'No model weights found for NFSP + Equity Agent at model_weights/{config["name"]}_{config["num_players"]}/short_state/NFSP Agent + Equity (Training)_0.pth. Please train the model first by running train_agents.py.')
    nfsp_equity_0 = models.NFSP(
        config=config,
        **kwargs
    )
    nfsp_equity_0.load_state_dict(weights)

    nfsp_equity_agent_0 = agents.NFSPAgent(
        config=config,
        model = nfsp_equity_0,
        hand_strength_predictor=hand_strength_predictor
    )

    try:
        kwargs, weights = torch.load(f'model_weights/{config["name"]}_{config["num_players"]}/short_state/NFSP Agent + Equity (Training)_1.pth')
    except FileNotFoundError:
        raise FileNotFoundError(f'No model weights found for NFSP + Equity Agent at model_weights/{config["name"]}_{config["num_players"]}/short_state/NFSP Agent + Equity (Training)_1.pth. Please train the model first by running train_agents.py.')
    nfsp_equity_1 = models.NFSP(
        config=config,
        **kwargs
    )
    nfsp_equity_1.load_state_dict(weights)

    nfsp_equity_agent_1 = agents.NFSPAgent(
        config=config,
        model = nfsp_equity_1,
        hand_strength_predictor=hand_strength_predictor
    )
   
    agent_list = [random_agent, random_agent]

    config['start_stack']=50
    env = ClubsEnv(config, agent_list)

    max_rounds = 25
    hands = 5000

    wins = {f'{p.name}_{i}_chips/hand':0 for i, p in enumerate(agent_list)}
    
    #env.render(port=4789)
    #env.render(mode='ascii')
    for _ in range(hands):
        obs = env.reset(reset_stacks=False)
        #env.render(sleep=1)
        round = 0
        done = [False]
        while not all(done):
            bet = env.act(obs, verbose=False)
            obs, reward, done, info = env.step(bet)
            #env.render(sleep=1)
            if all(done):
                for i, chips in enumerate(reward):
                    wins[f'{agent_list[i].name}_{i}_chips/hand'] += chips
            if info["tournament_ended"]:
                print(f'Tournament winner: {agent_list[info["winner"]].name}_{info["winner"]}')
                env.reset(reset_stacks=True)
    
    for k,v in wins.items():
        wins[k]=v/hands

    print('done')
    print(f'{wins}')


if __name__ == "__main__":
    main()