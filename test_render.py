import clubs
from poker import agents
from poker.envs import ClubsEnv


def main():
    simple_agent = agents.SimpleAgent()
    random_agent = agents.RandomAgent()
    call_agent = agents.CallAgent()
    config = clubs.configs.NO_LIMIT_HOLDEM_SIX_PLAYER
    config['start_stack'] = 10
    num_players = config['num_players']
    agent_list = [simple_agent, random_agent] * (num_players // 2)
    #agent_list = [call_agent] * num_players
    env = ClubsEnv(config, agent_list)

    max_rounds = 1

    obs = env.reset()
    round = 0
    env.render(port=4789,sleep=2)
    # env.render(mode='ascii')
    print(obs)
    while True and round <= max_rounds:        
        env.render()
        bet = env.act(obs)
        print(f'Agent bets: {bet}')
        obs, reward, done, info = env.step(bet)
        if all(done):
            round += 1
            print(obs)
            print(reward)
            # env.render()
            if info['tournament_ended']:
                print(f"Tournament ended. Winner: {info['winner']}")
                break
            obs = env.reset()
            print(obs)

    print(reward)
    print('done')


if __name__ == "__main__":
    main()