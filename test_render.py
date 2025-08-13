import clubs

def main():
    dealer = clubs.Dealer(**clubs.configs.LIMIT_HOLDEM_SIX_PLAYER)

    obs = dealer.reset()
    dealer.render(port=4789)
    print(obs)
    while True:
        obs, reward, done = dealer.step(obs['call'])
        dealer.render(sleep=2)
        if all(done):
            break

    print(reward)
    print('done')


if __name__ == "__main__":
    main()