import random

import clubs


def main():
    # 1-2 no limit 6 player texas hold'em
    config = clubs.configs.NO_LIMIT_HOLDEM_NINE_PLAYER
    dealer = clubs.poker.Dealer(**config)
    obs = dealer.reset()

    dealer.render(sleep=1)

    while True:
        # number of chips a player must bet to call
        call = obs["call"]
        # smallest number of chips a player is allowed to bet for a raise
        min_raise = obs["min_raise"]
        max_raise = obs["max_raise"]

        rand = random.random()
        # 15% chance to fold
        if rand < 0.15:
            bet = 0
        # 50% chance to call
        elif rand < 0.65:
            bet = call
        # 35% to raise
        else:
            bet = random.randint(min_raise, max_raise)

        obs, rewards, done, info = dealer.step(bet)

        dealer.render(sleep=1)

        if all(done):
            if info["tournament_ended"]:
                print(f"Tournament ended. Winner: {info['winner']}")
                break
            obs = dealer.reset()

    print(rewards)


if __name__ == "__main__":
    main()
