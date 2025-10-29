import os
import numpy as np
import math
from typing import List
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from seaborn import heatmap, scatterplot
from sklearn.calibration import CalibrationDisplay
from sklearn.manifold import TSNE

import torch
from torch.utils.data import DataLoader

import clubs
from clubs import Card
from clubs.poker.card import CHAR_RANK_TO_INT_RANK
from poker.models import HandStrengthModel
from poker.utils import ints_to_cards, cards_to_ints, build_dataset, CHAR_RANK_TO_INT_RANK_REVERSED, UNPRETTY_SUITS


def visualize_preflop_hand_strength(model: HandStrengthModel, fig:Figure = None) -> Figure:
    """
    Visualizes the preflop hand strengths learned by the HandStrengthModel.

    Args:
        model (HandStrengthModel): The trained model.

    Returns:
        matplotlib.figure.Figure: The generated heatmap figure.
    """
    model.eval()
    # Create a grid of card indices
    num_ranks = model.num_ranks
    num_suits = model.num_suits
    num_hole_cards = model.num_hole_cards
    if num_hole_cards == 2:   # Visualization for 2 hole cards (e.g., Texas Hold'em)
        heatmap_data = np.zeros((num_ranks, num_ranks))
        count_matrix = np.zeros((num_ranks, num_ranks))
        annotations = np.empty((num_ranks, num_ranks), dtype=object)
    elif num_hole_cards == 1: # Visualization for 1 hole card (e.g., Leduc or Kuhn Poker)
        heatmap_data = np.zeros((num_ranks, 1))
        count_matrix = np.zeros((num_ranks, 1))
        annotations = np.empty((num_ranks, 1), dtype=object)
    else:
        raise NotImplementedError("Visualization only implemented for 1 or 2 hole cards.")
    
    # Generate all possible hole card combinations
    hole_card_combinations = torch.combinations(torch.arange(num_ranks * num_suits) + 1, r=num_hole_cards)
    
    pretty_cards = [ints_to_cards(card_int, num_ranks) for card_int in hole_card_combinations.tolist()]
    if num_hole_cards > 1:
        sorted_hands = []
        for hand in pretty_cards:
            hand.sort(key=lambda card: CHAR_RANK_TO_INT_RANK[card.rank])
            sorted_hands.append(cards_to_ints(hand, num_ranks))

        hole_card_combinations = torch.tensor(sorted_hands, dtype=torch.int64)
    # Add padding to match the expected input size of the model
    community_card_padding = torch.zeros((len(hole_card_combinations), model.num_community_cards), dtype=torch.int64)
    input_tensor = torch.cat((hole_card_combinations, community_card_padding), dim=1)
    input_tensor = input_tensor.to(next(model.parameters()).device)
    # Predict hand strengths
    with torch.no_grad():
        predicted_strengths = model.win_prob(input_tensor).cpu().detach().numpy()

    for hand, win_prob in zip(pretty_cards, predicted_strengths):
        if num_hole_cards == 2:
            card_1, card_2 = hand
            card_1_idx = CHAR_RANK_TO_INT_RANK_REVERSED[card_1.rank]
            card_2_idx = CHAR_RANK_TO_INT_RANK_REVERSED[card_2.rank]
        elif num_hole_cards == 1:
            card_1 = hand[0]
            card_2 = card_1
            card_1_idx = CHAR_RANK_TO_INT_RANK_REVERSED[card_1.rank]
            card_2_idx = 0

        if card_1.rank == card_2.rank:  # Pocket pair
            pos = (card_1_idx, card_2_idx)
            if num_hole_cards == 2:
                hand_notation = f"{card_1.rank}{card_2.rank}"  # e.g., "AA"
            elif num_hole_cards == 1:
                hand_notation = f"{card_1.rank}"  # e.g., "A"
        elif card_1.suit == card_2.suit:  # Suited
            pos = (card_2_idx, card_1_idx)  # Upper triangle
            hand_notation = f"{card_2.rank}{card_1.rank}s"  # e.g., "AKs"
        else:  # Unsuited
            pos = (card_1_idx, card_2_idx)  # Lower triangle (swap indexes)
            hand_notation = f"{card_2.rank}{card_1.rank}o"  # e.g., "AKo"

        # Accumulate win probabilities for averaging later
        heatmap_data[pos] += win_prob.item()
        count_matrix[pos] += 1
        annotations[pos] = hand_notation

    # Average the win probabilities for each hand
    heatmap_data /= count_matrix
    annotations = np.char.add(annotations, np.char.mod('\n%d%%', np.round(heatmap_data*100)))

    if fig is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot()
    else:
        ax = fig.axes[0]
        ax.clear()
    my_fontsize = 50/np.sqrt(len(heatmap_data))
    ax.set_title(f'Preflop Hand Strengths {model.name} {model.num_players} Player', fontsize=22, pad=20)

    if num_hole_cards == 2:
        ax = heatmap(
            heatmap_data, 
            annot=annotations,
            vmin=0, 
            fmt='', 
            cmap="RdGy_r",
            cbar=False,
            square=True,
            annot_kws={"fontsize": my_fontsize, "fontweight": "bold"},
            ax=ax
            )
        ax.set_xticks(np.arange(num_ranks) + 0.5, list(CHAR_RANK_TO_INT_RANK_REVERSED.keys())[:num_ranks], fontsize=my_fontsize)
        ax.set_yticks(np.arange(num_ranks) + 0.5, list(CHAR_RANK_TO_INT_RANK_REVERSED.keys())[:num_ranks], fontsize=my_fontsize, rotation=0)
        ax.xaxis.tick_top()
    elif num_hole_cards == 1:
        ax = heatmap(
            heatmap_data,
            annot=annotations,
            vmin=0, vmax=1,
            fmt='',
            cmap="RdGy_r",
            cbar=False,
            square=True,
            annot_kws={"fontsize": my_fontsize, "fontweight": "bold"},
            ax=ax
            )
        ax.set_yticks(np.arange(num_ranks) + 0.5, list(CHAR_RANK_TO_INT_RANK_REVERSED.keys())[:num_ranks], fontsize=my_fontsize, rotation=0)
        ax.set_xlabel('Hole Card', fontsize=my_fontsize, labelpad=10)
        ax.tick_params(left=False, bottom=False, labelleft=True, labelbottom=False)

    fig.tight_layout()
    return fig

def visualize_hand_strength_on_board(model: HandStrengthModel, board: List[Card] = []) -> Figure:
    """
    Visualizes the hand strengths learned by the HandStrengthModel on specific boards.

    Args:
        model (HandStrengthModel): The trained model.
        board: A list of community cards

    Returns:
        matplotlib.figure.Figure: The generated heatmap figure.
    """
    model.eval()
    # Create a grid of card indices
    num_ranks = model.num_ranks
    num_suits = model.num_suits
    num_hole_cards = model.num_hole_cards
    if num_hole_cards == 2:   # Visualization for 2 hole cards (e.g., Texas Hold'em)
        heatmap_data = np.zeros((num_ranks*num_suits, num_ranks*num_suits))
        annotations = np.zeros((num_ranks*num_suits, num_ranks*num_suits), dtype=object)
    elif num_hole_cards == 1: # Visualization for 1 hole card (e.g., Leduc or Kuhn Poker)
        heatmap_data = np.zeros((num_ranks, num_suits))
        annotations = np.zeros((num_ranks, num_suits), dtype=object)
    else:
        raise NotImplementedError("Visualization only implemented for 1 or 2 hole cards.")
    
    board.sort(key=lambda card: CHAR_RANK_TO_INT_RANK[card.rank])
    community_cards = cards_to_ints(board, num_ranks)
    all_cards = torch.arange(num_ranks * num_suits) + 1

    # Remove community cards from possible hands
    for card in community_cards:
        all_cards = all_cards[all_cards!=card]

    # Generate all possible hole card combinations
    hole_card_combinations = torch.combinations(all_cards, r=num_hole_cards)
    
    pretty_cards = [ints_to_cards(card_int, num_ranks) for card_int in hole_card_combinations.tolist()]
    if num_hole_cards > 1:
        sorted_hands = []
        for hand in pretty_cards:
            hand.sort(key=lambda card: CHAR_RANK_TO_INT_RANK[card.rank])
            sorted_hands.append(cards_to_ints(hand, num_ranks))

        hole_card_combinations = torch.tensor(sorted_hands, dtype=torch.int64)

    community_card_tensor = torch.tensor(community_cards, dtype=torch.int64).expand(len(hole_card_combinations),-1)

    # Add padding to match the expected input size of the model
    if model.num_community_cards-len(board) > 0:
        community_card_padding = torch.zeros((len(hole_card_combinations), model.num_community_cards-len(board)), dtype=torch.int64)
        community_card_tensor = torch.cat((community_card_tensor, community_card_padding), dim=1)
    input_tensor = torch.cat((hole_card_combinations, community_card_tensor), dim=1)
    input_tensor = input_tensor.to(next(model.parameters()).device)

    # Predict hand strengths
    with torch.no_grad():
        predicted_strengths = model.win_prob(input_tensor).cpu().detach().numpy()

    for hand, idx, win_prob in zip(pretty_cards, hole_card_combinations.numpy(), predicted_strengths):
        if num_hole_cards == 2:
            card_1, card_2 = hand
            card_1_idx, card_2_idx = idx-1
            hand_notation = f"{card_1.__str__()}{card_2.__str__()}"
            pos_2 = (card_2_idx, card_1_idx)
            heatmap_data[pos_2] = win_prob.item()
            annotations[pos_2] = hand_notation
        elif num_hole_cards == 1:
            card_1 = hand[0]
            card_1_idx = CHAR_RANK_TO_INT_RANK_REVERSED[card_1.rank]
            card_2_idx = int(math.log2(UNPRETTY_SUITS[card_1.suit]))
            hand_notation = f"{card_1.__str__()}"
        
        pos = (card_1_idx, card_2_idx)  # Lower triangle (swap indexes)
        
        heatmap_data[pos] = win_prob.item()
        annotations[pos] = hand_notation

    if num_hole_cards == 2:
        heatmap_data = heatmap_data[::-1,::-1]
        annotations = annotations[::-1,::-1]

    annotations[annotations==0]=''
    annotations = np.char.add(annotations, np.char.mod('\n%d%%', np.round(heatmap_data*100)))
    my_fontsize = 50/np.sqrt(len(heatmap_data))

    if num_hole_cards == 2:
        fig = plt.figure(figsize=(15, 15))
        ax = fig.add_subplot()
        ax = heatmap(
            heatmap_data, 
            annot=annotations,
            vmin=0, vmax=1,
            fmt='', 
            cmap="RdGy_r",
            cbar=False,
            square=True,
            annot_kws={"fontsize": my_fontsize, "fontweight": "bold"},
            ax=ax
            )
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    elif num_hole_cards == 1:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot()
        ax = heatmap(
            heatmap_data,
            annot=annotations,
            vmin=0, vmax=1,
            fmt='',
            cmap="RdGy_r",
            cbar=False,
            square=True,
            annot_kws={"fontsize": my_fontsize, "fontweight": "bold"},
            ax=ax
            )
        ax.set_yticks(np.arange(num_ranks) + 0.5, list(CHAR_RANK_TO_INT_RANK_REVERSED.keys())[:num_ranks], fontsize=20, rotation=0)
        ax.set_xticks(np.arange(num_suits) + 0.5, list(UNPRETTY_SUITS.keys())[:num_suits], fontsize=20, rotation=0)
        ax.tick_params(left=False, bottom=False, labelleft=True, labelbottom=True)
    
    ax.set_title(f'Hand Strength {model.name} {model.num_players} Player', fontsize=22, pad=20)
    ax.set_xlabel(f'Board: {[c.__str__() for c in board]}', fontsize=20, labelpad=10)
    
    fig.tight_layout()
    return fig

def visualize_calibration_curve(model: HandStrengthModel, data: DataLoader, fig:Figure = None) -> Figure:
    """
    Visualizes the calibration curve of the HandStrengthModel.

    Args:
        model (HandStrengthModel): The trained model.
        data (DataLoader): DataLoader containing the dataset for calibration.
        fig (Figure, optional): Existing figure to update. Defaults to None.

    Returns:
        matplotlib.figure.Figure: The generated calibration curve figure.
    """
    model.eval()
    probs = []
    targets = []
    with torch.no_grad():
        for input_tensor, target in data:
            input_tensor = input_tensor.to(next(model.parameters()).device)
            probs.extend(model.win_prob(input_tensor).cpu().detach().numpy())
            targets.extend(target.numpy())

    if fig is None:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot()
    else:
        ax = fig.axes[0]
        ax.clear()
    
    ax.set_title(f'Calibration Curve {model.name} {model.num_players} Player', fontsize=22, pad=20)
    display = CalibrationDisplay.from_predictions(targets, probs, n_bins=10, ax=ax, name=f'{model.name} {model.num_players} Player')
    fig.tight_layout()
    return fig

def visualize_card_embeddings(model: HandStrengthModel, fig:Figure = None) -> Figure:
    """
    Visualizes the card embeddings learned by the HandStrengthModel.

    Args:
        model (HandStrengthModel): The trained model.
        fig (Figure, optional): Existing figure to update. Defaults to None.

    Returns:
        matplotlib.figure.Figure: The generated scatter plot figure.
    """

    CARD_COLORS = {
        '♠': 'black',
        '♥': 'red',
        '♦': 'blue',
        '♣': 'green'
        }

    embeddings = model.card_embedding.weight.cpu().detach().numpy()
    tsne = TSNE(n_components=2,perplexity=model.num_classes//3)
    embeddings_tsne = tsne.fit_transform(embeddings[1:])  # Exclude padding index

    cards = ints_to_cards(list(range(1, model.num_classes)), model.num_ranks)
    suits = [card.suit for card in cards]

    if fig is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot()
    else:
        ax = fig.axes[0]
        ax.clear()

    ax.set_title(f'Card Embeddings {model.name} {model.num_players} Player', fontsize=22, pad=20)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    ax = scatterplot(
        x=embeddings_tsne[:,0],
        y=embeddings_tsne[:,1],
        hue=suits,              # Color per suits
        palette=CARD_COLORS,
        style=[i for i, _ in enumerate(cards)], # Different 52 shapes for each card
        markers={i: f'${card.rank}{card.suit}$' for i, card in enumerate(cards)},   # Card notation as marker
        s=500,
        linewidth=0,
        legend=False,
        ax=ax
        )

    fig.tight_layout()
    return fig

if __name__ == "__main__":
    config=clubs.configs.LEDUC_TWO_PLAYER.copy()

    leduc_boards = [[],[Card('AS')],[Card('KS')],[Card('QS')],[Card('Ah')],[Card('Kh')],[Card('QH')]]

    flush_draw = [Card('4h'), Card('2c'),Card('2h')]
    street_draw = [Card('5s'), Card('2h'),Card('8d')]
    flush = [Card('Qc'), Card('2c'), Card('7c'), Card('Kc')]
    street = [Card('Ts'), Card('8c'), Card('9h'), Card('Qd')]
    paper_example = [Card('2d'), Card('Ts'), Card('Kh')] 


    holdem_boards = [flush_draw, street_draw, flush, street, paper_example]

    try:
        kwargs, weights = torch.load(f'model_weights/{config["name"]}_{config["num_players"]}/hand_strength_predictor.pth')
    except FileNotFoundError:
        raise FileNotFoundError(f"No model weights found for {config['name']}_{config['num_players']} Player. Please train the model first by running train_handstrength.py with the same config.")
    model = HandStrengthModel(
        config=config,
        **kwargs
    )

    model.load_state_dict(weights)
    
    os.makedirs(f'hand_strength/{model.name}_{model.num_players}', exist_ok=True)

    try:
        card_data_array = np.loadtxt(f'data/{config["name"]}_{config["num_players"]}/simulated_cards.csv', delimiter=',', dtype=int)
        outcome_data_array = np.expand_dims(np.loadtxt(f'data/{config["name"]}_{config["num_players"]}/simulated_outcomes.csv', delimiter=',', dtype=np.float32), 1)
        train_data, _ = build_dataset(card_data_array, outcome_data_array, batch_size=1024, test_split=0)
    except FileNotFoundError:
        raise FileNotFoundError(f"No dataset found for {config['name']}_{config['num_players']} Player. Please generate the dataset first by running train_handstrength.py with the same config.")

    fig = visualize_preflop_hand_strength(model)
    plt.savefig(f'hand_strength/{model.name}_{model.num_players}/poker_hand_strength.png', dpi=300, bbox_inches='tight')
    plt.show()

    fig = visualize_calibration_curve(model, train_data)
    plt.savefig(f'hand_strength/{model.name}_{model.num_players}/calibration_curve.png', dpi=300, bbox_inches='tight')
    plt.show()

    fig = visualize_card_embeddings(model)
    
    plt.savefig(f'hand_strength/{model.name}_{model.num_players}/card_embeddings.png', dpi=300, bbox_inches='tight')
    plt.show()

    if config["name"] == 'Leduc' :
        for board in leduc_boards:
            fig=visualize_hand_strength_on_board(model, board)
            plt.savefig(f'hand_strength/{model.name}_{model.num_players}/hand_strength_{[c.__str__() for c in board]}.png', dpi=300, bbox_inches='tight')
            plt.show()
    elif config["name"] == 'Holdem':
        for board in holdem_boards:
            fig=visualize_hand_strength_on_board(model, board)
            plt.savefig(f'hand_strength/{model.name}_{model.num_players}/hand_strength_{[c.__str__() for c in board]}.png', dpi=300, bbox_inches='tight')
            plt.show()


    