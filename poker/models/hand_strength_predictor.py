import torch
import torch.nn as nn
import torch.nn.functional as F

import torchinfo

import clubs
from clubs.configs import PokerConfig

class HandStrengthModel(nn.Module):
    def __init__(self, config: PokerConfig, embedding_dim: int, hidden_size: int, num_hidden_layer: int):
        """
        Initializes the HandStrengthModel.
        Model takes a poker hand consisting of 2 hole cards and up to 5 community cards as input.
        The input is represented as a sequence of integers, where each integer corresponds to a card.
        The model uses an embedding layer to convert the card integers into dense vectors.
        It then passes these vectors through several fully connected layers to predict
        the log probability to win with the given hand.

        Args:
            config (PokerConfig): Configuration object containing poker game settings.
            embedding_dim (int): Dimension of the embedding vectors.
            hidden_size (int): Size of the hidden layers.
            num_hidden_layer (int): Number of hidden layers.

        Attributes:
            name (str): Name of the game.
            num_players (int): Number of player.
            num_ranks (int): Number of ranks in the deck.
            num_suits (int): Number of suits in the deck.
            num_hole_cards (int): Number of hole cards in the hand.
            num_community_cards (int): Total number of community cards.
        """

        super(HandStrengthModel, self).__init__()
        self.kwargs = {
            'embedding_dim': embedding_dim,
            'hidden_size': hidden_size,
            'num_hidden_layer': num_hidden_layer
            }
        self.name = config["name"]
        self.num_players = config["num_players"]
        self.num_ranks = config['num_ranks']
        self.num_suits = config['num_suits']
        self.num_hole_cards = config['num_hole_cards']
        self.num_community_cards = sum(config['num_community_cards'])
        
        self.num_classes = self.num_ranks * self.num_suits + 1 # +1 for padding index
        self.num_cards = self.num_hole_cards + self.num_community_cards

        self.card_embedding = nn.Embedding(self.num_classes, embedding_dim, padding_idx=0)

        self.fc1 = nn.Linear(embedding_dim * self.num_cards, hidden_size)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(num_hidden_layer)]
        )

        self.out = nn.Linear(hidden_size, 1)  # Output layer for win log probability

    def forward(self, cards: torch.Tensor) -> torch.Tensor:
        cards = self.card_embedding(cards)

        cards = cards.view(cards.size(0), -1)  # Flatten the input

        cards = F.relu(self.fc1(cards))
        for layer in self.hidden_layers:
            cards = F.relu(layer(cards))
        cards = self.out(cards)
        return cards
    
    def win_prob(self, cards: torch.Tensor) -> torch.Tensor:
        """
        Predicts the win probability for a given poker hand.

        Args:
            cards (torch.Tensor): Input tensor of shape (batch_size, num_cards) containing card indices.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1) containing the predicted win probabilities.
        """
        with torch.no_grad():
            p = torch.sigmoid(self.forward(cards))
        return p

if __name__ == "__main__":
    # Example usage
    config = clubs.configs.NO_LIMIT_HOLDEM_SIX_PLAYER
    embedding_dim = 10  # Dimension of the embedding vectors
    hidden_size = 16  # Size of the hidden layers
    num_hidden_layer = 1 # Number of hidden layers
    model = HandStrengthModel(config, embedding_dim, hidden_size, num_hidden_layer)
    model.eval()

    # Example input: batch of 2 poker hands, each with 7 cards (2 hole cards + 5 community cards)
    example_input = torch.randint(1, model.num_classes, (2, model.num_cards))
    example_input[0, 0] = 0
    example_input[1, 0] = 52
    torchinfo.summary(model, input_data=example_input, col_names=["input_size", "output_size", "num_params"])
    with torch.no_grad():
        output = model(example_input)
    print("Example output (log win probabilities):", output)    
    print("Example output (win probabilities):", model.win_prob(example_input))