import os
import numpy as np
import math

import torch
from torch.utils.data import TensorDataset, DataLoader

from clubs import poker
from clubs.poker.card import PRETTY_SUITS, CHAR_RANK_TO_INT_RANK, CHAR_SUIT_TO_INT_SUIT

UNPRETTY_SUITS = {value: key for key, value in PRETTY_SUITS.items()}
INT_RANK_TO_CHAR_RANK = {value: key for key, value in CHAR_RANK_TO_INT_RANK.items()}
INT_SUIT_TO_CHAR_SUIT = {value: key for key, value in CHAR_SUIT_TO_INT_SUIT.items()}
CHAR_RANK_TO_INT_RANK_REVERSED = {key: value for key, value in zip(reversed(CHAR_RANK_TO_INT_RANK.keys()), CHAR_RANK_TO_INT_RANK.values())}

def cards_to_ints(cards: list[poker.Card], num_ranks : int = 13) -> list[int]:
    """
    Converts a list of lists of Card objects to a list of lists of integers.
    Each card is represented as an integer, e.g. 52 integers represent the cards in a standard deck.

    Args:
        cards (list[Card]): List of Card objects.
        num_ranks (int): Number of ranks in the deck, default is 13 for a standard deck.

    Returns:
        list[int]: List of integers representing the cards.
    """
    return [int(math.log2(UNPRETTY_SUITS[card.suit])) * num_ranks + (CHAR_RANK_TO_INT_RANK[card.rank] - (13-num_ranks)) + 1 for card in cards]

def ints_to_cards(card_ints: list[int], num_ranks: int = 13) -> list[poker.Card]:
    """
    Converts a list of integers to a list of Card objects.
    Each card is represented as an integer, e.g. 52 integers represent the cards in a standard deck.

    Args:
        card_ints (list[int]): List of integers representing the cards.
        num_ranks (int): Number of ranks in the deck, default is 13 for a standard deck.

    Returns:
        list[Card]: List of Card objects.
    """
    cards = []
    for card_int in card_ints:
        suit = INT_SUIT_TO_CHAR_SUIT[2 ** ((card_int - 1) // num_ranks)]
        rank = INT_RANK_TO_CHAR_RANK[(card_int - 1) % num_ranks + (13 - num_ranks)]
        cards.append(poker.Card(f"{rank}{suit}"))

    return cards

def build_dataset(input: np.ndarray, target: np.ndarray, batch_size: int = 128, test_split:int = 20) -> tuple[DataLoader, DataLoader]:
    """
    Builds a PyTorch DataLoader from the given input and target numpy arrays.

    Args:
        input (np.ndarray): Numpy array containing input data.
        target (np.ndarray): Numpy array containing target data.
        batch_size (int): Batch size for the DataLoader.
        test_split (int): Percentage of data to use for testing (default is 20%).

    Returns:
        tuple[DataLoader, DataLoader]: A tuple containing:
            - train_loader (DataLoader): DataLoader for the training data.
            - val_loader (DataLoader): DataLoader for the validation data.
    """
    test_split = test_split / 100
    train_dataset = TensorDataset(torch.from_numpy(input[:round(len(input)*(1-test_split))]), torch.from_numpy(target[:round(len(target)*(1-test_split))]))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_dataset = TensorDataset(torch.from_numpy(input[:round(len(input)*test_split)]), torch.from_numpy(target[:round(len(target)*test_split)]))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    return train_loader, test_loader

def all_subdirs_of(b="."):
    result = []
    for d in os.listdir(b):
        bd = os.path.join(b, d)
        if os.path.isdir(bd):
            result.append(bd)
    return result

def moving_avg(arr, window=100) -> np.ndarray:
    """
    Computes the trailing moving average of a 1D array.

    Args:
        arr (array-like): Input 1D array.
        window (int): Window size for the moving average.

    Returns:
        np.ndarray: Array containing the moving average values.
    """
    # Trailing moving average, returns same-length array.
    arr = np.asarray(arr, dtype=float)
    n = len(arr)
    if n == 0:
        return arr
    w = min(window, n)
    c = np.cumsum(np.insert(arr, 0, 0.0))
    out = np.empty(n, dtype=float)
    for i in range(n):
        start = max(0, i - w + 1)
        count = i - start + 1
        out[i] = (c[i + 1] - c[start]) / count
    return out
