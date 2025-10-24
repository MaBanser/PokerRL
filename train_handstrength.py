import os
import time
from time import process_time
import numpy as np
import math
from typing import TypedDict

from sklearn.metrics import brier_score_loss

import torch
from torch.utils.data import DataLoader

import clubs
from clubs.configs import PokerConfig
from clubs.poker.card import CHAR_RANK_TO_INT_RANK

from poker import agents
from poker.models.hand_strength_predictor import HandStrengthModel
from poker.envs import ClubsEnv
from poker.utils import cards_to_ints, build_dataset
from poker.visualization import create_brier_plot, update_brier_plot
from visualize_hand_strength import visualize_preflop_hand_strength, visualize_calibration_curve, visualize_card_embeddings

class ModelPerformance(TypedDict):
    train_losses: float
    test_losses: float
    brier_scores: float



def train_hand_strength_predictor(
        config: PokerConfig,
        num_samples: int = 1000000,
        batch_size: int = 128,
        test_split: int = 20,
        embedding_dim: int = 8,
        hidden_size: int = 32,
        num_hidden_layers: int = 2,
        num_epochs: int = 100,
        learning_rate: float = 0.0001,
        min_delta: float = 0.0001,
        patience: int = 20,
        load_data: bool = False,
        visualize_progress: bool = False,
        device: torch.device = torch.device('cpu')
        ) -> tuple[ModelPerformance, HandStrengthModel]:
    """
    Trains a hand strength predictor model on a given poker configuration.

    Args:
        config (PokerConfig): Configuration for the poker game, see poker/clubs/clubs/configs.py for details.
        num_samples (int): Number of samples to collect, default is 1000000.
        batch_size (int): Size of the batches to be used in the DataLoader, default is 128.
        test_split (int): Percentage of data to use for testing (default is 20%).
        embedding_dim (int): Dimension of the card embedding vectors, default is 8.
        hidden_size (int): Size of the hidden layers in the model, default is 32.
        num_hidden_layers (int): Number of hidden layers in the model, default is 2.
        num_epochs (int): Maximum number of epochs to train the model, default is 100.
        learning_rate (float): Learning rate for the optimizer, default is 0.0001.
        min_delta (float): Minimum change in loss to qualify as an improvement, default is 0.0001.
        patience (int): Number of epochs with no improvement after which training will be stopped early, default is 20.
        load_data (bool): If True, loads previously collected data from disk instead of collecting new data, default is False.
        visualize_progress (bool): If True, visualizes the training progress, default is False.
        device (torch.device): The device the training should be performed on.

    Returns:
        tuple[HandStrengthModel, ModelPerformance]: A tuple containing:
            - HandStrengthModel: The trained hand strength predictor model.
            - ModelPerformance: A dictionary containing the losses and brier score during training.
    """

    if load_data:
        card_data_array = np.loadtxt(f'data/{config["name"]}_{config["num_players"]}/simulated_cards.csv', delimiter=',', dtype=int)
        outcome_data_array = np.expand_dims(np.loadtxt(f'data/{config["name"]}_{config["num_players"]}/simulated_outcomes.csv', delimiter=',', dtype=np.float32), 1)
    else:
        card_data_array, outcome_data_array = collect_data(config, num_samples)
        os.makedirs(f'data/{config["name"]}_{config["num_players"]}', exist_ok=True)
        np.savetxt(f'data/{config["name"]}_{config["num_players"]}/simulated_cards.csv', card_data_array, delimiter=',', fmt='%d')
        np.savetxt(f'data/{config["name"]}_{config["num_players"]}/simulated_outcomes.csv', outcome_data_array, delimiter=',', fmt='%.2f')
    train_data, test_data = build_dataset(card_data_array, outcome_data_array, batch_size, test_split)

    model = HandStrengthModel(
        config,
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        num_hidden_layer=num_hidden_layers
    )
    model.to(device)

    binary_cross_entropy = torch.nn.BCEWithLogitsLoss()
    brier_loss = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print(f'Start training with {len(card_data_array)} samples')
    model_performance = train_model(model, train_data, test_data, num_epochs, binary_cross_entropy, brier_loss, optimizer, min_delta, patience, visualize_progress, device)

    return model, model_performance


def collect_data(config : PokerConfig, num_samples : int = 100000, verbose : bool = False) -> tuple[np.ndarray, np.ndarray]:
    """
    Collects data for training the HandStrengthPredictor.
    Simulates games in the Clubs environment and collects cards and game outcome.
    100000 samples from six player Holdem game take about 7 seconds to collect on a standard laptop.

    Args:
        config (PokerConfig): Configuration for the poker game, see poker/clubs/clubs/configs.py for details.
        num_samples (int): Number of samples to collect, default is 100000.
        verbose (bool): If True, prints progress information during data collection.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - cards (np.ndarray): Array of shape (num_samples, num_cards) containing the cards played in each hand.
            - outcomes (np.ndarray): Array of shape (num_samples, 1) containing a binary outcome for each hand (1 for win, 0 for loss).
    """

    num_hole_cards = config['num_hole_cards']
    num_community_cards = sum(config['num_community_cards'])
    num_cards = num_hole_cards + num_community_cards
    num_ranks = config['num_ranks']
    num_players = config['num_players']
    num_streets = config['num_streets']

    num_runs = int(math.ceil(num_samples / (num_players * num_streets))) # Total number of runs required to collect num_samples

    # Initialize Clubs environment
    env = ClubsEnv(config, [agents.CallAgent()] * num_players)

    data_cards = np.zeros((num_runs*num_players*num_streets, num_cards), dtype=int)
    data_outcomes = np.zeros((num_runs*num_players*num_streets, 1), dtype=np.float32)  # Binary outcome for each hand

    number_of_hands=0
    number_of_rounds = 0

    idx = 0
    t = process_time()
    for _ in range(num_runs):
        obs = env.reset(reset_stacks=True)
        while True:
            bet = env.act(obs)
            obs, rewards, done, _ = env.step(bet)
            if all(done):
                # Collect cards and outcome for every player
                hole_cards = env.hole_cards
                community_cards = obs['community_cards']
                outcomes = np.zeros_like(rewards, dtype=np.float32)
                outcomes[np.argwhere(np.array(rewards)==max(rewards))] = 1  # Set the winner's outcome to 1, others to 0]

                for hand, outcome in zip(hole_cards, outcomes):
                    # Sort the hand for sample efficiency
                    hand.sort(key=lambda card: CHAR_RANK_TO_INT_RANK[card.rank])
                    # Convert cards to embedding index
                    hand = cards_to_ints(hand, num_ranks)
                    for street_length in np.cumsum(config['num_community_cards']):
                        table = community_cards[:street_length]
                        table.sort(key=lambda card: CHAR_RANK_TO_INT_RANK[card.rank])
                        table = cards_to_ints(table, num_ranks)
                        cards = hand + table
                        data_cards[idx][:len(cards)] = cards
                        data_outcomes[idx] = outcome
                        idx += 1

                    number_of_hands += 1

                number_of_rounds += 1
                if verbose:
                    print(f"Round {number_of_rounds}, Hands collected: {number_of_hands}, Samples collected: {idx}")
                break

    elapsed_time = process_time() - t
    print(f"Round {number_of_rounds}, Hands collected: {number_of_hands}, Samples collected: {idx}")
    print(f"Data generation completed in {elapsed_time:.2f} seconds.")

    return data_cards[:num_samples], data_outcomes[:num_samples]


def train_model(model: HandStrengthModel,
                train_data: DataLoader,
                test_data: DataLoader,
                num_epochs: int,
                criterion_1: torch.nn.modules.loss._Loss,
                criterion_2: torch.nn.modules.loss._Loss,
                optimizer: torch.optim.Optimizer,
                min_delta: float = 0.0001,
                patience: int = 10,
                visualize_progress: bool = False,
                device: torch.device = torch.device('cpu')
                ) -> ModelPerformance:
    """
    Trains the HandStrengthModel on the provided training data.

    Args:
        model (HandStrengthModel): The model to be trained.
        train_data (DataLoader): DataLoader containing the training data.
        test_data (DataLoader): DataLoader containing the validation data.
        num_epochs (int): Number of epochs to train the model.
        criterion_1 (torch.nn.modules.loss._Loss): Loss function to be used for training.
        criterion_2 (torch.nn.modules.loss._Loss): Second loss function to be used for training.
        optimizer (torch.optim.Optimizer): Optimizer to be used for training.
        min_delta (float): Minimum change in loss to qualify as an improvement, default is 0.0001.
        patience (int): Number of epochs with no improvement after which training will be stopped, default is 10.
        visualize_progress (bool): If True, visualizes the training progress, default is False.
        device (torch.device): The device the training should be performed on.

    Returns:
        ModelPerformance: A dictionary containing the train_losses and brier score during training.
    """
    os.makedirs(f'training_progress/hand_strength/{model.name}_{model.num_players}', exist_ok=True)
    running_average_factor = 0.95

    # Initialize lists to store loss and brier score
    train_losses = []
    test_losses = []
    brier_scores = []
    epoch_times = []

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_brier_score = float('inf')
    epochs_no_improve = 0
    best_model_weights = None

    if visualize_progress:
        # Initialize plot
        training_curves = create_brier_plot(num_epochs, train_losses, test_losses, brier_scores, threshold=0.25)
        training_curves.savefig(f'training_progress/hand_strength/{model.name}_{model.num_players}/training_curves_epoch_0.png', dpi=300, bbox_inches='tight')
        hand_strength_chart = visualize_preflop_hand_strength(model)
        hand_strength_chart.axes[0].set_xlabel(f'Epoch 0', fontsize=20, labelpad=10)
        hand_strength_chart.savefig(f'training_progress/hand_strength/{model.name}_{model.num_players}/poker_hand_strength_epoch_0.png', dpi=300, bbox_inches='tight')
        calibration_curve = visualize_calibration_curve(model, test_data)
        calibration_curve.savefig(f'training_progress/hand_strength/{model.name}_{model.num_players}/calibration_curve_epoch_0.png', dpi=300, bbox_inches='tight')
        embedding_plot = visualize_card_embeddings(model)
        embedding_plot.savefig(f'training_progress/hand_strength/{model.name}_{model.num_players}/card_embeddings_epoch_0.png', dpi=300, bbox_inches='tight')

    for epoch in range(num_epochs):
        t = process_time()
        running_loss = 0.0
        
        for data_tensor, target_tensor in train_data:
            data_tensor = data_tensor.to(device, non_blocking=True)
            target_tensor = target_tensor.to(device, non_blocking=True)
            loss = train_step(model, data_tensor, target_tensor, criterion_1, criterion_2, optimizer)
            running_loss = running_average_factor * running_loss + (1 - running_average_factor) * loss

        train_losses.append(running_loss)

        test_loss, brier_score = test(model, test_data, criterion_1, criterion_2, device)
        test_losses.append(test_loss)
        brier_scores.append(brier_score)

        scheduler.step(test_loss)

        elapsed_time = process_time() - t
        epoch_times.append(elapsed_time)
        remaining_time = np.mean(epoch_times) * (num_epochs - epoch - 1)
        print(f'Epoch {epoch+1}/{num_epochs} | '
              f'Train Loss: {running_loss:.6f} | '
              f'Test Loss: {test_loss:.6f} | '
              f'Brier: {brier_score:.6f} | '
              f'LR: {scheduler.get_last_lr()[0]:.6f} | '
              f'ETA: {time.strftime("%H:%M:%S", time.gmtime(remaining_time))}')
        
        if (epoch+1) % 10 == 0 and visualize_progress:
            update_brier_plot(training_curves, epoch, train_losses, test_losses, brier_scores)
            training_curves.savefig(f'training_progress/hand_strength/{model.name}_{model.num_players}/training_curves_epoch_{epoch+1}.png', dpi=300, bbox_inches='tight')
            hand_strength_chart = visualize_preflop_hand_strength(model, hand_strength_chart)
            hand_strength_chart.axes[0].set_xlabel(f'Epoch {epoch+1}', fontsize=20, labelpad=10)
            hand_strength_chart.savefig(f'training_progress/hand_strength/{model.name}_{model.num_players}/poker_hand_strength_epoch_{epoch+1}.png', dpi=300, bbox_inches='tight')
            calibration_curve = visualize_calibration_curve(model, test_data, calibration_curve)
            calibration_curve.savefig(f'training_progress/hand_strength/{model.name}_{model.num_players}/calibration_curve_epoch_{epoch+1}.png', dpi=300, bbox_inches='tight')
            embedding_plot = visualize_card_embeddings(model, embedding_plot)
            embedding_plot.savefig(f'training_progress/hand_strength/{model.name}_{model.num_players}/card_embeddings_epoch_{epoch+1}.png', dpi=300, bbox_inches='tight')

        if brier_score < best_brier_score - min_delta:
            best_brier_score = brier_score
            epochs_no_improve = 0
            best_model_weights = model.state_dict().copy()
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                if best_model_weights is not None:
                    model.load_state_dict(best_model_weights)
                break

    if not visualize_progress:
        training_curves = create_brier_plot(num_epochs, train_losses, test_losses, brier_scores, threshold=0.25)

    update_brier_plot(training_curves, epoch, train_losses, test_losses, brier_scores)
    training_curves.savefig(f'training_progress/hand_strength/{model.name}_{model.num_players}/training_curves.png', dpi=300, bbox_inches='tight')

    return ModelPerformance(train_losses=train_losses, test_losses=test_losses, brier_scores=brier_scores)

def train_step(model: HandStrengthModel, data_tensor: torch.Tensor, target_tensor: torch.Tensor, criterion_1: torch.nn.modules.loss._Loss,  criterion_2: torch.nn.modules.loss._Loss,optimizer: torch.optim.Optimizer) -> tuple[float, float]: 
    """    Performs a single training step on the model.

    Args:
        model (HandStrengthModel): The model to be trained.
        data_tensor (torch.Tensor): Input data tensor.
        target_tensor (torch.Tensor): Target tensor for the input data.
        criterion_1 (torch.nn.modules.loss._Loss): Loss function to be used for training.
        criterion_2 (torch.nn.modules.loss._Loss): Second loss function to be used for training.
        optimizer (torch.optim.Optimizer): Optimizer to be used for training.

    Returns:
        losses: A tuple containing:
            float: The loss value for the training step.
            float: The brier_score of the model on the training data.
    """
    model.train()
    optimizer.zero_grad()
    output = model(data_tensor)
    predictions = torch.sigmoid(output)
    loss = 0.3 * criterion_1(output, target_tensor) + 0.7 * criterion_2(predictions, target_tensor)
    loss.backward()
    optimizer.step()

    return loss.item()

def test(model: HandStrengthModel, data_loader: DataLoader, criterion_1: torch.nn.modules.loss._Loss,  criterion_2: torch.nn.modules.loss._Loss,device: torch.device) -> tuple[float, float]:
    """
    Evaluates the model on the provided test data.

    Args:
        model (HandStrengthModel): The model to be evaluated.
        data_loader (DataLoader): DataLoader containing the test data.
        criterion_1 (torch.nn.modules.loss._Loss): Loss function to be used for evaluation.
        criterion_2 (torch.nn.modules.loss._Loss): Second loss function to be used for evaluation.
        device (torch.device): The device the evaluation should be performed on.

    Returns:
        losses: A tuple containing:
            float: The average loss over the test dataset.
            float: The brier score over the test dataset.
    """
    model.eval()
    losses = []
    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for data_tensor, target_tensor in data_loader:
            data_tensor = data_tensor.to(device, non_blocking=True)
            target_tensor = target_tensor.to(device, non_blocking=True)
            output = model(data_tensor)
            predictions = torch.sigmoid(output)
            loss = 0.3 * criterion_1(output, target_tensor) + 0.7 * criterion_2(predictions, target_tensor)
            losses.append(loss.item())

            
            all_targets.extend(target_tensor.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    avg_loss = np.mean(losses)
    brier_score = brier_score_loss(np.array(all_targets), np.array(all_predictions))

    return avg_loss, brier_score

if __name__ == "__main__":
    config = clubs.configs.LIMIT_HOLDEM_TWO_PLAYER.copy()

    use_cuda = torch.cuda.is_available()
    device   = torch.device("cuda" if use_cuda else "cpu")

    training_kwargs ={
        'config': config,
        'num_samples': 1000000,
        'batch_size': 256,
        'test_split': 20,
        'embedding_dim': 4,
        'hidden_size': 32,
        'num_hidden_layers': 2,
        'num_epochs': 100,
        'learning_rate': 0.0005,
        'min_delta': 0.0001,
        'patience': 30,
        'load_data': True,
        'visualize_progress': True,
        'device': device
}
    
    t = process_time()
    model, _ = train_hand_strength_predictor(**training_kwargs)
    elapsed_time = process_time() - t
    print(f"Training completed in {elapsed_time:.2f} seconds.")
    os.makedirs(f'model_weights/{model.name}_{model.num_players}', exist_ok=True)
    torch.save([model.kwargs, model.state_dict()], f'model_weights/{model.name}_{model.num_players}/hand_strength_predictor.pth')