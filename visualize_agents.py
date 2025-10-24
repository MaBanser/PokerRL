import os
import numpy as np
import matplotlib.pyplot as plt
import json
import re

import clubs
from poker.utils import moving_avg


def get_names(data_dict) -> tuple[list[str], list[str]]:
    """
    Extract agent and opponents names from dictionary keys.

    Args:
        data_dict: Dictionary containing results data.

    Returns:
        tuple: A tuple containing two lists - agents and opponents.
    """
    agents = sorted(set(list(data_dict.keys())))
    opponents = list(data_dict[agents[0]].keys())
    return agents, opponents

def process_data(baseline_dict, compare_dict, comp_name='comparison') -> dict:
    """
    Process the raw data into structured format for plotting.
    
    Args:
        baseline_dict: Dictionary containing baseline agent results.
        compare_dict: Dictionary containing comparison agent results (e.g., equity agents).
        comp_name: Name of the comparison agent.
    
    Returns:
        dict: A dictionary with organized data for baseline and comparison agents.
    """
    # Get agent and opponent names
    baseline_agents, baseline_opponents = get_names(baseline_dict)
    compare_agents, compare_opponents = get_names(compare_dict)

    if bool(set(baseline_opponents).intersection(compare_opponents)):
        opponents = baseline_opponents
    
    # Initialize result structure
    processed = {
        'baseline': {opp: {'epochs': None, 
                          'chips': {'mean': None, 'std': None, 'individual': []},
                          'win_tie_loss': {'mean': None, 'individual': []},
                          'actions': {'mean': None, 'individual': []}} 
                 for opp in opponents},
        comp_name: {opp: {'epochs': None, 
                            'chips': {'mean': None, 'std': None, 'individual': []},
                            'win_tie_loss': {'mean': None, 'individual': []},
                            'actions': {'mean': None, 'individual': []}} 
                 for opp in opponents}
    }

    n = re.search(r'\d+',list(baseline_dict[baseline_agents[0]][opponents[0]].keys())[1])[0]
    

    # Process baseline agents
    for agent_name in baseline_agents:
        agent_data = baseline_dict[agent_name]
        
        for opponent in opponents:
            if opponent not in agent_data:
                continue
                
            # Extract epochs
            epochs = np.array(agent_data[opponent]['Epoch'])
            
            # Process chips/hand data
            chips = np.array(agent_data[opponent][f'Chips/hand over {n} hands'])
            std = np.array(agent_data[opponent][f'Std over {n} hands'])
            
            # Process win/tie/loss rates
            wins = np.array(agent_data[opponent][f'Wins/{n} hands']) / float(n)
            ties = np.array(agent_data[opponent][f'Ties/{n} hands']) / float(n)
            losses = 1.0 - (wins + ties)
            
            # Process action proportions
            actions = []
            for action_dict in agent_data[opponent]['Actions']:
                total = sum(action_dict.values())
                fold = int(action_dict.get('0', 0)) / total
                call = int(action_dict.get('1', 0)) / total
                raise_ = int(action_dict.get('2', 0)) / total
                actions.append([fold, call, raise_])
            actions = np.array(actions)
            
            # Store individual agent data
            if processed['baseline'][opponent]['epochs'] is None:
                processed['baseline'][opponent]['epochs'] = epochs
            
            processed['baseline'][opponent]['chips']['individual'].append((chips, std))
            processed['baseline'][opponent]['win_tie_loss']['individual'].append(np.column_stack([wins, ties, losses]))
            processed['baseline'][opponent]['actions']['individual'].append(actions)
    
    # Process comparison agents
    for agent_name in compare_agents:
        agent_data = compare_dict[agent_name]
        
        for opponent in opponents:
            if opponent not in agent_data:
                continue
                
            # Extract epochs
            epochs = np.array(agent_data[opponent]['Epoch'])
            
            # Process chips/hand data
            chips = np.array(agent_data[opponent][f'Chips/hand over {n} hands'])
            std = np.array(agent_data[opponent][f'Std over {n} hands'])
            
            # Process win/tie/loss rates
            wins = np.array(agent_data[opponent][f'Wins/{n} hands']) / float(n)
            ties = np.array(agent_data[opponent][f'Ties/{n} hands']) / float(n)
            losses = 1.0 - (wins + ties)
            
            # Process action proportions
            actions = []
            for action_dict in agent_data[opponent]['Actions']:
                total = sum(action_dict.values())
                fold = int(action_dict.get('0', 0)) / total
                call = int(action_dict.get('1', 0)) / total
                raise_ = int(action_dict.get('2', 0)) / total
                actions.append([fold, call, raise_])
            actions = np.array(actions)
            
            # Store individual agent data
            if processed[comp_name][opponent]['epochs'] is None:
                processed[comp_name][opponent]['epochs'] = epochs
            
            processed[comp_name][opponent]['chips']['individual'].append((chips, std))
            processed[comp_name][opponent]['win_tie_loss']['individual'].append(np.column_stack([wins, ties, losses]))
            processed[comp_name][opponent]['actions']['individual'].append(actions)

    for opponent in opponents:
        # Calculate aggregated values for baseline
        if not processed['baseline'][opponent]['chips']['individual']:
            continue
            
        n_baseline = len(processed['baseline'][opponent]['chips']['individual'])

        # Chips/hand aggregation
        chips_list = [item[0] for item in processed['baseline'][opponent]['chips']['individual']]
        std_list = [item[1] for item in processed['baseline'][opponent]['chips']['individual']]
        
        # Calculate mean chips
        processed['baseline'][opponent]['chips']['mean'] = np.mean(chips_list, axis=0)
        
        # Calculate SE for confidence interval
        std_squared = np.array(std_list)**2
        se_agg = np.sqrt(np.sum(std_squared, axis=0) / (float(n) * n_baseline**2))
        processed['baseline'][opponent]['chips']['std'] = se_agg
        
        # Win/tie/loss aggregation
        win_tie_loss_data = processed['baseline'][opponent]['win_tie_loss']['individual']
        processed['baseline'][opponent]['win_tie_loss']['mean'] = np.mean(win_tie_loss_data, axis=0)
        
        # Actions aggregation
        actions_data = processed['baseline'][opponent]['actions']['individual']
        processed['baseline'][opponent]['actions']['mean'] = np.mean(actions_data, axis=0)

    
        # Calculate aggregated values for comparison
        if not processed[comp_name][opponent]['chips']['individual']:
            continue
            
        n_comparison = len(processed[comp_name][opponent]['chips']['individual'])
        # Chips/hand aggregation
        chips_list = [item[0] for item in processed[comp_name][opponent]['chips']['individual']]
        std_list = [item[1] for item in processed[comp_name][opponent]['chips']['individual']]
        
        # Calculate mean chips
        processed[comp_name][opponent]['chips']['mean'] = np.mean(chips_list, axis=0)
        
        # Calculate SE for confidence interval
        std_squared = np.array(std_list)**2
        se_agg = np.sqrt(np.sum(std_squared, axis=0) / (float(n) * n_comparison**2))
        processed[comp_name][opponent]['chips']['std'] = se_agg
        
        # Win/tie/loss aggregation
        win_tie_loss_data = processed[comp_name][opponent]['win_tie_loss']['individual']
        processed[comp_name][opponent]['win_tie_loss']['mean'] = np.mean(win_tie_loss_data, axis=0)
        
        # Actions aggregation
        actions_data = processed[comp_name][opponent]['actions']['individual']
        processed[comp_name][opponent]['actions']['mean'] = np.mean(actions_data, axis=0)
    
    return processed

def plot_chips_hand(processed_data, comp_name='comparison') -> plt.Figure:
    """
    Create the line plot for chips/hand with 95% confidence intervals.

    Args:
        processed_data: Processed data dictionary from process_data function.
        comp_name: Name of the comparison agent.
    
    Returns:
        fig: Matplotlib figure object
    """
    # Get opponents dynamically
    opponents = [k for k in processed_data['baseline'].keys() 
                if 'epochs' in processed_data['baseline'][k] and processed_data['baseline'][k]['epochs'] is not None]
    
    # If no valid opponents, return empty figure
    if not opponents:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center')
        return fig
    
    # Set up the figure with subplots (one for each opponent)
    fig, axes = plt.subplots(1, len(opponents), figsize=(6 * len(opponents), 6), sharey=True)
    fig.suptitle('Chips per Hand Over Training', fontsize=16)
    
    # Handle single opponent case
    if len(opponents) == 1:
        axes = [axes]
    
    # Define colors
    baseline_color = 'cornflowerblue'
    comparison_color = 'orangered'
    individual_alpha = 0.3
    
    # Y-axis limits
    y_min, y_max = 0, 1
    
    # Process each opponent
    for i, opponent in enumerate(opponents):
        ax = axes[i]
        
        # Create title
        title = opponent.replace("[", "").replace("]", "").replace("'", "")
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        if i == 0:
            ax.set_ylabel('Chips per Hand')
        
        # Plot baseline agents
        if processed_data['baseline'][opponent]['epochs'] is not None:
            baseline_epochs = processed_data['baseline'][opponent]['epochs']
            baseline_mean = processed_data['baseline'][opponent]['chips']['mean']
            baseline_se = processed_data['baseline'][opponent]['chips']['std']
            
            # 95% CI = mean ± 1.96 * SE
            ci_lower = baseline_mean - 1.96 * baseline_se
            ci_upper = baseline_mean + 1.96 * baseline_se
            
            # Plot individual baseline agents
            for chips, _ in processed_data['baseline'][opponent]['chips']['individual']:
                y_min = min(np.min(chips).item(),y_min)
                y_max = max(np.max(chips).item(),y_max)
                ax.plot(baseline_epochs, chips, color=baseline_color, alpha=individual_alpha, linewidth=1)
            
            # Plot aggregated baseline with CI
            ax.plot(baseline_epochs, baseline_mean, color=baseline_color, linewidth=1.5, label='Baseline')
            ax.fill_between(baseline_epochs, ci_lower, ci_upper, color=baseline_color, alpha=0.4)
        
        # Plot comparison agents
        if processed_data[comp_name][opponent]['epochs'] is not None:
            comparison_epochs = processed_data[comp_name][opponent]['epochs']
            comparison_mean = processed_data[comp_name][opponent]['chips']['mean']
            comparison_se = processed_data[comp_name][opponent]['chips']['std']
            
            # 95% CI = mean ± 1.96 * SE
            ci_lower = comparison_mean - 1.96 * comparison_se
            ci_upper = comparison_mean + 1.96 * comparison_se
            
            # Plot individual comparison agents
            for chips, _ in processed_data[comp_name][opponent]['chips']['individual']:
                y_min = min(np.min(chips).item(),y_min)
                y_max = max(np.max(chips).item(),y_max)
                ax.plot(comparison_epochs, chips, color=comparison_color, alpha=individual_alpha, linewidth=1)
            
            # Plot aggregated comparison with CI
            ax.plot(comparison_epochs, comparison_mean, color=comparison_color, linewidth=1.5, label=comp_name)
            ax.fill_between(comparison_epochs, ci_lower, ci_upper, color=comparison_color, alpha=0.4)
        
        # Set y-axis limits
        ax.set_ylim(y_min, y_max)
        # Set x-axis limits
        ax.set_xlim(comparison_epochs[0],comparison_epochs[-1])

        # Add Equilibrium Area
        ax.axhspan(-0.06, 0, color='grey', alpha=0.3)
        x_min, x_max = ax.get_xlim()
        x_annot = x_min + 0.9 * (x_max - x_min) 
        ax.annotate(
            'Expected value for equilibrium',
            xy=(x_annot, -0.03),                # Arrow tip position (in data coords)
            xytext=(0.95, 0.05),                # Text position (in axes %: bottom-right)
            textcoords='axes fraction',         # Interpret xytext as % of axes
            ha='right', va='bottom',            # Align text to bottom-right of xytext
            fontsize=9,
            bbox=dict(                          # Subtle background for readability
                boxstyle="round,pad=0.3", 
                fc="white", 
                ec="gray", 
                alpha=0.85,
                lw=0.5
            ),
            arrowprops=dict(                    # Customized arrow
                arrowstyle="-|>",               
                color='darkgrey',
                lw=1.2,
                shrinkA=5,
                shrinkB=5,
                connectionstyle="arc3,rad=-0.15"
            )
        )
        # Add grid for readability
        ax.grid(True, linestyle='--', alpha=0.7)
    
    # Create a single legend outside the plots
    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
        ax.legend().remove()
    
    # Remove duplicates from handles and labels
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) 
              if l not in labels[:i]]
    handles, labels = zip(*unique) if unique else ([], [])
    
    # Add the legend outside the rightmost subplot
    if handles:
        plt.tight_layout()
        fig.subplots_adjust(right=0.9)
        fig.legend(handles, labels, loc='center right', 
                   title='Agent Type', bbox_to_anchor=(1, 0.82))
    
    return fig

def plot_win_tie_loss(processed_data, comp_name='comparison') -> plt.Figure:
    """
    Create the stacked categorical plot for win/tie/loss rates.

    Args:
        processed_data: Processed data dictionary from process_data function.
        comp_name: Name of the comparison agent.
    
    Returns:
        fig: Matplotlib figure object
    """
    # Get opponents
    opponents = [k for k in processed_data['baseline'].keys() 
                if 'epochs' in processed_data['baseline'][k] and processed_data['baseline'][k]['epochs'] is not None]
    
    # If no valid opponents, return empty figure
    if not opponents:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center')
        return fig
    
    # Set up the figure with subplots (one for each opponent)
    fig, axes = plt.subplots(1, len(opponents), figsize=(6 * len(opponents), 6), sharey=True)
    fig.suptitle('\u0394 Outcomes Over Training', fontsize=16)
    
    # Handle single opponent case
    if len(opponents) == 1:
        axes = [axes]
    
    # Define colors
    win_color = 'forestgreen'
    tie_color = 'tan'
    loss_color = 'maroon'
    
    # Process each opponent
    for i, opponent in enumerate(opponents):
        ax = axes[i]
        
        # Create title
        title = opponent.replace("[", "").replace("]", "").replace("'", "")
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        if i == 0:
            ax.set_ylabel('\u0394 Outcome')
        
        # Set y-axis limits
        ax.set_ylim(-0.16, 0.16)
        
        # Plot baseline agents
        if processed_data['baseline'][opponent]['epochs'] is not None:
            baseline_epochs = processed_data['baseline'][opponent]['epochs']
            baseline_data = processed_data['baseline'][opponent]['win_tie_loss']['mean']

        if processed_data[comp_name][opponent]['epochs'] is not None:
            comparison_epochs = processed_data[comp_name][opponent]['epochs']
            comparison_data = processed_data[comp_name][opponent]['win_tie_loss']['mean']

            # Plot aggregated baseline
            ax.plot(baseline_epochs, comparison_data[:, 0]-baseline_data[:, 0], color=win_color, linewidth=1.5, label=f'\u0394 Win ({comp_name} - Baseline)')
            ax.fill_between(baseline_epochs, comparison_data[:, 0]-baseline_data[:, 0], color=win_color, alpha=0.3)
            ax.plot(baseline_epochs, comparison_data[:, 1]-baseline_data[:, 1], color=tie_color, linewidth=1.5, label=f'\u0394 Tie ({comp_name} - Baseline)')
            ax.fill_between(baseline_epochs, comparison_data[:, 1]-baseline_data[:, 1], color=tie_color, alpha=0.3)
            ax.plot(baseline_epochs, comparison_data[:, 2]-baseline_data[:, 2], color=loss_color, linewidth=1.5, label=f'\u0394 Loss ({comp_name} - Baseline)')
            ax.fill_between(baseline_epochs, comparison_data[:, 2]-baseline_data[:, 2], color=loss_color, alpha=0.3)
            ax.axhline(0.0, color='k', lw=1, alpha=0.4)
        
        # Set x-axis limits
        ax.set_xlim(comparison_epochs[0],comparison_epochs[-1])

        # Add grid for readability
        ax.grid(True, linestyle='--', alpha=0.7)

    # Create a single legend outside the plots
    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
        ax.legend().remove()
    
    # Remove duplicates from handles and labels
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) 
              if l not in labels[:i]]
    handles, labels = zip(*unique) if unique else ([], [])
    
    # Add the legend outside the rightmost subplot
    if handles:
        plt.tight_layout()
        fig.subplots_adjust(right=0.9)
        fig.legend(handles, labels, loc='center right', 
                   title='Agent Type', bbox_to_anchor=(1, 0.805))
    
    return fig

def plot_action_proportions(processed_data, comp_name='comparison') -> plt.Figure:
    """
    Create the stacked area plot for action proportions.

    Args:
        processed_data: Processed data dictionary from process_data function.
        comp_name: Name of the comparison agent.
    
    Returns:
        fig: Matplotlib figure object
    """
    # Get opponents
    opponents = [k for k in processed_data['baseline'].keys() 
                if 'epochs' in processed_data['baseline'][k] and processed_data['baseline'][k]['epochs'] is not None]
    
    # If no valid opponents, return empty figure
    if not opponents:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center')
        return fig
    
    # Set up the figure with subplots (one for each opponent)
    fig, axes = plt.subplots(1, len(opponents), figsize=(6 * len(opponents), 6), sharey=True)
    fig.suptitle('\u0394 Action Proportions Over Training', fontsize=16)
    
    # Handle single opponent case
    if len(opponents) == 1:
        axes = [axes]
    
    # Define colors
    fold_color = 'navy'
    call_color = 'mediumseagreen'
    raise_color = 'firebrick'
    
    # Process each opponent
    for i, opponent in enumerate(opponents):
        ax = axes[i]
        
        # Create title
        title = opponent.replace("[", "").replace("]", "").replace("'", "")
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        if i == 0:
            ax.set_ylabel('\u0394 action proportion')
        
        # Set y-axis limits
        ax.set_ylim(-0.3, 0.3)
        
        # Plot baseline agents
        if processed_data['baseline'][opponent]['epochs'] is not None:
            baseline_epochs = processed_data['baseline'][opponent]['epochs']
            baseline_data = processed_data['baseline'][opponent]['actions']['mean']

        if processed_data[comp_name][opponent]['epochs'] is not None:
            comparison_epochs = processed_data[comp_name][opponent]['epochs']
            comparison_data = processed_data[comp_name][opponent]['actions']['mean']
        
            # Plot aggregated baseline
            ax.plot(baseline_epochs, comparison_data[:, 0]-baseline_data[:, 0], color=fold_color, linewidth=1.5, label=f'\u0394 Fold ({comp_name} - Baseline)')
            ax.fill_between(baseline_epochs, comparison_data[:, 0]-baseline_data[:, 0], color=fold_color, alpha=0.3)
            ax.plot(baseline_epochs, comparison_data[:, 1]-baseline_data[:, 1], color=call_color, linewidth=1.5, label=f'\u0394 Call ({comp_name} - Baseline)')
            ax.fill_between(baseline_epochs, comparison_data[:, 1]-baseline_data[:, 1], color=call_color, alpha=0.3)
            ax.plot(baseline_epochs, comparison_data[:, 2]-baseline_data[:, 2], color=raise_color, linewidth=1.5, label=f'\u0394 Raise ({comp_name} - Baseline)')
            ax.fill_between(baseline_epochs, comparison_data[:, 2]-baseline_data[:, 2], color=raise_color, alpha=0.3)
            ax.axhline(0.0, color='k', lw=1, alpha=0.4)
        
        # Set x-axis limits
        ax.set_xlim(comparison_epochs[0],comparison_epochs[-1])

        # Add grid for readability
        ax.grid(True, linestyle='--', alpha=0.7)
        
    # Create a single legend outside the plots
    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
        ax.legend().remove()
    
    # Remove duplicates from handles and labels
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) 
              if l not in labels[:i]]
    handles, labels = zip(*unique) if unique else ([], [])
    
    # Add the legend outside the rightmost subplot
    if handles:
        plt.tight_layout()
        fig.subplots_adjust(right=0.9)
        fig.legend(handles, labels, loc='center right', 
                   title='Agent Type', bbox_to_anchor=(1, 0.805))
    
    return fig

def create_all_plots(baseline_dict, compare_dict, comp_name='comparison') -> tuple[plt.Figure, plt.Figure, plt.Figure]:
    """
    Create all three plot types from the input data.

    Args:
        baseline_dict: Dictionary containing baseline agent results.
        compare_dict: Dictionary containing comparison agent results (e.g., equity agents).
        comp_name: Name of the comparison agent.
    
    Returns:
        figures: A tuple of (chips_fig, win_tie_loss_fig, action_fig)
    """
    # Process the data
    processed_data = process_data(baseline_dict, compare_dict, comp_name)
    
    # Create each figure
    chips_fig = plot_chips_hand(processed_data, comp_name)
    win_tie_loss_fig = plot_win_tie_loss(processed_data, comp_name)
    action_fig = plot_action_proportions(processed_data, comp_name)
    
    return chips_fig, win_tie_loss_fig, action_fig

def plot_reward_and_loss(loss_dict, reward_dict) -> plt.Figure:
    """
    Create a figure with three subplots: rewards, RL loss, and SL loss.

    Args:
        loss_dict: Dictionary containing loss data for agents.
        reward_dict: Dictionary containing reward data for agents.

    Returns:
        fig: Matplotlib figure object
    """
    # Collect all agent keys
    agents, opponents = get_names(reward_dict)
    eval_agents = [f'{agent}_vs_{opponent.replace("[", "").replace("]", "").replace("'", "")}' for agent in agents for opponent in opponents]
    cmap = plt.cm.tab10
    colors = {agent: cmap(i % cmap.N) for i, agent in enumerate(agents)}
    eval_colors = {agent: cmap(i % cmap.N) for i, agent in enumerate(eval_agents)}


    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    fig.suptitle("Training metrics over epochs", fontsize=22)

    # Reward subplot
    ax_r = axes[0]
    i = 0
    for agent in agents:
        for opponent in opponents:
            rewards = reward_dict[agent][opponent][list(reward_dict[agent][opponent].keys())[1]]
            r = np.asarray(rewards, dtype=float)
            x = np.asarray(reward_dict[agent][opponent]["Epoch"], dtype=float)
            ax_r.plot(x, r, color=eval_colors[eval_agents[i]], label=eval_agents[i], linewidth=1.5)
            ax_r.plot(x, moving_avg(r, 3), color=eval_colors[eval_agents[i]], linestyle="--", linewidth=1.2)
            ax_r.set_ylabel(list(reward_dict[agent][opponent].keys())[1])
            i += 1
    ax_r.axhspan(-0.06, 0, color='grey', alpha=0.3)
    ax_r.set_ylim((-1, 2))
    ax_r.set_xmin = 0
    x_min, x_max = ax_r.get_xlim()
    x_annot = x_min + 0.9 * (x_max - x_min) 
    ax_r.annotate(
        'Expected value for equilibrium',
        xy=(x_annot, -0.03),                # Arrow tip position (in data coords)
        xytext=(0.95, 0.05),                # Text position (in axes %: bottom-right)
        textcoords='axes fraction',         # Interpret xytext as % of axes
        ha='right', va='bottom',            # Align text to bottom-right of xytext
        fontsize=9,
        bbox=dict(                          # Subtle background for readability
            boxstyle="round,pad=0.3", 
            fc="white", 
            ec="gray", 
            alpha=0.85,
            lw=0.5
        ),
        arrowprops=dict(                    # Customized arrow
            arrowstyle="-|>",
            color='darkgrey',
            lw=1.2,
            shrinkA=5,
            shrinkB=5,
            connectionstyle="arc3,rad=-0.15"  # Gentle curve
        )
    )
    ax_r.set_title("Rewards")
    ax_r.grid(True, alpha=0.3)
    ax_r.legend(loc="best", fontsize=9)

    # RL Loss subplot
    ax_rl = axes[1]
    for agent in agents:
        data = loss_dict[agent]
        rl = data['RL_Loss']
        if rl is None:
            continue
        y = np.asarray(rl, dtype=float)
        x = np.arange(1, len(y) + 1)
        ax_rl.plot(x, y, color=colors[agent], label=agent, linewidth=1.0)
    ax_rl.set_ylabel("RL Loss")
    ax_rl.set_title("RL Losses")
    ax_rl.grid(True, alpha=0.3)
    ax_rl.legend(loc="best", fontsize=9)

    # SL Loss subplot
    ax_sl = axes[2]
    for agent in agents:
        data = loss_dict[agent]
        sl = data['SL_Loss']
        if sl is None:
            continue
        y = np.asarray(sl, dtype=float)
        x = np.arange(1, len(y) + 1)
        ax_sl.plot(x, y, color=colors[agent], label=agent, linewidth=1.0)
    ax_sl.set_ylabel("SL Loss")
    ax_sl.set_xlabel("Epoch", fontsize=20)
    ax_sl.grid(True, alpha=0.3)
    ax_sl.set_title("SL Losses")
    ax_sl.legend(loc="best", fontsize=9)

    fig.tight_layout()
    return fig

def plot_rewards(reward_dict) -> plt.Figure:
    """
    Create a figure plotting rewards over epochs.

    Args:
        reward_dict: Dictionary containing reward data for agents.

    Returns:
        fig: Matplotlib figure object
    """
    # Collect all agent keys
    agents, opponents = get_names(reward_dict)
    eval_agents = [f'{agent}_vs_{opponent.replace("[", "").replace("]", "").replace("'", "")}' for agent in agents for opponent in opponents]
    cmap = plt.cm.tab10
    eval_colors = {agent: cmap(i % cmap.N) for i, agent in enumerate(eval_agents)}


    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot()
    fig.suptitle("Training metrics over epochs", fontsize=22)

    i = 0
    for agent in agents:
        for opponent in opponents:
            rewards = reward_dict[agent][opponent][list(reward_dict[agent][opponent].keys())[1]]
            r = np.asarray(rewards, dtype=float)
            x = np.asarray(reward_dict[agent][opponent]["Epoch"], dtype=float)
            ax.plot(x, r, color=eval_colors[eval_agents[i]], label=eval_agents[i], linewidth=1.5)
            ax.set_ylabel(list(reward_dict[agent][opponent].keys())[1])
            i += 1
    ax.axhspan(-0.06, 0, color='grey', alpha=0.3)
    ax.set_ylim((-1, 2))
    ax.set_xmin = 0
    x_min, x_max = ax.get_xlim()
    x_annot = x_min + 0.9 * (x_max - x_min) 
    ax.annotate(
        'Expected value for equilibrium',
        xy=(x_annot, -0.03),                # Arrow tip position (in data coords)
        xytext=(0.95, 0.05),                # Text position (in axes %: bottom-right)
        textcoords='axes fraction',         # Interpret xytext as % of axes
        ha='right', va='bottom',            # Align text to bottom-right of xytext
        fontsize=9,
        bbox=dict(                          # Subtle background for readability
            boxstyle="round,pad=0.3", 
            fc="white", 
            ec="gray", 
            alpha=0.85,
            lw=0.5
        ),
        arrowprops=dict(                    # Customized arrow
            arrowstyle="-|>",
            color='darkgrey',
            lw=1.2,
            shrinkA=5,
            shrinkB=5,
            connectionstyle="arc3,rad=-0.15"  # Gentle curve
        )
    )
    ax.set_title("Rewards")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Epoch", fontsize=20)
    ax.legend(loc="best", fontsize=9)
    
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    config=clubs.configs.LEDUC_TWO_PLAYER.copy()

    model_state = 'short_state'
    # model_state = 'long_state'

    baseline_name = 'NFSP Agent (Training)'
    agent_name = 'NFSP Agent + Equity (Training)'

    try:
        with open(f'training_progress/player/{config['name']}_{config['num_players']}/{model_state}/{baseline_name}_loss_hist.txt') as file:
            baseline_losses = json.load(file)
        with open(f'training_progress/player/{config['name']}_{config['num_players']}/{model_state}/{baseline_name}_reward_hist.txt') as file:
            baseline_metrics = json.load(file)
        with open(f'training_progress/player/{config['name']}_{config['num_players']}/{model_state}/{agent_name}_loss_hist.txt') as file:
            agent_losses = json.load(file)
        with open(f'training_progress/player/{config['name']}_{config['num_players']}/{model_state}/{agent_name}_reward_hist.txt') as file:
            agent_metrics = json.load(file)
    except FileNotFoundError:
        raise FileNotFoundError("Training progress files not found. Please ensure that the training has been completed and the files exist.")

    os.makedirs(f'agent_eval/{config['name']}_{config['num_players']}/{model_state}', exist_ok=True)

    fig = plot_reward_and_loss(baseline_losses, baseline_metrics)
    fig.savefig(f'agent_eval/{config['name']}_{config['num_players']}/{model_state}/{baseline_name}_loss.png', dpi=300, bbox_inches='tight')

    fig = plot_rewards(baseline_metrics)
    fig.savefig(f'agent_eval/{config['name']}_{config['num_players']}/{model_state}/{baseline_name}_rewards.png', dpi=300, bbox_inches='tight')

    fig = plot_reward_and_loss(agent_losses, agent_metrics)
    fig.savefig(f'agent_eval/{config['name']}_{config['num_players']}/{model_state}/{agent_name}_loss.png', dpi=300, bbox_inches='tight')

    fig = plot_rewards(agent_metrics)
    fig.savefig(f'agent_eval/{config['name']}_{config['num_players']}/{model_state}/{agent_name}_rewards.png', dpi=300, bbox_inches='tight')

    chips_fig, win_tie_loss_fig, action_fig = create_all_plots(baseline_metrics,agent_metrics,'Hand_Strength')
    chips_fig.savefig(f'agent_eval/{config['name']}_{config['num_players']}/{model_state}/chips_compare.png', dpi=300, bbox_inches='tight')
    win_tie_loss_fig.savefig(f'agent_eval/{config['name']}_{config['num_players']}/{model_state}/win_rate_compare.png', dpi=300, bbox_inches='tight')
    action_fig.savefig(f'agent_eval/{config['name']}_{config['num_players']}/{model_state}/actions_compare.png', dpi=300, bbox_inches='tight')    