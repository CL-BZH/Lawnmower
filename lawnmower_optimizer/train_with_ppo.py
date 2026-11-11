"""
Complete Training Pipeline: Imitation Learning → PPO Fine-tuning

This script trains an agent in two phases:
1. Phase 1: Imitation Learning (fast, stable baseline)
2. Phase 2: PPO Fine-tuning (discover improvements beyond experts)
"""

import argparse
import matplotlib.pyplot as plt
import pickle
import os

from train_rl_agent import LawnmowerRLAgent
from ppo_trainer import PPOTrainer
from utils.environment import LawnmowerEnvironment


def plot_training_results(agent, ppo_trainer, save_dir='plots'):
    """
    Plot comprehensive training results from both phases
    """
    os.makedirs(save_dir, exist_ok=True)
    
    fig = plt.figure(figsize=(16, 10))
    
    # Phase 1: Imitation Learning
    if agent.train_losses and agent.val_accuracies:
        # Training loss
        ax1 = plt.subplot(3, 3, 1)
        ax1.plot(agent.train_losses, label='Train Loss', color='blue')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Phase 1: Imitation Learning Loss')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Validation accuracy
        ax2 = plt.subplot(3, 3, 2)
        ax2.plot(agent.val_accuracies, label='Val Accuracy', color='green')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Phase 1: Validation Accuracy')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
    
    # Phase 2: PPO Fine-tuning
    if ppo_trainer and ppo_trainer.episode_rewards:
        # Episode rewards (smoothed)
        ax3 = plt.subplot(3, 3, 3)
        rewards = ppo_trainer.episode_rewards
        # Moving average
        window = min(50, len(rewards) // 10)
        if window > 1:
            smoothed = [sum(rewards[max(0, i-window):i+1]) / min(i+1, window) 
                       for i in range(len(rewards))]
            ax3.plot(smoothed, label='Smoothed Reward', color='orange', linewidth=2)
        ax3.plot(rewards, alpha=0.3, label='Raw Reward', color='orange')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Reward')
        ax3.set_title('Phase 2: PPO Episode Rewards')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Episode lengths
        ax4 = plt.subplot(3, 3, 4)
        lengths = ppo_trainer.episode_lengths
        if window > 1:
            smoothed = [sum(lengths[max(0, i-window):i+1]) / min(i+1, window)
                       for i in range(len(lengths))]
            ax4.plot(smoothed, label='Smoothed Length', color='purple', linewidth=2)
        ax4.plot(lengths, alpha=0.3, label='Raw Length', color='purple')
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Steps')
        ax4.set_title('Phase 2: Episode Lengths')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        # Policy loss
        ax5 = plt.subplot(3, 3, 5)
        ax5.plot(ppo_trainer.policy_losses, label='Policy Loss', color='red')
        ax5.set_xlabel('Update')
        ax5.set_ylabel('Loss')
        ax5.set_title('Phase 2: PPO Policy Loss')
        ax5.grid(True, alpha=0.3)
        ax5.legend()
        
        # Value loss
        ax6 = plt.subplot(3, 3, 6)
        ax6.plot(ppo_trainer.value_losses, label='Value Loss', color='brown')
        ax6.set_xlabel('Update')
        ax6.set_ylabel('Loss')
        ax6.set_title('Phase 2: PPO Value Loss')
        ax6.grid(True, alpha=0.3)
        ax6.legend()
        
        # Entropy
        ax7 = plt.subplot(3, 3, 7)
        ax7.plot(ppo_trainer.entropies, label='Entropy', color='teal')
        ax7.set_xlabel('Update')
        ax7.set_ylabel('Entropy')
        ax7.set_title('Phase 2: Action Entropy')
        ax7.grid(True, alpha=0.3)
        ax7.legend()
        
        # Combined reward trend
        ax8 = plt.subplot(3, 3, 8)
        if len(rewards) > 100:
            # Plot by iteration (average of multiple episodes)
            iter_rewards = []
            for i in range(0, len(rewards), 4):  # 4 episodes per iteration
                iter_rewards.append(sum(rewards[i:i+4]) / len(rewards[i:i+4]))
            ax8.plot(iter_rewards, label='Avg Reward per Iteration', 
                    color='darkgreen', linewidth=2)
        else:
            ax8.plot(rewards, label='Episode Reward', color='darkgreen')
        ax8.set_xlabel('Iteration / Episode')
        ax8.set_ylabel('Reward')
        ax8.set_title('Phase 2: Learning Progress')
        ax8.grid(True, alpha=0.3)
        ax8.legend()
        
        # Summary statistics
        ax9 = plt.subplot(3, 3, 9)
        ax9.axis('off')
        # Calculate moving averages for rewards/lengths safely
        avg_reward = sum(rewards[-50:]) / len(rewards[-50:]) if rewards else 0.0
        avg_length = sum(lengths[-50:]) / len(lengths[-50:]) if lengths else 0

        summary_text = f"""
            TRAINING SUMMARY
            
            Phase 1: Imitation Learning
            Epochs: {len(agent.train_losses) if agent.train_losses else 0}
            Final Loss: {f"{agent.train_losses[-1]:.4f}" if agent.train_losses else 'N/A'}
            Best Val Acc: {f"{max(agent.val_accuracies):.1f}%" if agent.val_accuracies else 'N/A'}
            
            Phase 2: PPO Fine-tuning
            Episodes: {len(ppo_trainer.episode_rewards)}
            Avg Reward: {avg_reward:.2f}
            Avg Length: {avg_length:.0f}
            Final Entropy: {f"{ppo_trainer.entropies[-1]:.4f}" if ppo_trainer.entropies else 'N/A'}
            """
        ax9.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
                verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/complete_training_curves.png', dpi=150, bbox_inches='tight')
    print(f"Training curves saved to {save_dir}/complete_training_curves.png")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Train lawnmower agent with IL + PPO')
    parser.add_argument('--map', default='maps/maze6.txt', help='Map file')
    parser.add_argument('--skip-il', action='store_true', 
                       help='Skip imitation learning (load pretrained)')
    parser.add_argument('--skip-ppo', action='store_true',
                       help='Skip PPO fine-tuning (IL only)')
    
    # Imitation learning args
    parser.add_argument('--il-demos', type=int, default=3,
                       help='Demonstrations per algorithm for IL')
    parser.add_argument('--il-epochs', type=int, default=50,
                       help='Epochs for imitation learning')
    parser.add_argument('--il-batch-size', type=int, default=64,
                       help='Batch size for IL')
    
    # PPO args
    parser.add_argument('--ppo-iterations', type=int, default=1000,
                       help='PPO training iterations')
    parser.add_argument('--ppo-trajectories', type=int, default=4,
                       help='Trajectories per PPO iteration')
    parser.add_argument('--ppo-epochs', type=int, default=4,
                       help='PPO epochs per update')
    parser.add_argument('--ppo-lr', type=float, default=3e-4,
                       help='PPO learning rate')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='Discount factor')
    parser.add_argument('--clip-epsilon', type=float, default=0.2,
                       help='PPO clipping parameter')
    
    args = parser.parse_args()
    
    # Load environment
    print("="*80)
    print("LOADING ENVIRONMENT")
    print("="*80)
    env = LawnmowerEnvironment(args.map)
    print(f"Map: {args.map}")
    print(f"Grid: {env.width}x{env.height}")
    print(f"Reachable cells: {len(env.reachable_lawn_cells)}")
    print(f"Valid positions: {len(env.lawn_cells)}")
    print()
    
    # Create agent
    print("="*80)
    print("CREATING AGENT")
    print("="*80)
    agent = LawnmowerRLAgent(env, learning_rate=1e-3)
    print(f"State features: {agent.encoder.get_feature_size()}")
    print(f"Action space: 8 directions")
    print(f"Policy network parameters: ~{sum(p.numel() for p in agent.policy.parameters()):,}")
    print()
    
    # ===========================================================================
    # PHASE 1: IMITATION LEARNING
    # ===========================================================================
    
    if not args.skip_il:
        print("="*80)
        print("PHASE 1: IMITATION LEARNING")
        print("="*80)
        print(f"Collecting demonstrations from: hybrid, ccpp, sweepopt")
        print(f"Demonstrations per algorithm: {args.il_demos}")
        print()
        
        # Collect or load demonstrations
        demo_file = 'data/expert_demonstrations.pkl'
        if os.path.exists(demo_file):
            print(f"Loading existing demonstrations from {demo_file}")
            with open(demo_file, 'rb') as f:
                demonstrations = pickle.load(f)
            print(f"Loaded {len(demonstrations)} demonstrations")
        else:
            print("Collecting new demonstrations...")
            algorithms = ['hybrid', 'ccpp', 'sweepopt']
            demonstrations = agent.collect_demonstrations(
                algorithms=algorithms,
                num_demos_per_algo=args.il_demos
            )
            
            # Save demonstrations
            os.makedirs('data', exist_ok=True)
            with open(demo_file, 'wb') as f:
                pickle.dump(demonstrations, f)
            print(f"Saved demonstrations to {demo_file}")
        
        print()
        print(f"Training with imitation learning...")
        print(f"  Epochs: {args.il_epochs}")
        print(f"  Batch size: {args.il_batch_size}")
        print()
        
        # Train with imitation learning
        agent.pretrain_from_demonstrations(
            demonstrations=demonstrations,
            batch_size=args.il_batch_size,
            num_epochs=args.il_epochs,
            validation_split=0.2
        )
        
        # Save IL model
        agent.save_model('models/pretrained_il_model.pt')
        print("\nPhase 1 complete! Model saved to models/pretrained_il_model.pt")
    
    else:
        print("="*80)
        print("SKIPPING PHASE 1: Loading pretrained model")
        print("="*80)
        agent.load_model('models/pretrained_il_model.pt')
        print("Loaded model from models/pretrained_il_model.pt")
    
    # Evaluate after IL
    print("\n" + "="*80)
    print("EVALUATING AFTER IMITATION LEARNING")
    print("="*80)
    il_results, _ = agent.evaluate(num_episodes=5)
    print(f"IL Results:")
    print(f"  Coverage: {il_results['avg_coverage']:.1f}%")
    print(f"  Steps: {il_results['avg_steps']:.0f}")
    print(f"  Efficiency: {il_results['avg_efficiency']:.1f}%")
    print(f"  Direction Changes: {il_results['avg_direction_changes']:.0f}")
    print()
    
    # ===========================================================================
    # PHASE 2: PPO FINE-TUNING
    # ===========================================================================
    
    ppo_trainer = None
    
    if not args.skip_ppo:
        print("="*80)
        print("PHASE 2: PPO FINE-TUNING")
        print("="*80)
        print(f"Starting PPO fine-tuning from IL baseline...")
        print(f"  Iterations: {args.ppo_iterations}")
        print(f"  Trajectories per iteration: {args.ppo_trajectories}")
        print(f"  PPO epochs per update: {args.ppo_epochs}")
        print(f"  Learning rate: {args.ppo_lr}")
        print(f"  Discount factor (gamma): {args.gamma}")
        print(f"  Clip epsilon: {args.clip_epsilon}")
        print()
        
        # Create PPO trainer
        ppo_trainer = PPOTrainer(
            agent=agent,
            env=env,
            learning_rate=args.ppo_lr,
            gamma=args.gamma,
            clip_epsilon=args.clip_epsilon
        )
        
        # Train with PPO
        ppo_trainer.train(
            num_iterations=args.ppo_iterations,
            trajectories_per_iteration=args.ppo_trajectories,
            epochs_per_update=args.ppo_epochs,
            eval_frequency=50
        )
        
        print("\nPhase 2 complete! Model saved to models/best_ppo_model.pt")
        
        # Final evaluation
        print("\n" + "="*80)
        print("FINAL EVALUATION AFTER PPO")
        print("="*80)
        ppo_results, _ = agent.evaluate(num_episodes=10)
        print(f"PPO Results:")
        print(f"  Coverage: {ppo_results['avg_coverage']:.1f}%")
        print(f"  Steps: {ppo_results['avg_steps']:.0f}")
        print(f"  Efficiency: {ppo_results['avg_efficiency']:.1f}%")
        print(f"  Direction Changes: {ppo_results['avg_direction_changes']:.0f}")
        print()
        
        # Improvement comparison
        print("="*80)
        print("IMPROVEMENT FROM IL TO PPO")
        print("="*80)
        print(f"  Coverage: {il_results['avg_coverage']:.1f}% → "
              f"{ppo_results['avg_coverage']:.1f}% "
              f"({'+'if ppo_results['avg_coverage'] >= il_results['avg_coverage'] else ''}"
              f"{ppo_results['avg_coverage'] - il_results['avg_coverage']:.1f}%)")
        print(f"  Steps: {il_results['avg_steps']:.0f} → "
              f"{ppo_results['avg_steps']:.0f} "
              f"({'-' if ppo_results['avg_steps'] <= il_results['avg_steps'] else '+'}"
              f"{abs(ppo_results['avg_steps'] - il_results['avg_steps']):.0f})")
        print(f"  Efficiency: {il_results['avg_efficiency']:.1f}% → "
              f"{ppo_results['avg_efficiency']:.1f}% "
              f"({'+'if ppo_results['avg_efficiency'] >= il_results['avg_efficiency'] else ''}"
              f"{ppo_results['avg_efficiency'] - il_results['avg_efficiency']:.1f}%)")
        print("="*80)
    
    # ===========================================================================
    # VISUALIZATION
    # ===========================================================================
    
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    os.makedirs('plots', exist_ok=True)
    plot_training_results(agent, ppo_trainer)
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"Models saved:")
    print(f"  - models/pretrained_il_model.pt (after Phase 1)")
    if ppo_trainer:
        print(f"  - models/best_ppo_model.pt (best during Phase 2)")
        print(f"  - models/final_ppo_model.pt (final after Phase 2)")
    print(f"\nVisualizations saved:")
    print(f"  - plots/complete_training_curves.png")
    print("="*80)


if __name__ == "__main__":
    main()
