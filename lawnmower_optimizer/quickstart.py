#!/usr/bin/env python3
"""
Quick Start Script for Lawnmower RL System

This script provides an easy way to train and test the RL agent.
"""

import os
import sys
import argparse


def train(args):
    """Run training"""
    print("="*80)
    print("TRAINING LAWNMOWER RL AGENT")
    print("="*80)
    
    if args.ppo:
        # Train with IL + PPO
        print("Mode: Imitation Learning + PPO Fine-tuning")
        from train_with_ppo import main as train_ppo_main
        import sys
        sys.argv = ['train_with_ppo.py']
        if args.quick:
            sys.argv.extend(['--ppo-iterations', '100'])
        train_ppo_main()
    else:
        # Train with IL only
        print("Mode: Imitation Learning only")
        from train_rl_agent import main as train_main
        train_main()


def test(args):
    """Run testing"""
    print("="*80)
    print("TESTING LAWNMOWER RL AGENT")
    print("="*80)
    
    # Import and run testing
    from test_rl_agent import main as test_main
    test_main()


def demo(args):
    """Run a quick demo"""
    print("="*80)
    print("QUICK DEMO - Single Episode")
    print("="*80)
    
    import torch
    from train_rl_agent import LawnmowerRLAgent, ActionMapper
    from utils.environment import LawnmowerEnvironment
    
    # Load environment
    env = LawnmowerEnvironment(args.map)
    
    # Load agent
    agent = LawnmowerRLAgent(env)
    
    model_path = 'models/best_pretrained_model.pt'
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        print("Please run 'python quickstart.py train' first!")
        return
    
    agent.load_model(model_path)
    print(f"Loaded model from {model_path}")
    
    # Run episode
    print("\nRunning episode...")
    current_pos = env.start_pos
    mowed_cells = set()
    path = [current_pos]
    last_action = None
    direction_changes = 0
    
    for step in range(args.max_steps):
        # Update mowed cells
        blade = env.get_blade_coverage(current_pos)
        for cell in blade:
            if cell in env.reachable_lawn_cells:
                mowed_cells.add(cell)
        
        # Check if done
        coverage = len(mowed_cells & env.reachable_lawn_cells)
        if coverage >= len(env.reachable_lawn_cells):
            print(f"\n✓ 100% coverage achieved in {len(path)} steps!")
            break
        
        # Progress update
        if step % 100 == 0:
            coverage_pct = 100 * coverage / len(env.reachable_lawn_cells)
            print(f"  Step {step}: Coverage {coverage_pct:.1f}%")
        
        # Get action
        state = agent.encoder.encode_state(current_pos, mowed_cells, last_action)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            action = agent.policy.get_action(state_tensor, deterministic=True).item()
        
        # Execute action
        dx, dy = ActionMapper.action_to_delta(action)
        next_pos = (current_pos[0] + dx, current_pos[1] + dy)
        
        # Check valid
        if next_pos not in env.lawn_cells:
            neighbors = env.get_neighbors(current_pos)
            if neighbors:
                next_pos = list(neighbors)[0]
            else:
                break
        
        # Count direction change
        if last_action is not None and action != last_action:
            direction_changes += 1
        
        path.append(next_pos)
        current_pos = next_pos
        last_action = action
    
    # Final statistics
    final_coverage = len(mowed_cells & env.reachable_lawn_cells)
    coverage_pct = 100 * final_coverage / len(env.reachable_lawn_cells)
    efficiency = 100 * final_coverage / len(path) if len(path) > 0 else 0
    
    print("\n" + "="*80)
    print("RESULTS:")
    print(f"  Coverage: {coverage_pct:.1f}% ({final_coverage}/{len(env.reachable_lawn_cells)} cells)")
    print(f"  Steps: {len(path)}")
    print(f"  Efficiency: {efficiency:.1f}%")
    print(f"  Direction Changes: {direction_changes}")
    print("="*80)
    
    # Optionally save visualization
    if args.visualize:
        print("\nCreating visualization...")
        from test_rl_agent import visualize_path
        os.makedirs('plots', exist_ok=True)
        visualize_path(env, path, 
                      title=f"RL Agent Demo - Coverage: {coverage_pct:.1f}%",
                      save_path='plots/demo_path.png')
        print("Saved to plots/demo_path.png")


def collect_demos(args):
    """Collect and save demonstrations only"""
    print("="*80)
    print("COLLECTING EXPERT DEMONSTRATIONS")
    print("="*80)
    
    import pickle
    from train_rl_agent import LawnmowerRLAgent
    from utils.environment import LawnmowerEnvironment
    
    env = LawnmowerEnvironment(args.map)
    agent = LawnmowerRLAgent(env)
    
    # Collect demonstrations
    algorithms = ['hybrid', 'ccpp', 'sweepopt']
    demonstrations = agent.collect_demonstrations(
        algorithms=algorithms,
        num_demos_per_algo=args.num_demos
    )
    
    # Save
    os.makedirs('data', exist_ok=True)
    with open('data/expert_demonstrations.pkl', 'wb') as f:
        pickle.dump(demonstrations, f)
    
    print(f"\n✓ Saved {len(demonstrations)} demonstrations to data/expert_demonstrations.pkl")


def main():
    parser = argparse.ArgumentParser(
        description='Lawnmower RL System - Quick Start',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with Imitation Learning only (fast)
  python quickstart.py train
  
  # Train with IL + PPO fine-tuning (best performance)
  python quickstart.py train --ppo
  
  # Quick training mode (for testing)
  python quickstart.py train --ppo --quick
  
  # Test the agent
  python quickstart.py test
  
  # Run a quick demo
  python quickstart.py demo --visualize
  
  # Collect demonstrations only
  python quickstart.py collect --num-demos 5
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the RL agent')
    train_parser.add_argument('--ppo', action='store_true',
                             help='Include PPO fine-tuning (default: IL only)')
    train_parser.add_argument('--quick', action='store_true',
                             help='Quick training mode (fewer iterations)')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Test the RL agent')
    
    # Demo command
    demo_parser = subparsers.add_parser('demo', help='Run a quick demo episode')
    demo_parser.add_argument('--map', default='maps/maze6.txt', help='Map file')
    demo_parser.add_argument('--max-steps', type=int, default=2000, help='Maximum steps')
    demo_parser.add_argument('--visualize', action='store_true', help='Save visualization')
    
    # Collect command
    collect_parser = subparsers.add_parser('collect', help='Collect demonstrations only')
    collect_parser.add_argument('--map', default='maps/maze6.txt', help='Map file')
    collect_parser.add_argument('--num-demos', type=int, default=3, help='Demos per algorithm')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train(args)
    elif args.command == 'test':
        test(args)
    elif args.command == 'demo':
        demo(args)
    elif args.command == 'collect':
        collect_demos(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
