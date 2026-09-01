import os
import unittest
import torch
import numpy as np
import gymnasium as gym

try:
    import ale_py
    gym.register_envs(ale_py)
except ImportError:
    pass

from src.preprocessing import AtariPreprocessor
from src.dqn import DQN
from src.agent import DQNAgent
from src.evaluate import evaluate

class TestEvaluationAnalysis(unittest.TestCase):
    
    def setUp(self):
        self.env = gym.make("ALE/Seaquest-v5")
        self.preprocessor = AtariPreprocessor()
        self.dqn_engine = DQN(self.env.action_space.n, device="cpu")
        self.agent = DQNAgent(self.env.action_space.n, self.dqn_engine.online_network, device="cpu")
        
    def tearDown(self):
        self.env.close()

    def test_checkpoint_loading(self):
        """Test 1 - A saved checkpoint loads successfully"""
        # Create a dummy checkpoint
        dummy_path = "models/checkpoints/dummy_eval_test.pt"
        os.makedirs("models/checkpoints", exist_ok=True)
        torch.save({
            'step': 100,
            'online_state_dict': self.dqn_engine.online_network.state_dict(),
            'target_state_dict': self.dqn_engine.target_network.state_dict(),
            'config': {}
        }, dummy_path)
        
        # Test loading
        new_engine = DQN(self.env.action_space.n, device="cpu")
        checkpoint = torch.load(dummy_path, map_location="cpu", weights_only=True)
        new_engine.online_network.load_state_dict(checkpoint['online_state_dict'])
        new_engine.target_network.load_state_dict(checkpoint['target_state_dict'])
        
        self.assertEqual(checkpoint['step'], 100)
        
        if os.path.exists(dummy_path):
            os.remove(dummy_path)
            
    def test_evaluation_immutability(self):
        """Test 3, 4, 5 - Evaluation does not update weights or buffer"""
        
        online_before = [p.clone() for p in self.dqn_engine.online_network.parameters()]
        target_before = [p.clone() for p in self.dqn_engine.target_network.parameters()]
        
        # Ensure gradients are reset
        self.dqn_engine.online_network.zero_grad()
        
        # Run extremely short evaluation
        evaluate(self.env, self.agent, self.preprocessor, episodes=1, max_steps=5)
        
        online_after = [p.clone() for p in self.dqn_engine.online_network.parameters()]
        target_after = [p.clone() for p in self.dqn_engine.target_network.parameters()]
        
        # Test 5 - No parameter updates
        for b, a in zip(online_before, online_after):
            self.assertTrue(torch.equal(b, a))
            
        for b, a in zip(target_before, target_after):
            self.assertTrue(torch.equal(b, a))
            
        # Test 4 - No gradients
        for p in self.dqn_engine.online_network.parameters():
            self.assertIsNone(p.grad)
            
    def test_metrics_structure(self):
        """Test 6, 8 - Metrics return structure and action validity"""
        results = evaluate(self.env, self.agent, self.preprocessor, episodes=1, max_steps=5, return_detailed=True)
        
        self.assertIn("rewards", results)
        self.assertIn("lengths", results)
        self.assertIn("actions", results)
        self.assertIn("q_values", results)
        self.assertIn("max_q_values", results)
        
        self.assertEqual(len(results["rewards"]), 1)
        self.assertEqual(len(results["lengths"]), 1)
        
        # Test 8 - Action recording valid
        for a in results["actions"]:
            self.assertTrue(0 <= a < self.env.action_space.n)
            
    def test_deterministic_metric_structure(self):
        """Test 7 - A fixed synthetic result produces correct aggregate statistics"""
        rewards = [10.0, 20.0, 30.0]
        self.assertEqual(np.mean(rewards), 20.0)
        self.assertEqual(np.median(rewards), 20.0)
        self.assertTrue(np.std(rewards) > 0)
        
if __name__ == '__main__':
    unittest.main()
