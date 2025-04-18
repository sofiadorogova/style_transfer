import unittest
import torch
from algos import EG


class TestEG(unittest.TestCase):
    def setUp(self):
        self.x = torch.tensor([1.0, 2.0], requires_grad=True)
        self.y = torch.tensor([3.0, 4.0], requires_grad=True)

        # argmin_x[agrmax_y(x^2 - y^2)] - solution is 0, 0
        self.problem = lambda: (self.x**2).sum() - (self.y**2).sum()

        self.optimizer = EG([self.x, self.y], lr=0.1)

    def test_initialization(self):
        """Test that optimizer initializes correctly."""
        self.assertEqual(len(self.optimizer.param_groups), 1)
        self.assertEqual(len(self.optimizer.param_groups[0]['params']), 2)
        self.assertEqual(self.optimizer.param_groups[0]['lr'], 0.1)

    def test_zero_grad(self):
        """Test that gradients are properly zeroed."""
        loss = self.problem()
        loss.backward()

        self.assertIsNotNone(self.x.grad)
        self.assertIsNotNone(self.y.grad)

        self.optimizer.zero_grad()

        self.assertTrue(torch.allclose(self.x.grad, torch.tensor([0.0, 0.0])))
        self.assertTrue(torch.allclose(self.y.grad, torch.tensor([0.0, 0.0])))

    def test_multiple_steps(self):
        """Test that multiple steps lead to expected behavior."""

        for _ in range(100):
            self.optimizer.zero_grad()
            loss = self.problem()
            loss.backward()
            self.optimizer.step()

        self.assertTrue(torch.allclose(
            self.x, torch.tensor([0.0, 0.0]), atol=1e-5))
        self.assertTrue(torch.allclose(
            self.y, torch.tensor([0.0, 0.0]), atol=1e-5))

    def test_closure(self):
        """Test that closure works correctly."""
        def closure():
            self.optimizer.zero_grad()
            loss = self.problem()
            loss.backward()
            return loss

        initial_x = self.x.clone().detach()
        initial_y = self.y.clone().detach()

        loss = self.optimizer.step(closure)

        self.assertFalse(torch.allclose(self.x, initial_x))
        self.assertFalse(torch.allclose(self.y, initial_y))

        self.assertIsInstance(loss, torch.Tensor)

    def test_mixed_parameter_types(self):
        """Test with both tensor and parameter inputs"""
        x = torch.nn.Parameter(torch.tensor([1.0]))
        y = torch.tensor([2.0], requires_grad=True)
        optimizer = EG([x, y], lr=0.1)

        def problem():
            return x**2 + y**3

        for _ in range(10):
            optimizer.zero_grad()
            problem().backward()
            optimizer.step()

        self.assertTrue(x.grad is not None)
        self.assertTrue(y.grad is not None)

    def test_zero_learning_rate_raises_error(self):
        """Verify EG raises ValueError when initialized with lr=0"""
        x = torch.tensor([1.0], requires_grad=True)

        with self.assertRaises(ValueError) as context:
            EG([x], lr=0.0)

        # Optional: Check the error message
        self.assertIn("Invalid learning rate", str(context.exception))

    def test_sparse_gradients(self):
        """Test with sparse gradient updates"""
        x = torch.zeros(3, requires_grad=True)
        optimizer = EG([x], lr=0.1)

        def problem():
            return x[0]**2 + x[2]**4

        for _ in range(5):
            optimizer.zero_grad()
            problem().backward()
            optimizer.step()

        # Middle element should remain zero
        self.assertEqual(x[1].item(), 0.0)

    def test_parameter_groups(self):
        """Test different hyperparameters for different groups"""
        x = torch.tensor([1.0], requires_grad=True)
        y = torch.tensor([2.0], requires_grad=True)

        optimizer = EG([
            {'params': [x], 'lr': 0.1},
            {'params': [y], 'lr': 0.01}
        ])

        for _ in range(10):
            optimizer.zero_grad()
            (x**2 + y**2).backward()
            optimizer.step()

        # x should change more than y due to higher LR
        self.assertLess(x.abs().item(), y.abs().item())


    def test_numerical_stability(self):
        """Test with extreme values"""
        x = torch.tensor([1e10], requires_grad=True)
        y = torch.tensor([1e-10], requires_grad=True)
        optimizer = EG([x, y], lr=0.1)

        def problem():
            return (x * y).sum()

        for _ in range(10):
            optimizer.zero_grad()
            problem().backward()
            optimizer.step()
            self.assertFalse(torch.isnan(x).any())
            self.assertFalse(torch.isnan(y).any())


    def test_memory_usage(self):
        """Verify no memory leaks"""
        x = torch.tensor([1.0], requires_grad=True)
        optimizer = EG([x], lr=0.1)

        initial_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

        for _ in range(100):
            optimizer.zero_grad()
            (x**2).backward()
            optimizer.step()

        if torch.cuda.is_available():
            self.assertAlmostEqual(
                torch.cuda.memory_allocated(), initial_mem, delta=1000)


if __name__ == '__main__':
    unittest.main()
