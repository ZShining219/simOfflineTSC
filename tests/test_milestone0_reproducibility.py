import random
import unittest

import numpy as np
import torch

from utils.logger import controlled_random_probe, hash_torch_state_dict


class ReproducibilityTest(unittest.TestCase):
    def test_state_dict_hash_is_order_independent_and_content_sensitive(self):
        first = {'b': torch.tensor([2.0]), 'a': torch.tensor([[1, 3]])}
        reordered = {'a': first['a'].clone(), 'b': first['b'].clone()}
        changed = {'a': torch.tensor([[1, 4]]), 'b': first['b'].clone()}
        self.assertEqual(hash_torch_state_dict(first), hash_torch_state_dict(reordered))
        self.assertNotEqual(hash_torch_state_dict(first), hash_torch_state_dict(changed))

    def test_random_probe_does_not_advance_python_or_numpy_rng(self):
        random.seed(11)
        np.random.seed(11)
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        first = controlled_random_probe()
        self.assertEqual(python_state, random.getstate())
        current_numpy = np.random.get_state()
        self.assertEqual(numpy_state[0], current_numpy[0])
        self.assertTrue(np.array_equal(numpy_state[1], current_numpy[1]))
        self.assertEqual(numpy_state[2:], current_numpy[2:])
        self.assertEqual(first, controlled_random_probe())


if __name__ == '__main__':
    unittest.main()
