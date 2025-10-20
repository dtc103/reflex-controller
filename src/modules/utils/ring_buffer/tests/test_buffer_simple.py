import unittest
import torch
from ring_buffer import FiFoRingBufferSimple

class TestFiFoRingBufferSimple(unittest.TestCase):
    num_environments = 5
    buffer_capacity = 3
    num_items = 5
    def test_append(self):
        buffer = FiFoRingBufferSimple(self.num_environments, self.num_items, self.buffer_capacity, "cpu")

        # test appending one-dimensional element and type(env_idx) = int
        item_to_add = torch.ones(5)
        buffer.append(0, item_to_add)

        self.assertTrue(torch.all(buffer.first(0) == item_to_add))

        # test appending multi-dimensional element and type(env_idx) = list
        items_to_add = torch.rand(2, 5)
        buffer.append([1, 3], items_to_add)

        self.assertTrue(torch.all(buffer.first(1) == items_to_add[0]) and torch.all(buffer.first(3) == items_to_add[1]))
        
        # test appending one-dimensional element and type(env_idx) = list
        item_to_add = torch.full((5,), 10.0)
        buffer.append([2, 4], item_to_add)

        self.assertTrue(torch.all(buffer.first(2) == item_to_add) and torch.all(buffer.first(4) == item_to_add))

    def test_overflowing(self):
        buffer = FiFoRingBufferSimple(self.num_environments, self.num_items, self.buffer_capacity, "cpu")

        # add more elements than the buffer length
        for i in range(12):
            buffer.append(0, torch.full((5,), float(i)))

        self.assertTrue(torch.all(buffer.first(0) == torch.full((5,), 9.0)))

    def test_wrong_item_dimensions(self):
        buffer = FiFoRingBufferSimple(self.num_environments, self.num_items, self.buffer_capacity, "cpu")

        # add multidimensional element, altough 1 dim is expected
        item_to_add = torch.rand(2, 5)

        with self.assertRaises(RuntimeError):
            buffer.append(0, item_to_add)

        # this should work fine
        buffer.append([0, 1], item_to_add)

        self.assertTrue(torch.all(buffer.first([0, 1]) == item_to_add))

    def test_for_nan(self):
        buffer = FiFoRingBufferSimple(self.num_environments, self.num_items, self.buffer_capacity, "cpu")

        self.assertTrue(torch.all(torch.isnan(buffer.first([0, 1, 2, 3, 4]))))

        item_to_add = torch.ones(5)
        buffer.append(0, item_to_add)

        self.assertFalse(torch.all(torch.isnan(buffer.first([0, 1, 2, 3, 4]))))
        
    def test_pop(self):
        buffer = FiFoRingBufferSimple(self.num_environments, self.num_items, self.buffer_capacity, "cpu")

        for i in range(5):
            buffer.append([0, 1, 2, 3, 4], torch.full((5,), float(i)))

        # elements shouldn't be empty here
        for i in range(5):
            self.assertFalse(torch.any(torch.isnan(buffer.first(i))))

        buffer.pop([[0, 1, 2, 3, 4]])

        # element of ith env should be empty
        self.assertTrue(torch.all(buffer._storage[[0, 1, 2, 3, 4], :, 2]))

if __name__ == "__main__":
    unittest.main()