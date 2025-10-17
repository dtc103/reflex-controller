import unittest
import torch
from ring_buffer import FiFoRingBuffer

class TestFiFoRingBuffer(unittest.TestCase):
    num_environments = 5
    buffer_capacity = 3
    itemlength = 5
    def test_append(self):
        buffer = FiFoRingBuffer(self.num_environments, self.buffer_capacity, self.itemlength, "cpu")

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
        buffer = FiFoRingBuffer(self.num_environments, self.buffer_capacity, self.itemlength, "cpu")

        # add more elements than the buffer length
        for i in range(12):
            buffer.append(0, torch.full((5,), float(i)))

        self.assertTrue(torch.all(buffer.first(0) == torch.full((5,), 9.0)))

    def test_wrong_item_dimensions(self):
        buffer = FiFoRingBuffer(self.num_environments, self.buffer_capacity, self.itemlength, "cpu")

        # add multidimensional element, altough 1 dim is expected
        item_to_add = torch.ones(2, 5)
        with self.assertRaises(RuntimeError):
            buffer.append(0, item_to_add)

        # this should work fine
        buffer.append([0, 1], item_to_add)

    def test_for_nan(self):
        buffer = FiFoRingBuffer(self.num_environments, self.buffer_capacity, self.itemlength, "cpu")

        self.assertTrue(torch.all(torch.isnan(buffer.first([0, 1, 2, 3, 4]))))

        item_to_add = torch.ones(5)
        buffer.append(0, item_to_add)

        self.assertFalse(torch.all(torch.isnan(buffer.first([0, 1, 2, 3, 4]))))

    def test_variable_capacity(self):
        buffer = FiFoRingBuffer(self.num_environments, [1, 2, 3, 4, 5], self.itemlength, "cpu")

        for i in range(5):
            buffer.append([0, 1, 2, 3, 4], torch.full((5,), float(i)))

        self.assertTrue(torch.all(buffer.first(0) == torch.full((5,), 4.0)))
        self.assertTrue(torch.all(buffer.first(1) == torch.full((5,), 3.0)))
        self.assertTrue(torch.all(buffer.first(2) == torch.full((5,), 2.0)))
        self.assertTrue(torch.all(buffer.first(3) == torch.full((5,), 1.0)))
        self.assertTrue(torch.all(buffer.first(4) == torch.full((5,), 0.0)))

        
    def test_capacity_exceptions(self):
        with self.assertRaises(RuntimeError):
            FiFoRingBuffer(self.num_environments, -2, self.itemlength, "cpu")
        
        with self.assertRaises(RuntimeError):
            FiFoRingBuffer(self.num_environments, [1, 2], self.itemlength, "cpu")

        with self.assertRaises(RuntimeError):
            FiFoRingBuffer(self.num_environments, [-2, 1, 2, 3, 4], self.itemlength, "cpu")
            
    def test_pop(self):
        buffer = FiFoRingBuffer(self.num_environments, [1, 2, 3, 4, 5], self.itemlength, "cpu")

        for i in range(5):
            buffer.append([0, 1, 2, 3, 4], torch.full((5,), float(i)))

        # elements shouldn't be empty here
        for i in range(5):
            self.assertFalse(torch.all(torch.isnan(buffer.first(i))))

        for i in range(5):
            buffer.pop([range(4, i - 1, -1)])

            # element of ith env should be empty
            self.assertTrue(torch.all(torch.isnan(buffer.first(i))))

if __name__ == "__main__":
    unittest.main()