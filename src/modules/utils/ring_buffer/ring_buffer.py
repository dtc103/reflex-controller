import torch


class FiFoRingBufferSimple:
    def __init__(self, num_envs, num_items, capacity, device):
        self.device = device
        self.num_items = int(num_items)
        self.num_envs = int(num_envs)

        # At first we will keep the capacity same (basically the delay froe ach muscle the same)
        self._capacity = int(capacity)

        self._storage = torch.full((self.num_envs, self.num_items, self._capacity), torch.nan, device=self.device)

        #since for the muscles later the all the info for the msucels are added in parallel and also 
        #read out in parallel, we only need 1 head for all of the muscles. 
        #This only allows for constant capacity size and has to be changed for variable capacity
        self._head = torch.zeros(self.num_envs, device=self.device).int()
        self._size = torch.zeros(self.num_envs, device=self.device).int()

    def append(self, env_ids, items):
        self._storage[env_ids, : ,self._head[env_ids]] = items
        self._head[env_ids] = (self._head[env_ids] + 1) % self._capacity
        self._size[env_ids] = torch.clamp(self._size[env_ids] + 1, max = self._capacity)

    def first(self, env_ids):
        h_idx = (self._head[env_ids] - self._size[env_ids]) % self._capacity
        return self._storage[env_ids, :, h_idx]

    def pop(self, env_ids):
        if torch.any(self._size[env_ids] <= 0):
            raise RuntimeError(f"Tried to pop from an empty array")

        h_idx = (self._head[env_ids] - self._size[env_ids]) % self._capacity
        res = self._storage[env_ids, :, h_idx].clone()
        
        self._storage[env_ids, :, h_idx] = torch.nan

        self._size = self._size[env_ids] - 1

        return res

    def size(self):
        return self._size[:]

    def clear(self):
        self._head.zero_()
        self._size.zero_()
        self._storage.fill_(torch.nan)

class FiFoRingBufferVariableHeads:
    def __init__(self, num_envs, num_items, capacity, device):
        self.device = device
        self.num_items = int(num_items)
        self.num_envs = int(num_envs)

        # At first we will keep the capacity same (basically the delay froe ach muscle the same)
        self._capacity = int(capacity)

        self._storage = torch.full((self.num_envs, self.num_items, self._capacity), torch.nan, device=self.device)
        self._head = torch.zeros(self.num_envs, self.num_items, device=self.device).int()
        self._size = torch.zeros(self.num_envs, self.num_items, device=self.device).int()

    def append(self, env_ids, items):
        """Always append to all environments and also always first and pop from every environment
        """
        items = items.unsqueeze(-1)
        env_sel = self._storage[env_ids]
        selections = self._head[env_ids].unsqueeze(-1)
        env_sel.scatter_(dim=2, index=selections, src=items)
        self._storage[env_ids] = env_sel


        self._storage[env_ids, :, self._head[env_ids]] = items
        self._head[env_ids] = (self._head[env_ids] + 1) % self._capacity
        self._size[env_ids] = torch.clamp(self._size[env_ids] + 1, max=self._capacity)

    ##### CONTINUE IMPLEMENTING FROM HERE

    def first(self, env_ids):
        """Returns a view to the first element in the fifo, without removing it"""
        h_idxs = (self._head[env_ids] - self._size[env_ids]) % self._capacity[env_ids]
        return self._storage[env_ids, h_idxs]
    
    def pop(self, env_ids):
        """Returns the first element, while also removing it from the fifo"""

        h_idxs = (self._head[env_ids] - self._size[env_ids]) % self._capacity[env_ids]

        result = self._storage[env_ids, h_idxs].clone()
        self._storage[env_ids, h_idxs] = torch.nan
        
        # we can always do -1 here, since we already checked before if a buffer is empty
        self._size[env_ids] = self._size[env_ids] - 1

        return result
    

    def size(self):
        return self._size[:]
    
    def clear(self):
        self._head.zero_()
        self._size.zero_()
        self._storage.zero_()