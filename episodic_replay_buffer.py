import os
import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, List
import shutil
import numba
from numba import jit


@jit(nopython=True)
def st_update(tree: np.ndarray, capacity: int, idx: int, value: float):
    i = idx + capacity
    tree[i] = value
    i //= 2
    while i >= 1:
        tree[i] = tree[2 * i] + tree[2 * i + 1]
        i //= 2


@jit(nopython=True)
def st_update_many(tree: np.ndarray, capacity: int, idxs: np.ndarray, vals: np.ndarray):
    n = idxs.shape[0]
    for k in range(n):
        i = idxs[k] + capacity
        tree[i] = vals[k]
        i //= 2
        while i >= 1:
            tree[i] = tree[2 * i] + tree[2 * i + 1]
            i //= 2


@jit(nopython=True)
def st_find_prefix(tree: np.ndarray, capacity: int, s: float) -> int:
    i = 1
    while i < capacity:
        left = 2 * i
        if tree[left] > s:
            i = left
        else:
            s -= tree[left]
            i = left + 1
    return i - capacity


@jit(nopython=True)
def st_sample_indices(tree: np.ndarray, capacity: int, batch_size: int, total: float) -> np.ndarray:
    out = np.empty(batch_size, dtype=np.int32)
    seg = total / batch_size
    for i in range(batch_size):
        a = seg * i
        b = seg * (i + 1)
        s = np.random.uniform(a, b)
        out[i] = st_find_prefix(tree, capacity, s)
    return out


class EpisodicReplayBuffer:
    def __init__(self,
                 seq_len: int,
                 device: torch.device,
                 capacity_sequences: int = 16000,
                 stride: int = None,
                 alpha: float = 0.6,
                 cache_dir: str = "./episode_cache"):
        self.seq_len = seq_len
        self.device = device
        self.alpha = alpha
        self.eps = 1e-6
        self.cache_dir = cache_dir
        
        self.stride = stride or max(1, seq_len // 3)

        os.makedirs(self.cache_dir, exist_ok=True)

        self.ep_files = [] # str list
        self.ep_file_slots = [] # Available file slot numbers

        self.ep_refcnt = []
        self.ep_actions = []  # uint8 (T,)
        self.ep_rewards = []  # float32 (T,)
        self.ep_dones = []    # uint8 (T,)
        self.ep_shapes = []  # (C,H,W)
        self.ep_lengths = []
        self.total_frames = 0
        
        self.hidden_size = 512
        self.size = 0

        # Sequence index
        self.capacity = int(capacity_sequences)
        self.seq_ep = np.zeros(self.capacity, dtype=np.int32)
        self.seq_t  = np.zeros(self.capacity, dtype=np.int32)
        self.seq_h = torch.zeros(self.capacity, 2, self.hidden_size,
                                 dtype=torch.float32, device=self.device)

        self.write_pos = 0

        # Sum-tree for priorities
        tree_cap = 1
        while tree_cap < self.capacity:
            tree_cap *= 2
        self.tree_capacity = tree_cap
        self.sum_tree = np.zeros(2 * tree_cap, dtype=np.float32)
        self.max_priority = 1.0

    def _get_file_slot(self) -> int:
        if self.ep_file_slots:
            return self.ep_file_slots.pop()
        else:
            return len(self.ep_files)

    def _save_episode_to_file(self, ep_id: int, frames: torch.Tensor, 
                              actions: np.ndarray, rewards: np.ndarray, 
                              dones: np.ndarray) -> str:
        if ep_id < len(self.ep_files) and self.ep_files[ep_id] is not None:
            # Reuse existing file
            filepath = self.ep_files[ep_id]
        else:
            # Create new file
            file_slot = self._get_file_slot()
            filename = f"episode_{file_slot:06d}.pt"
            filepath = os.path.join(self.cache_dir, filename)
            
            # Extend ep_files list
            while len(self.ep_files) <= ep_id:
                self.ep_files.append(None)
            self.ep_files[ep_id] = filepath

        episode_data = {
            'frames': frames.cpu() if frames.device.type != 'cpu' else frames,
            'actions': actions.copy(),
            'rewards': rewards.copy(),
            'dones': dones.copy(),
            'shape': tuple(frames.shape[1:]),  # (C, H, W)
            'length': len(frames)
        }

        torch.save(episode_data, filepath)
        return filepath

    def _load_episode_from_file(self, ep_id: int) -> torch.Tensor:
        # Load episode frames from binary file directly to device. Return: (T, C, H, W) tensor with all frames on device

        filepath = self.ep_files[ep_id]
        if filepath is None or not os.path.exists(filepath):
            raise ValueError(f"Episode {ep_id} file not found: {filepath}")

        episode_data = torch.load(filepath, map_location=self.device)
        frames = episode_data['frames']

        if frames.device != self.device:
            frames = frames.to(self.device, non_blocking=True)

        del episode_data
            
        return frames

    def _inc_ref(self, ep_id: int):
        self.ep_refcnt[ep_id] += 1

    def _dec_ref(self, ep_id: int):
        self.ep_refcnt[ep_id] -= 1
        if self.ep_refcnt[ep_id] == 0:
            # Mark file slot as available for reuse but don't delete file yet
            if ep_id < len(self.ep_files) and self.ep_files[ep_id] is not None:
                filepath = self.ep_files[ep_id]
                filename = os.path.basename(filepath)
                if filename.startswith("episode_") and filename.endswith(".pt"):
                    try:
                        file_slot = int(filename[8:14]) # Extract 6-digit slot number
                        self.ep_file_slots.append(file_slot)
                    except ValueError:
                        pass
                
                self.ep_files[ep_id] = None
                
            # Clear metadata
            self.ep_actions[ep_id] = None
            self.ep_rewards[ep_id] = None
            self.ep_dones[ep_id] = None
            self.ep_shapes[ep_id] = None

    def __len__(self):
        return self.size

    def _write_slot(self, slot_idx: int, ep_id: int, t_start: int, hid: Tuple):
        if self.size == self.capacity:
            old_ep = int(self.seq_ep[slot_idx])
            self._dec_ref(old_ep)

        self.seq_ep[slot_idx] = ep_id
        self.seq_t[slot_idx] = t_start
        self.seq_h[slot_idx, 0].copy_(hid[0].squeeze())  # (H,)
        self.seq_h[slot_idx, 1].copy_(hid[1].squeeze())
        self._inc_ref(ep_id)

        st_update(self.sum_tree, self.tree_capacity, slot_idx, self.max_priority)

    def push_episode(self, frames_u8: np.ndarray = None, actions_u8: np.ndarray = None,
                     rewards: np.ndarray = None, dones_u8: np.ndarray = None, 
                     hiddens=None):
        # Store episode by saving to binary file
        assert frames_u8 is not None and (frames_u8.dtype == np.uint8 or frames_u8.dtype == torch.uint8)
        T, C, H, W = frames_u8.shape

        # Convert to tensor if needed
        if isinstance(frames_u8, np.ndarray):
            frames_tensor = torch.from_numpy(frames_u8)
        else:
            frames_tensor = frames_u8

        ep_id = len(self.ep_refcnt)

        filepath = self._save_episode_to_file(ep_id, frames_tensor, 
                                              actions_u8, rewards, dones_u8)
        
        # Store episode metadata
        self.ep_refcnt.append(0)
        self.ep_actions.append(actions_u8.astype(np.uint8, copy=True))
        self.ep_rewards.append(rewards.astype(np.float32, copy=True))
        self.ep_dones.append(dones_u8.astype(np.uint8, copy=True))
        self.ep_shapes.append((C, H, W))
        self.ep_lengths.append(T)
        self.total_frames += T

        # Build sequence indices
        max_start = T - self.seq_len
        if max_start >= 0:
            starts = np.arange(0, max_start + 1, self.stride)
            starts += np.random.randint(0, self.stride, size=starts.shape)
            starts = np.clip(starts, None, max_start)
            for t in starts:
                t = min(max_start, t)
                self._write_slot(self.write_pos, ep_id, t, hiddens[t])
                self.write_pos = (self.write_pos + 1) % self.capacity
                if self.size < self.capacity:
                    self.size += 1
            if starts[-1] < max_start:
                self._write_slot(self.write_pos, ep_id, max_start, hiddens[max_start])
                self.write_pos = (self.write_pos + 1) % self.capacity
                if self.size < self.capacity:
                    self.size += 1

    def get_storage_stats(self):
        # Return storage statistics
        total_files = 0
        total_size_mb = 0
        active_episodes = 0
        
        for ep_id in range(len(self.ep_files)):
            if self.ep_files[ep_id] is not None and os.path.exists(self.ep_files[ep_id]):
                total_files += 1
                file_size = os.path.getsize(self.ep_files[ep_id])
                total_size_mb += file_size / (1024**2)
                
                if ep_id < len(self.ep_refcnt) and self.ep_refcnt[ep_id] > 0:
                    active_episodes += 1
        
        return {
            "total_files": total_files,
            "total_size_mb": total_size_mb,
            "active_episodes": active_episodes,
            "available_file_slots": len(self.ep_file_slots),
            "total_frames": self.total_frames
        }

    def sample(self, batch_size: int, beta: float = 0.4, out=None):
        """
        Sample sequences by loading episodes directly from files on-demand. Arguments:
            batch_size: Number of sequences to sample
            beta: Importance sampling exponent
            out: Optional dict with pre-allocated buffers for the output: {
                'states': (B, S, C, H, W) tensor,
                'next_states': (B, S, C, H, W) tensor, 
                'actions': (B, S) tensor,
                'rewards': (B, S) tensor,
                'dones': (B, S) tensor,
                'hidden': (B, 2, H) tensor,
                'weights': (B,) tensor
            }
        """
        # Draw indices
        total = np.float32(self.sum_tree[1])
        if total <= 0.0 or self.size == 0:
            idxs = np.random.randint(0, max(1, self.size), size=batch_size, dtype=np.int32)
            pri = np.ones_like(idxs, dtype=np.float32)
            total = np.float32(pri.sum())
        else:
            idxs = st_sample_indices(self.sum_tree, self.tree_capacity, batch_size, total)
            np.clip(idxs, 0, self.size - 1, out=idxs)

        # Compute weights
        pri = self.sum_tree[idxs + self.tree_capacity].astype(np.float32)
        probs = np.clip(pri / total, np.float32(1e-12), np.float32(1.0)).astype(np.float32, copy=False)
        weights = np.power(self.size * probs, -np.float32(beta), dtype=np.float32)
        wmax = weights.max()
        if wmax > 0:
            weights /= wmax
        weights = weights.astype(np.float32, copy=False)

        # Get episodes needed
        episode_ids = [int(self.seq_ep[idx]) for idx in idxs]

        # Use pre-allocated buffers if provided, otherwise allocate new ones
        if out is not None:
            states_f = out['states']
            next_states_f = out['next_states']
            actions_t = out['actions']
            rewards_t = out['rewards']
            dones_t = out['dones']
            hidden0 = out['hidden']
            weights_t = out['weights']
            weights_t.copy_(torch.from_numpy(weights))
            hidden0.copy_(self.seq_h[idxs])
        else:
            first_ep = episode_ids[0]
            C, H, W = self.ep_shapes[first_ep]
            S = self.seq_len
            dev = self.device
            
            states_f = torch.empty((batch_size, S, C, H, W), dtype=torch.float32, device=dev)
            next_states_f = torch.empty_like(states_f)
            actions_t = torch.empty((batch_size, S), dtype=torch.int64, device=dev)
            rewards_t = torch.empty((batch_size, S), dtype=torch.float32, device=dev)
            dones_t = torch.empty((batch_size, S), dtype=torch.float32, device=dev)
            weights_t = torch.from_numpy(weights).to(dev, dtype=torch.float32, non_blocking=True)
            hidden0 = self.seq_h[idxs]  # (B, 2, H)

        S = self.seq_len
        
        # Process each sequence individually to minimize memory usage
        for i in range(batch_size):
            ep = episode_ids[i]
            t0 = int(self.seq_t[idxs[i]])
            T = self.ep_lengths[ep]
            t1 = t0 + S

            # Load episode from file directly to device
            episode_frames = self._load_episode_from_file(ep)
            
            # Extract sequence
            states_u8 = episode_frames[t0:t1]
            states_f[i].copy_(states_u8.to(torch.float32).div_(255.0))
            
            # Next states
            next_states_u8 = episode_frames[t0+1:t1+1]
            if t1 >= T:
                # Handle episode boundary
                next_states_u8 = torch.cat([episode_frames[t0+1:T], episode_frames[T-1:T]], dim=0)
            next_states_f[i].copy_(next_states_u8.to(torch.float32).div_(255.0))

            del episode_frames, states_u8, next_states_u8

            # Copy actions, rewards, dones from metadata
            a_np = self.ep_actions[ep][t0:t1].astype(np.int64, copy=False)
            r_np = self.ep_rewards[ep][t0:t1].astype(np.float32, copy=False)
            d_np = self.ep_dones[ep][t0:t1].astype(np.float32, copy=False)
            actions_t[i].copy_(torch.from_numpy(a_np))
            rewards_t[i].copy_(torch.from_numpy(r_np))
            dones_t[i].copy_(torch.from_numpy(d_np))

        return (states_f, actions_t, rewards_t, next_states_f, dones_t, hidden0, idxs, weights_t)

    def update_priorities(self, idxs: np.ndarray, td_errors: torch.Tensor):
        # Update sum-tree priorities
        te = td_errors.abs()
        if te.dim() == 2:
            pr = 0.9 * te.amax(dim=1) + 0.1 * te.mean(dim=1)
        elif te.dim() == 1:
            pr = te
        else:
            raise ValueError(f"Unexpected td_errors shape {tuple(te.shape)}")
        pr = torch.pow(pr + max(self.eps, 1e-6), self.alpha)
        vals = pr.detach().cpu().numpy().astype(np.float32)
        idxs = idxs.astype(np.int32)
        np.clip(idxs, 0, self.size - 1, out=idxs)
        st_update_many(self.sum_tree, self.tree_capacity, idxs, vals)
        mx = float(vals.max())
        if mx > self.max_priority:
            self.max_priority = mx

    def cleanup_unused_files(self):
        # Remove files for episodes with 0 references
        removed_count = 0
        for ep_id in range(len(self.ep_refcnt)):
            if (self.ep_refcnt[ep_id] == 0 and 
                ep_id < len(self.ep_files) and 
                self.ep_files[ep_id] is not None):
                
                filepath = self.ep_files[ep_id]
                if os.path.exists(filepath):
                    os.remove(filepath)
                    removed_count += 1
                self.ep_files[ep_id] = None
        
        return removed_count

    def cleanup_cache_directory(self):
        # Remove entire cache directory and recreate it
        if os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Clear all file tracking
        self.ep_files.clear()
        self.ep_file_slots.clear()