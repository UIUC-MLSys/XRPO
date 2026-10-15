import uuid
import torch
import numpy as np
from verl.utils.model import compute_position_id_with_mask
from verl.utils.torch_functional import postprocess_data
from collections import defaultdict
from abc import ABC, abstractmethod


class BaseRolloutPolicy(ABC):
    def __init__(self, config, tokenizer, actor_rollout_wg, reward_fn):
        self.config = config
        self.tokenizer = tokenizer
        self.actor_rollout_wg = actor_rollout_wg
        self.reward_fn = reward_fn
        self.global_step = -1

    @abstractmethod
    def expand(self, gen_batch, batch, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def compute_reward(self, batch, reward_fn, *args, **kwargs):
        """
        Compute the reward from the batch.
        """
        raise NotImplementedError

    @staticmethod
    def get_policy_name() -> str:
        raise NotImplementedError

    def repeat_and_merge_batch(self, 
                               batch, 
                               gen_batch_output,
                               original_batch_size=None):
        """
        Repeat the batch and merge with gen_batch_output.
        """
        # TODO: Need to support variable size of rollouts per prompt
        batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)
        rollout_batch_size = gen_batch_output.batch.batch_size[0]
        assert rollout_batch_size % original_batch_size == 0, f"rollout_batch_size {rollout_batch_size} must be divisible by original_batch_size {original_batch_size}"

        batch = batch.repeat(repeat_times=rollout_batch_size//original_batch_size, interleave=True)
        batch = batch.union(gen_batch_output)
        return batch

    def _unpad_tokens(self, token_ids, pad_token_id):
        """
        Unpad tokens by removing padding tokens entirely.
        
        Args:
            token_ids (torch.Tensor): Token IDs that may contain padding
            pad_token_id (int): Pad token ID
            
        Returns:
            torch.Tensor: Unpadded token IDs containing only valid tokens
        """
        # Find valid (non-padding) tokens using attention mask
        valid_mask = token_ids != pad_token_id
        
        # Extract only the valid tokens
        unpadded_tokens = token_ids[valid_mask]
        
        return unpadded_tokens

    def _postprocess_data(self, unpadded_input_ids, max_length, left_pad=True):
        """
        Get input_ids, attention_mask and position_ids for a batch of input_ids.
        Args:
            unpadded_input_ids: [batch_size, seq_len]
            max_length: int
            left_pad: bool
        Returns:
            input_ids: [batch_size, max_length]
            attention_mask: [batch_size, max_length]
            position_ids: [batch_size, max_length]
        """
        input_ids, attention_mask = postprocess_data(
                    input_ids=unpadded_input_ids,
                    attention_mask=torch.ones_like(unpadded_input_ids),
                    max_length=max_length,
                    pad_token_id=self.tokenizer.pad_token_id,
                    left_pad=left_pad,
                    truncation='error'
                )

        position_ids = compute_position_id_with_mask(attention_mask)
        return input_ids, attention_mask, position_ids
    
    def get_metrics(self, batch):
        """
        Get metrics from the batch.
        """
        scores = batch.batch["token_level_scores"].sum(-1).numpy()
        index = batch.non_tensor_batch["uid"]

        # Metrics: Effective Batch size
        effective_batch_size = 0
        all_rollouts_incomplete = 0
        all_rollouts_correct = 0
        all_rollouts_incorrect = 0
        all_rollouts_incorrect_incomplete = 0
        score_std = []
        id2score = defaultdict(list)
        with torch.no_grad():
            bsz = scores.shape[0]
            for i in range(bsz):
                id2score[index[i]].append(scores[i])
            for idx in id2score:
                scores = np.array(id2score[idx])
                score_std.append(np.std(scores))
                # Atleast one correct and wrong sample
                if any(scores != 1) and any(scores == 1):
                    effective_batch_size += 1
                if all(scores == 1):
                    all_rollouts_correct += 1
                if all(scores == 0):
                    all_rollouts_incomplete += 1
                if all(scores == 0.1):
                    all_rollouts_incorrect += 1
                if all(scores != 1):
                    all_rollouts_incorrect_incomplete+=1

        return {
            "additional_metrics/effective_batch_size": effective_batch_size,
            "additional_metrics/all_rollouts_correct": all_rollouts_correct,
            "additional_metrics/all_rollouts_incorrect": all_rollouts_incorrect,
            "additional_metrics/all_rollouts_incomplete": all_rollouts_incomplete,
            "additional_metrics/all_rollouts_incorrect_incomplete": all_rollouts_incorrect_incomplete,
            "critic/score/std": np.mean(score_std)
        }

    def update_global_step(self, global_step):
        self.global_step = global_step

    def load_checkpoint(self, global_step_folder, global_step):
        pass

    def postprocess_batch(self, batch):
        return batch