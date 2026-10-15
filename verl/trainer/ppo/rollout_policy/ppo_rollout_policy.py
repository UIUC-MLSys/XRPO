from verl.trainer.ppo.reward import compute_reward
from verl.trainer.ppo.rollout_policy.base_rollout_policy import BaseRolloutPolicy

class PPORolloutPolicy(BaseRolloutPolicy):
    def __init__(self, config, tokenizer, actor_rollout_wg, reward_fn):
        super().__init__(config, tokenizer, actor_rollout_wg, reward_fn)

    def expand(self, gen_batch, **kwargs):
        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
        return gen_batch_output
    
    def compute_reward(self, batch, reward_fn):
        return compute_reward(batch, reward_fn)

    def get_policy_name():
        return "ppo_rollout_policy"