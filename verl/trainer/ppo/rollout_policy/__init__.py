from verl.trainer.ppo.rollout_policy.ppo_rollout_policy import PPORolloutPolicy
from verl.trainer.ppo.rollout_policy.explore_exploit_rollout_policy import ExploreExploitRolloutPolicy

registered_rollout_policies = [PPORolloutPolicy, ExploreExploitRolloutPolicy]

def get_rollout_policy(config, tokenizer, actor_rollout_wg, reward_fn):
    for rollout_policy in registered_rollout_policies:
        if rollout_policy.get_policy_name() == config.rollout_policy.policy_name:
            return rollout_policy(config, tokenizer, actor_rollout_wg, reward_fn)
    raise ValueError(f"Invalid rollout policy name: {config.rollout_policy.policy_name}")
