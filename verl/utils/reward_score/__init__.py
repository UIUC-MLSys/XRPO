# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# from . import gsm8k, math, prime_math, prime_code

from verl.utils.import_utils import deprecated


def default_compute_score(data_source, solution_str, ground_truth, extra_info=None, sandbox_fusion_url=None, concurrent_semaphore=None):
    """Compute the score for a given solution based on the data source.

    Args:
        data_source (str): The source dataset identifier which determines the scoring method.
        solution_str (str): The solution string to be evaluated.
        ground_truth (str): The ground truth answer for comparison.
        extra_info (dict, optional): Additional information that might be needed for scoring. Defaults to None.

    Returns:
        float: The computed score as a floating point number. If the result is a dictionary,
               it returns the dictionary instead.

    Note:
        XRPO replaces upstream verl's per-dataset dispatch with two scorers: the
        code data sources listed below go to `rllm_code`, and **everything else**
        falls through to `latex_math` (boxed-answer extraction + math-verify).
        Upstream's gsm8k / geo3k / search-R1 scorers are therefore no longer
        reachable through this function -- add a branch here if you need them,
        otherwise those data sources would be scored as math and silently
        mis-rewarded.
    """
    # Code
    if data_source in ["taco", "apps", "code_contests", "leetcode", "livecodebench", "codeforces", "primeintellect", "kodcode", "humanevalplus"]:
        from .rllm_code import code_verify
        res = code_verify.compute_score(solution_str, ground_truth, data_source)
    # Math
    else:
        from . import latex_math
        res = latex_math.compute_score(solution_str, ground_truth)

    if isinstance(res, dict):
        return res
    elif isinstance(res, (int, float, bool)):
        return float(res)
    else:
        return float(res[0])

@deprecated("verl.utils.reward_score.default_compute_score")
def _default_compute_score(data_source, solution_str, ground_truth, extra_info=None, sandbox_fusion_url=None, concurrent_semaphore=None):
    """
    Legacy function API to be deprecated. Please use `default_compute_score` instead.
    """
    return default_compute_score(data_source, solution_str, ground_truth, extra_info, sandbox_fusion_url, concurrent_semaphore)


__all__ = ["default_compute_score"]
