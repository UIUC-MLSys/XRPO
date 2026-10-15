# Copyright 2025 Agentica Project (rllm) and contributors
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
# Vendored from https://github.com/agentica-project/rllm/blob/main/rllm/rewards/code_reward.py (Apache-2.0).
# Modified: import paths rewritten for verl, sandbox entry point renamed.

"""
This module contains the RewardCode class, which evaluates code datasets answers
and assigns rewards based on their correctness on unit tests.
"""

import ast
import json
import multiprocessing
import re
from multiprocessing import Manager
from typing import Any

from verl.utils.reward_score.rllm_code.code_utils.firejail_exec import code_exec_firejail as lc_code_exec
from verl.utils.reward_score.rllm_code.code_utils.humanevalplus import get_num_test_cases
from verl.utils.reward_score.rllm_code.code_utils.humanevalplus import run_test as humanevalplus_run_test
from verl.utils.reward_score.rllm_code.code_utils.kodcode import code_exec as kod_code_exec
from verl.utils.reward_score.rllm_code.code_utils.livecodebench import run_test as lcb_run_test
from verl.utils.reward_score.rllm_code.code_utils.taco import run_test as taco_run_test


def extract_code_from_model(model_response: str):
    """
    Extracts the code from a Markdown-style code block in an LLM output.

    Parameters:
        model_response (str): The text output from the LLM.

    Returns:
        str: The extracted code, or an empty string if no code block is found.
    """
    code_blocks = re.findall(r"```(?:\w+)?\n(.*?)```", model_response, re.DOTALL)
    if not code_blocks:
        return None
    return code_blocks[-1].strip()


def clean_code_main_block(code: str) -> str:
    """
    Removes `if __name__ == "__main__"` blocks from Python code.

    Args:
        code (str): The input Python code.

    Returns:
        str: Cleaned code without the main execution block.
    """
    code_lines = code.split("\n")
    filtered_lines = []
    skip_block = False

    for line in code_lines:
        if line.strip().startswith('if __name__ == "__main__"') or line.strip().startswith("if __name__ == '__main__'"):
            skip_block = True
            continue
        if skip_block:
            # Check if we're out of the block (less indentation)
            if line.strip() and not line.startswith(" ") and not line.startswith("\t"):
                skip_block = False
            else:
                continue
        filtered_lines.append(line)

    return "\n".join(filtered_lines)


def check_correctness(tests: list[dict[str, str]] | dict[str, list[str]], code: str, test_fn, timeout_per_test: int = 12, max_tests: int = 15) -> tuple[bool, dict[str, Any]]:
    """
    Check if generated code passes all test cases within a timeout period.

    Args:
        tests: Test cases in either list of dictionaries or dictionary of lists format
        code: Generated code to test
        test_fn: Function to run tests
        timeout: Maximum execution time in seconds before killing process

    Returns:
        tuple: (bool, dict) where:
            - bool: True if all tests pass, False otherwise
            - dict: Detailed test results with test cases and pass/fail status
    """
    manager = Manager()
    test_results = manager.list()

    def evaluate_code(tests, generation, debug, test_results, test_fn):
        """Helper function to run tests in separate process."""
        try:
            test_results.append(test_fn(tests, test=generation, debug=debug, timeout=timeout_per_test))
        except Exception as e:
            print(f"Error in evaluate_code: {e}")

    original_tests = tests
    if isinstance(tests, list):
        list_tests = tests
        total_tests = len(list_tests)
        if total_tests > max_tests:
            # Sort indices by test input length and take the max_tests longest ones
            selected_indices = sorted(range(total_tests), key=lambda i: len(list_tests[i]["input"]), reverse=True)[:max_tests]
            tests = [list_tests[i] for i in selected_indices]
        num_tests = len(tests)
    else:
        dict_tests = tests
        total_tests = len(dict_tests["inputs"])
        if total_tests > max_tests:
            # Select the tests with the longest input length.
            selected_indices = sorted(range(total_tests), key=lambda i: len(dict_tests["inputs"][i]), reverse=True)[:max_tests]
            # Create a new dict with only the selected test cases
            selected_tests: dict[str, list[str]] = {"inputs": [dict_tests["inputs"][i] for i in selected_indices], "outputs": [dict_tests["outputs"][i] for i in selected_indices]}
            tests = selected_tests
        num_tests = len(tests["inputs"])

    process = multiprocessing.Process(target=evaluate_code, args=(tests, code, False, test_results, test_fn))
    process.start()
    process.join()

    if process.is_alive():
        process.kill()
    test_results_list = list(test_results)

    detailed_results: dict[str, Any] = {"all_passed": False, "test_results": [], "total_tests": num_tests, "passed_tests": 0}

    if len(test_results_list) == 0:
        return False, detailed_results

    test_results_data = test_results_list[0]
    passed_results = [r == True for r in test_results_data]

    # Create detailed test results
    test_results_list_typed: list[dict[str, Any]] = detailed_results["test_results"]
    if isinstance(original_tests, list):
        assert isinstance(tests, list)
        for i, (test, result) in enumerate(zip(tests, passed_results, strict=False)):
            test_results_list_typed.append({"input": test.get("input", ""), "expected": test.get("output", ""), "passed": result})
    else:
        assert isinstance(tests, dict)
        for i, (inp, out, result) in enumerate(zip(tests["inputs"], tests["outputs"], passed_results, strict=False)):
            test_results_list_typed.append({"input": inp, "expected": out, "passed": result})

    detailed_results["passed_tests"] = sum(passed_results)
    detailed_results["all_passed"] = all(passed_results)

    return all(passed_results), detailed_results


def postprocess_lcb_sample(sample):
    sample_inputs = [sample["input"] for sample in sample]
    sample_outputs = [sample["output"] for sample in sample]

    sample_dict = {
        "inputs": sample_inputs,
        "outputs": sample_outputs,
    }

    if sample[0].get("testtype") == "functional":
        metadata = sample[0].get("metadata", {})
        fn_name = metadata.get("func_name", None)
        assert fn_name is not None, f"Function name is not found, check if your LCB data is preprocessed correctly: {metadata}\nSample: {sample}"
        # Fill in the blank
        sample_dict["fn_name"] = fn_name

    sample = {
        "input_output": json.dumps(sample_dict),
    }
    return sample


def primeintellect_check_correctness(tests, code):
    if isinstance(tests, str):
        try:
            tests = ast.literal_eval(tests)
            assert isinstance(tests, dict)
        except (ValueError, SyntaxError) as e:
            print(f"Error parsing string: {e}")
            return False, {"all_passed": False, "error": str(e)}

    assert len(tests) >= 1, "PrimeIntellect needs at least one test case"
    # Convert the tests to the format expected by the taco_run_test function
    inputs = [t["input"] for t in tests]
    outputs = [t["output"] for t in tests]
    fn_name = tests[0].get("fn_name", None)
    tests_formatted = {
        "inputs": inputs,
        "outputs": outputs,
    }
    if fn_name:
        tests_formatted["fn_name"] = fn_name

    return check_correctness(tests_formatted, code, taco_run_test)


def _temp_run(sample, generation, debug, result, metadata_list, timeout):
    res, metadata = lcb_run_test(sample, test=generation, debug=debug, timeout=timeout)
    result.append(res)
    metadata_list.append(metadata)


def lcb_check_correctness_v2(sample, generation, timeout=6, debug=False):
    """Check correctness of code generation with a global timeout.
    The global timeout is to catch some extreme/rare cases not handled by the timeouts
    inside `run_test`"""
    assert len(sample) >= 1, "Sample must contain at least one test case"
    sample = postprocess_lcb_sample(sample)

    manager = multiprocessing.Manager()
    result = manager.list()
    metadata_list = manager.list()

    p = multiprocessing.Process(
        target=_temp_run,
        args=(sample, generation, debug, result, metadata_list, timeout),
    )
    p.start()
    p.join(timeout=(timeout + 1) * len(json.loads(sample["input_output"])["inputs"]) + 5)

    detailed_results = {"all_passed": False, "test_results": [], "total_tests": 0, "passed_tests": 0}

    if p.is_alive():
        p.kill()
    if not result:
        in_outs = json.loads(sample["input_output"])
        # consider that all tests failed
        result.extend([[-1 for i in range(len(in_outs["inputs"]))]])
        detailed_results["total_tests"] = len(in_outs["inputs"])
        detailed_results["test_results"] = [{"input": inp, "expected": out, "passed": False, "error": "global timeout"} for inp, out in zip(in_outs["inputs"], in_outs["outputs"], strict=False)]
        if debug:
            print("global timeout")
        return False, detailed_results

    if not result:
        return False, detailed_results

    # Create detailed test results
    in_outs = json.loads(sample["input_output"])
    detailed_results["total_tests"] = len(result[0])
    detailed_results["test_results"] = [{"input": inp, "expected": out, "passed": res == True, "error": metadata_list[0].get("error", None), "error_message": metadata_list[0].get("error_message", None), "output": metadata_list[0].get("output", None)} for inp, out, res in zip(in_outs["inputs"], in_outs["outputs"], result[0], strict=False)]
    detailed_results["passed_tests"] = sum(1 for r in result[0] if r == True)
    detailed_results["all_passed"] = all(r == True for r in result[0])

    return all(x == True for x in result[0]), detailed_results


def leetcode_check_correctness(tests: dict[str, str], code: str) -> tuple[bool, dict[str, Any]]:
    """
    Check if generated code passes all LeetCode test cases.

    Args:
         tests: Dict of test cases with "functional" key containing test code
         code: Generated code to test
         timeout: Maximum execution time in seconds before killing process
         runtime_debug: Whether to print debug info during test execution

    Returns:
         tuple: (bool, dict) where:
           - bool: True if all tests pass, False otherwise
           - dict: Detailed test results
    """
    succ, output = lc_code_exec(code + "\n" + tests["functional"])
    detailed_results = {"all_passed": succ, "output": output, "test_results": [{"passed": succ, "output": output}]}

    if not succ:
        print(f"Error in code execution: {output}")
    return succ, detailed_results


def kodcode_check_correctness(test: str, code: str, timeout_per_test: int = 5) -> tuple[bool, dict[str, Any]]:
    """
    Check if generated code passes all Kodcode test cases.

    Args:
        test: String of the test file content
        code: Generated code to test
        timeout: Maximum execution time in seconds before killing process
        runtime_debug: Whether to print debug info during test execution

    Returns:
        tuple: (bool, dict) where:
            - bool: True if all tests pass, False otherwise
            - dict: Detailed test results
    """
    # Count the number of test functions in the test file
    num_tests = test.count("def test")

    # Remove 'if __name__ == "__main__":' block if present
    code = clean_code_main_block(code)

    succ, output = kod_code_exec(code, test, timeout_per_test * num_tests)
    detailed_results = {"all_passed": succ, "output": output, "total_tests": num_tests, "test_results": [{"passed": succ, "output": output}]}

    if not succ:
        print(f"Error in code execution: {output}")
    return succ, detailed_results


def humanevalplus_check_correctness(test: str, code: str, timeout_per_test: int = 1) -> tuple[bool, dict[str, Any]]:
    """
    Check if generated code passes all HumanEvalPlus test cases.

    Args:
        test: String of the test file content
        code: Generated code to test
        timeout: Maximum execution time in seconds before killing process
        runtime_debug: Whether to print debug info during test execution

    Returns:
        tuple: (bool, dict) where:
            - bool: True if all tests pass, False otherwise
            - dict: Detailed test results
    """
    code = clean_code_main_block(code)

    num_test_cases = get_num_test_cases(test)
    succ, output = humanevalplus_run_test(code, test, timeout_per_test * num_test_cases)

    detailed_results = {"all_passed": succ, "output": output, "total_tests": num_test_cases, "test_results": [{"passed": succ, "output": output}]}

    if not succ:
        print(f"Error in code execution: {output}")
    return succ, detailed_results


def taco_to_lcb_format(tests):
    """
    Given a dictionary with keys "inputs" and "outputs", returns a list of test cases.
    Each test case is a dictionary with keys "input" and "output". If the lists are unequal,
    missing entries are filled by reusing the first element of the shorter list.

    Args:
        data (dict): A dictionary with keys "inputs" and "outputs", each mapped to a list of strings.

    Returns:
        list of dict: A list where each element is a dict with keys "input" and "output".
    """
    inputs = tests.get("inputs", [])
    outputs = tests.get("outputs", [])

    # Determine the number of test cases to create.
    n = max(len(inputs), len(outputs))

    test_cases = []
    for i in range(n):
        # Use the first element as a fallback if the list is shorter than n.
        inp = inputs[i] if i < len(inputs) else (inputs[0] if inputs else "")
        out = outputs[i] if i < len(outputs) else (outputs[0] if outputs else "")
        out = out[0] if isinstance(out, list) else out
        test_case: dict[str, Any] = {"input": inp, "output": out, "metadata": {}}
        if "fn_name" in tests:
            test_case["testtype"] = "functional"
            test_case["metadata"]["func_name"] = tests["fn_name"]
        test_cases.append(test_case)

    return test_cases

def compute_score(model_response, ground_truth, data_source, format_score=0.1, full_score=1.) -> float:
    """
    Compute the score for a code solution based on the ground truth.
    Args:
        model_output: The agent's response/solution (code)
        ground_truth: The ground truth answer for comparison.
        data_source: The source dataset identifier which determines the scoring method.
        format_score: The score for a partially correct solution
        full_score: The score for a fully correct solution

    Returns:
        float: The score for the code solution
    """    
    model_code = extract_code_from_model(model_response)
    if model_code is None: return 0.0

    if data_source in ["taco", "apps", "code_contests"]:
        tests = taco_to_lcb_format(ground_truth)
        is_correct, detailed_results = lcb_check_correctness_v2(tests, model_code, debug=False)
    elif data_source == "leetcode":
        is_correct, detailed_results = leetcode_check_correctness(ground_truth, model_code)
    elif data_source in ["livecodebench", "codeforces", "primeintellect"]:
        # Handle case where tests is a JSON string
        if isinstance(ground_truth, str):
            ground_truth = json.loads(ground_truth)
        is_correct, detailed_results = lcb_check_correctness_v2(ground_truth, model_code, debug=False)
    elif data_source == "kodcode":
        is_correct, detailed_results = kodcode_check_correctness(ground_truth, model_code)
    elif data_source == "humanevalplus":
        is_correct, detailed_results = humanevalplus_check_correctness(ground_truth, model_code)
    else:
        raise NotImplementedError(f"Dataset {data_source} not implemented")
    
    do_print = False
    if do_print:
        print("\n" + "="*80)
        print(f">> Ground Truth: {ground_truth}")
        print(f">> Score: {full_score if detailed_results['all_passed'] else format_score}")
        print(f">> All Passed: {detailed_results['all_passed']}")
        print(f">> Passed Tests: {detailed_results['passed_tests']}")
        print(f">> Total Tests: {detailed_results['total_tests']}")
        print(f">> Test Results: {detailed_results['test_results']}")

    return full_score if is_correct else format_score