
from dotenv import dotenv_values

env = dotenv_values('.env')

import unittest
from unittest import TestCase

from evalscope.config import TaskConfig
from evalscope.constants import EvalType, JudgeStrategy
from evalscope.run import run_task
from evalscope.utils.logger import get_logger

logger = get_logger()
base_config = {
            'model': 'qwen-plus',
            'api_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
            'api_key': env.get('DASHSCOPE_API_KEY'),
            'eval_type': EvalType.SERVICE,
            'eval_batch_size': 5,
            'limit': 5,
            'generation_config': {
                'max_tokens': 4096,
                'temperature': 0.0,
                'seed': 42,
                'parallel_tool_calls': True
            },
            'debug': True,
            'rerun_review': True
        }

def _run_dataset_test(dataset_name, dataset_args=None, use_mock=False, **config_overrides):
    """Helper method to run test for a specific dataset."""
    config = base_config.copy()
    config['datasets'] = [dataset_name]

    if not env.get('DASHSCOPE_API_KEY'):
        use_mock = True
        logger.warning('DASHSCOPE_API_KEY is not set. Using mock evaluation.')

    if use_mock:
        config['eval_type'] = EvalType.MOCK_LLM

    # 应用配置覆盖
    config.update(config_overrides)


    if dataset_args:
        config['dataset_args'] = {dataset_name: dataset_args}

    task_cfg = TaskConfig(**config)
    run_task(task_cfg=task_cfg)
def live_code_bench():
    """Test LiveCodeBench dataset."""
    dataset_args = {
        'extra_params': {
            'start_date': '2024-08-01',
            'end_date': '2025-02-28'
        },
        'local_path': '/root/.cache/modelscope/hub/datasets/AI-ModelScope/code_generation_lite'
    }
    _run_dataset_test('live_code_bench', dataset_args)

if __name__ == '__main__':
    live_code_bench()
    pass