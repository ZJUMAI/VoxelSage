import importlib.util
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1] / 'evaluation'))
import run_large_eval as evaluator


def test_numeric_scoring_rejects_substrings_and_wrong_unit():
    task = {'expected': 15.5, 'unit': 'mm', 'tolerance': .051}
    assert evaluator.score('{"value":15.5,"unit":"mm"}', task)['answer_correct']
    assert not evaluator.score('{"value":115.5,"unit":"mm"}', task)['answer_correct']
    assert not evaluator.score('{"value":15.5,"unit":"cm"}', task)['answer_correct']
    assert not evaluator.score('{"value":true,"unit":"mm"}', task)['answer_correct']


def test_unknown_is_not_zero_and_boolean_is_not_number():
    assert evaluator.score('{"value":null,"unit":"mm"}', {'expected': None, 'unit': 'mm'})['answer_correct']
    assert not evaluator.score('{"value":0,"unit":"mm"}', {'expected': None, 'unit': 'mm'})['answer_correct']
    assert not evaluator.score('{"value":0,"unit":"bool"}', {'expected': False, 'unit': 'bool'})['answer_correct']


def test_tasks_exclude_pilot_and_oracle_does_not_invent_zeros():
    tasks = evaluator.build_tasks()
    assert not (set(c for t in tasks for c in t['cases']) & set(evaluator.MANIFEST['pilot_excluded_cases']))
    assert len({t['id'] for t in tasks}) == len(tasks)
    assert evaluator.evidence('BDMAP_00009003').get('tumor_count') is None
