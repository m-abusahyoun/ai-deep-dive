#!/usr/bin/env python3
"""Evaluate the actual local model, not simulated responses. Never posts leaderboard scores."""
import argparse
import copy
import json
from pathlib import Path
import time
import start_ai_game as game

# Positives are existing valid examples, deliberately removed from the bank and score anchors
# during each test so that this measures the unlisted-answer path, not a dictionary lookup.
EXAMPLES = [
    ('chatbots', 'Pi', 'pi assistant', 'a wooden kitchen spoon'),
    ('companies', 'Cohere', 'coher', 'a wooden kitchen spoon'),
    ('characters', 'Wheatley', 'wheatly', 'Albert Einstein'),
    ('image_tools', 'Recraft', 'recrft', 'a wooden kitchen spoon'),
    ('devices', 'Hearing aid', 'hearing aide', 'ChatGPT'),
    ('games', 'Total War', 'total warr', 'Microsoft'),
    ('robots', 'Pipeline inspection', 'pipeline inspecton', 'OpenAI'),
    ('ethics', 'Accountability', 'accountabilty', 'a wooden kitchen spoon'),
    ('plots', 'Ridgeline plot', 'ridgline plot', 'Matplotlib'),
    ('languages', 'Scala', 'scalaa', 'TensorFlow'),
    ('data', 'Hyperspectral images', 'hyperspectal images', 'shat gpt'),
    ('learning', 'Online learning', 'online learnng', 'ChatGPT'),
]


def make_cases(quick):
    rows = EXAMPLES
    if quick:
        rows = [r for r in rows if r[0] in {'companies', 'plots', 'languages', 'data'}]
    cases = []
    for question_id, positive, typo, wrong in rows:
        cases.extend([
            (question_id, positive, positive, True, 'unlisted_valid'),
            (question_id, positive, typo, True, 'unlisted_typo'),
            (question_id, None, wrong, False, 'wrong_category'),
            (question_id, None, 'asdfghjkl qwertyuiop zxcvbnm', False, 'nonsense'),
        ])
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quick', action='store_true', help='16 cases in four technical/company categories')
    parser.add_argument('--output', default='judge_test_results.json')
    args = parser.parse_args()
    state = game.readiness()
    if not state['aiReady']:
        raise SystemExit(state['reason'])
    cases = make_cases(args.quick)
    print(f'Testing {game.MODEL} on {len(cases)} cases. This uses your actual laptop model.')
    print('This may take several minutes. It does not change the game or submit scores.')
    original = copy.deepcopy(game.QUESTIONS)
    results = []
    try:
        for index, (question_id, hidden_label, answer, expected_valid, kind) in enumerate(cases, 1):
            game.QUESTIONS.clear()
            game.QUESTIONS.update(copy.deepcopy(original))
            if hidden_label:
                game.QUESTIONS[question_id]['answers'] = [
                    a for a in game.QUESTIONS[question_id]['answers'] if a['label'] != hidden_label
                ]
            game.CACHE.clear()
            started = time.monotonic()
            result = game.judge(question_id, answer)
            actual_valid = result.get('valid') is True and result.get('review') is not True and result.get('points', 0) > 0
            outcome = 'REVIEW' if result.get('review') else 'PASS' if actual_valid == expected_valid else 'FAIL'
            record = {'question_id': question_id, 'question': original[question_id]['q'],
                      'answer': answer, 'test_kind': kind, 'expected_valid': expected_valid,
                      'actual_valid': actual_valid, 'outcome': outcome,
                      'seconds': round(time.monotonic() - started, 2), 'result': result}
            results.append(record)
            print(f'[{index}/{len(cases)}] {outcome}: {question_id}: {answer}', flush=True)
            if outcome != 'PASS':
                print('  ' + result.get('reason', 'No reason'), flush=True)
    except KeyboardInterrupt:
        print('\nStopped; saving completed cases.')
    finally:
        game.QUESTIONS.clear()
        game.QUESTIONS.update(original)
        game.CACHE.clear()
    summary = {name.lower(): sum(r['outcome'] == name for r in results) for name in ['PASS', 'FAIL', 'REVIEW']}
    summary['tested'] = len(results)
    summary['false_accepts'] = sum(not r['expected_valid'] and r['actual_valid'] for r in results)
    summary['false_rejects'] = sum(r['expected_valid'] and not r['actual_valid'] and r['outcome'] == 'FAIL' for r in results)
    output = Path(args.output).resolve()
    output.write_text(json.dumps({'model': game.MODEL, 'summary': summary, 'cases': results}, ensure_ascii=False, indent=2), encoding='utf-8')
    print('\nSummary: ' + json.dumps(summary))
    print('Report: ' + str(output))
    print('A passing suite does not prove perfect judging; add the actual failures from your event.')


if __name__ == '__main__':
    main()
