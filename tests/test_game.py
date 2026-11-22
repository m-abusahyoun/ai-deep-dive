"""Offline regressions. All inference and leaderboard calls are mocked."""
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
import start_ai_game as game

ROOT = Path(__file__).resolve().parents[1]


class GameTests(unittest.TestCase):
    def setUp(self):
        game.CACHE.clear()

    def test_banks_aligned(self):
        html = (ROOT / 'AI_Deep_Dive.html').read_text(encoding='utf-8')
        embedded = json.loads(html.split('const QUESTIONS = ', 1)[1].split(';\n', 1)[0])
        self.assertEqual(embedded, list(game.QUESTIONS.values()))
        self.assertEqual(len(embedded), 12)
        self.assertEqual(sum(q['technical'] for q in embedded), 4)

    def test_tiers(self):
        for q in game.QUESTIONS.values():
            self.assertEqual(max(a['pts'] for a in q['answers']), 1000)
            self.assertTrue(all(a['pts'] in [750, 850, 1000] for a in q['answers']))

    def test_company_aliases(self):
        aliases = {game.normalise_answer(v): a['pts'] for a in game.QUESTIONS['companies']['answers'] for v in [a['label'], *a['alts']]}
        for answer in ['Replit', 'repl.it', 'X', 'xAI', 'x ai']:
            self.assertEqual(aliases[game.normalise_answer(answer)], 850)
            self.assertIsNone(game.category_rejection('companies', answer))

    def test_known_wrong_category_is_zero_without_inference(self):
        with patch.object(game, 'local_json', side_effect=AssertionError('Unexpected inference')):
            for answer in ['ChatGPT', 'shat gpt', 'Python', 'Replit']:
                result = game.judge('data', answer)
                self.assertFalse(result['valid'])
                self.assertEqual(result['points'], 0)

    def test_invalid_unlisted_never_reaches_rarity(self):
        reply = {'done': True, 'message': {'content': json.dumps({'valid': False, 'uncertain': False, 'reason': 'Wrong category.'})}}
        with patch.object(game, 'local_json', return_value=reply) as call:
            result = game.judge('companies', 'asdfghjkl qwertyuiop')
        self.assertEqual(call.call_count, 1)
        self.assertEqual(result['points'], 0)

    def test_valid_unlisted_auto_scores(self):
        values = [{'valid': True, 'uncertain': False, 'reason': 'Valid data source.'}, {'points': 1000, 'label': 'Weather radar measurements', 'reason': 'Uncommon answer.'}]
        replies = [{'done': True, 'message': {'content': json.dumps(v)}} for v in values]
        with patch.object(game, 'local_json', side_effect=replies) as call:
            result = game.judge('data', 'Weather radar measurements')
        self.assertEqual(call.call_count, 2)
        self.assertEqual(call.call_args_list[1].args[1]['format'], game.RARITY_SCHEMA)
        self.assertNotIn('valid', game.RARITY_SCHEMA['properties'])
        self.assertTrue(result['valid'])
        self.assertFalse(result['review'])
        self.assertEqual(result['points'], 1000)

    def test_uncertain_response_has_no_points(self):
        reply = {'done': True, 'message': {'content': json.dumps({'valid': False, 'uncertain': True, 'reason': 'Cannot verify.'})}}
        with patch.object(game, 'local_json', return_value=reply):
            result = game.judge('companies', 'Unknown example')
        self.assertTrue(result['review'])
        self.assertEqual(result['points'], 0)

    def test_host_pin_and_zero(self):
        data = {'questionId': 'data', 'answer': 'Candidate', 'points': 750, 'pin': 'wrong'}
        with patch.object(game, 'HOST_PIN', '012345'):
            self.assertEqual(game.confirm_host_award(data)[0], 403)
            data.update(pin='012345', points=0)
            status, result = game.confirm_host_award(data)
        self.assertEqual(status, 200)
        self.assertEqual(result, {'confirmed': True, 'points': 0})

    def test_invalid_scores_never_sent(self):
        with patch.object(game, 'urlopen', side_effect=AssertionError('Unexpected network call')):
            for points in [-1, 1001, True, 3.5]:
                self.assertEqual(game.forward_score({'teamName': 'Test player', 'points': points}, 'test-key-1234')[0], 400)

    def test_submission_and_idempotency_key(self):
        with patch.object(game, 'urlopen', return_value=io.BytesIO(b'{"replayed":false}')) as call:
            status, result = game.forward_score({'teamName': 'Test player', 'points': 800}, 'test-key-1234')
        self.assertEqual(status, 200)
        self.assertTrue(result['saved'])
        req = call.call_args.args[0]
        self.assertEqual(json.loads(req.data), {'teamName': 'Test player', 'points': 800})
        self.assertEqual(req.get_header('Idempotency-key'), 'test-key-1234')

    def test_remote_rejection_is_not_saved(self):
        error = HTTPError('https://example.invalid', 409, 'Conflict', {}, io.BytesIO(b'{"error":"Already submitted"}'))
        with patch.object(game, 'urlopen', side_effect=error):
            status, result = game.forward_score({'teamName': 'Test player', 'points': 800}, 'test-key-1234')
        self.assertEqual(status, 409)
        self.assertNotIn('saved', result)


if __name__ == '__main__':
    unittest.main()
