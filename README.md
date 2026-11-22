# AI Deep Dive

An ocean-themed AI trivia game built for a university event. Players answer seven questions, dive deeper with rarer answers, and submit a score to an event leaderboard. The AI judge runs locally through Ollama.

## Features

- Twelve question categories covering everyday AI, entertainment, programming and data.
- Seven questions per round: five general categories and two technical categories.
- Instant scoring for listed answers and aliases; local AI judging for unlisted answers and spelling variations.
- Separate category-validity and rarity-scoring steps.
- A paused timer during AI judging and host review.
- A PIN-protected host override for uncertain answers, including a zero-point decision.
- Leaderboard submission through the Python server, with retry protection and an automatic return to the next player's name screen after confirmation.

## Requirements

- Python 3.10 or later; Anaconda Python works.
- [Ollama](https://ollama.com/download), running locally.
- The `qwen3:4b` model, downloaded once.

The Python server uses only the standard library. No pip packages or API keys are required.

## Run locally

Download or clone the repository, open a terminal in its folder, and run:

```text
ollama pull qwen3:4b
python start_ai_game.py
```

On macOS/Linux, use `python3` if needed. Keep Ollama and the game terminal open. The game opens at `http://127.0.0.1:8765`.

Windows users can run `SETUP_LOCAL_AI.bat` once, then `START_GAME.bat`. If the launcher cannot find Python, open **Anaconda Prompt**, navigate to this folder, and run `python start_ai_game.py`.

Judging works offline after setup. Leaderboard submission requires internet. Opening the HTML directly does not enable the local AI judge. GitHub Pages cannot run the Python server or Ollama; this repository provides source code to run locally.

## Scoring

| Answer | Depth |
| --- | ---: |
| Common valid answer | 750 m |
| Less-common valid answer | 850 m |
| Rare valid answer | 1,000 m |
| Incorrect or timed-out question | 0 m |

Maximum depth is **7,000 m**. The leaderboard score is `round(depth / 7000 * 1000)`.

Rarity is estimated for a university audience. Listed answers have fixed scores; unlisted answers receive an AI estimate. It is not measured from live player frequencies. Validity is checked before any rarity points are awarded. A round allows one accepted answer per question, with retries after a rejection until the remaining time expires.

## Host controls

The terminal prints a fresh six-digit host PIN when the server starts. Keep it private. It authorizes manual review awards; routine valid AI judgements do not need host approval.

The same name cannot start another round in the same browser profile. This is a convenience guard, not identity verification. The host is responsible for enforcing one attempt per participant.

## Leaderboard integration

`start_ai_game.py` contains the event's `LEADERBOARD_URL`. The local `/scores` route forwards a player's name and integer score to `/api/v1/booths/ai/scores`, preserving the Idempotency-Key on retries.

Submission happens only when the player presses **Submit score to leaderboard**. Configure this URL for your own compatible leaderboard before using the project at another event. Do not submit test scores to someone else's live event.

## Tests and diagnostics

Offline regression checks use Python's standard library:

```text
python -m unittest discover -s tests -v
```

To check a real local-model response:

```text
python start_ai_game.py --check-judge
```

To evaluate unseen valid answers, typos, incorrect answers and nonsense:

```text
python check_judge.py --quick
python check_judge.py
```

The quick suite runs 16 cases; the full suite runs 48 across all 12 categories. Valid examples are hidden from the bank during evaluation. The report is written to `judge_test_results.json` and is excluded from Git. These commands do not post leaderboard scores.

## Project files

| File | Purpose |
| --- | --- |
| `AI_Deep_Dive.html` | Game interface and instant-answer bank |
| `start_ai_game.py` | Local server, answer validation, rarity scoring and leaderboard forwarding |
| `questions.json` | Questions and scoring examples supplied to the judge |
| `check_judge.py` | Evaluation of the actual local model |
| `tests/` | Offline regression checks |
| `START_GAME.bat` | Windows launcher |
| `SETUP_LOCAL_AI.bat` | One-time Windows model setup |

When changing questions, keep `questions.json` and the HTML's embedded answer bank aligned.

## Current limitations

A local language model can falsely accept or reject unfamiliar answers. Test it with the intended audience before an event; the evaluation suite cannot guarantee perfect judging. Rarity estimates are subjective. The server is designed for one supervised laptop and listens only on localhost. Host controls and browser-side scoring are not a tamper-proof competition system.

## References

- [Ollama chat API](https://docs.ollama.com/api/chat)
- [Ollama structured outputs](https://docs.ollama.com/capabilities/structured-outputs)
- [Qwen3 4B](https://ollama.com/library/qwen3:4b)
