#!/bin/bash

python evaluation/run_forward_search_evaluation.py --base-policy-file just_policy_default_after_update_5100.pt --consider-all-moves-for-opening-placements --num-subprocesses 32 --dont-propose-devcards --dont-propose-trades --zero-opponent-hidden-states


