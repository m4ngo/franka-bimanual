#!/usr/bin/env bash

# Visualize a recorded LeRobot episode in Rerun.
# $1 repo_id           e.g. HuskyMango/red-block-simple_20260510_200349
# $2 episode_index     integer

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: $0 <repo_id> <episode_index>"
    exit 1
fi

python "$(dirname "$0")/visualize_episode.py" "$1" "$2"
