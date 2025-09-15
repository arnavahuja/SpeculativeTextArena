#!/bin/bash

# Script to run 10 speculative chess games serially with breaks
# Usage: ./run_10_games.sh

echo "Starting 10 speculative chess games..."
echo "Started at: $(date)"

for i in {1..10}; do
    echo ""
    echo "========================================"
    echo "Starting Game $i of 10"
    echo "Time: $(date)"
    echo "========================================"

    # Run the game
    uv run Speculative_Chess.py

    # Check if the game completed successfully
    if [ $? -eq 0 ]; then
        echo "Game $i completed successfully"
    else
        echo "Game $i failed with error code $?"
    fi

    # Break between games (skip break after last game)
    if [ $i -lt 10 ]; then
        echo "Taking a 5-second break before next game..."
        sleep 5
    fi
done

echo ""
echo "========================================"
echo "All 10 games completed!"
echo "Finished at: $(date)"
echo "========================================"