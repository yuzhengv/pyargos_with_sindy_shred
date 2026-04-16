#!/bin/bash

# Default values
ITERATIONS=150  # Total number of iterations to run
NUM_SENSORS=250  # Default number of sensors
SCRIPT_PATH="./analysis_script_ba.py"  # Path to your Python script

# Check if the script exists
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Error: Anslysis script not found at $SCRIPT_PATH"
    exit 1
fi

for ((i=0; i<ITERATIONS; i++)); do
    echo "Running iteration $i of $ITERATIONS"
    python_output=$(python "$SCRIPT_PATH" --iteration "$i" --num_sensors "$NUM_SENSORS" 2>&1)
    
    # Check if the script executed successfully
    if [ $? -ne 0 ]; then
        echo "Error: Anslysis script failed on iteration $i with the following error:"
        echo "---------------------------------------------------------------"
        echo "$python_output"
        echo "---------------------------------------------------------------"
        exit 1
    fi
    
    # echo "Completed iteration $i"
    # echo "---------------------------------------------------------------"
done

echo "All Anslysis runs completed successfully!"