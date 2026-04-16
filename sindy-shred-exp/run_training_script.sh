#!/bin/bash

# Default values
ITERATIONS=150  # Total number of iterations to run
NUM_SENSORS=250  # Default number of sensors
SCRIPT_PATH="./training_script.py"  # Path to your Python script

# Check if the script exists
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Error: Training script not found at $SCRIPT_PATH"
    exit 1
fi

# Parse command line arguments
# while [[ $# -gt 0 ]]; do
#     case $1 in
#         --iterations)
#             ITERATIONS="$2"
#             shift 2
#             ;;
#         --num_sensors)
#             NUM_SENSORS="$2"
#             shift 2
#             ;;
#         --help)
#             echo "Usage: $0 [--iterations N] [--num_sensors M]"
#             echo "  --iterations N    Run N iterations (default: 10)"
#             echo "  --num_sensors M   Use M sensors in each run (default: 250)"
#             exit 0
#             ;;
#         *)
#             echo "Unknown option: $1"
#             echo "Use --help for usage information"
#             exit 1
#             ;;
#     esac
# done

# Run the training script multiple times
# echo "Starting $ITERATIONS training runs with $NUM_SENSORS sensors each"
# echo "==============================================================="

for ((i=0; i<ITERATIONS; i++)); do
    echo "Running iteration $i of $ITERATIONS"
    python_output=$(python "$SCRIPT_PATH" --iteration "$i" --num_sensors "$NUM_SENSORS" 2>&1)
    
    # Check if the script executed successfully
    if [ $? -ne 0 ]; then
        echo "Error: Training script failed on iteration $i with the following error:"
        echo "---------------------------------------------------------------"
        echo "$python_output"
        echo "---------------------------------------------------------------"
        exit 1
    fi
    
    # echo "Completed iteration $i"
    # echo "---------------------------------------------------------------"
done

echo "All training runs completed successfully!"