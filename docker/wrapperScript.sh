#!/bin/bash

# Start the first process
python3 -W ignore stream_cloud.py &

sleep 5
  
# Start the second process
python3 stream_edge.py &

# Wait for any process to exit
wait -n
  
# Exit with status of process that exited first
exit $?
