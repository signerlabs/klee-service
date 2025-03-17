#!/bin/bash

# find port 6190 process
echo "find port 6190 process..."

# try use lsof (适用于 macOS 和大多数 Linux)
if command -v lsof &> /dev/null; then
  PID=$(lsof -ti:6190)
  if [ -n "$PID" ]; then
    echo "find process id: $PID"
    echo "killing process..."
    kill -9 $PID
    echo "process killed"
    exit 0
  fi
fi

# if lsof is not available or not find process, try use netstat (适用于大多数 Linux)
if command -v netstat &> /dev/null; then
  PID=$(netstat -tulpn 2>/dev/null | grep ":6190" | awk '{print $7}' | cut -d'/' -f1)
  if [ -n "$PID" ]; then
    echo "find process id: $PID"
    echo "killing process..."
    kill -9 $PID
    echo "process killed"
    exit 0
  fi
fi

# if not find process, print not find port 6190 process
echo "not find port 6190 process"