#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p bin
swiftc -swift-version 5 -O -framework IOBluetooth -framework Foundation \
  -o bin/divoom-bridge bridge/DivoomBridge.swift
echo "built bin/divoom-bridge"
