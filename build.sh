#!/bin/bash
apt-get update
apt-get install -y $(cat Aptfile)
