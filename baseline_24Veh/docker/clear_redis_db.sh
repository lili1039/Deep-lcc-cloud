#!/bin/bash

redis-cli -h 127.0.0.1 -p 6379 -a chlpw1039<<EOF

SELECT 2

FLUSHDB

QUIT
EOF

echo "Clear Done"