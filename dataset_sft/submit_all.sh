#!/bin/bash

bash ~/cancel_aug.sh osk-gpu[61-68]
NODES=(osk-gpu61 osk-gpu62 osk-gpu63 osk-gpu64 osk-gpu65 osk-gpu66 osk-gpu67 osk-gpu68)

for i in {0..7}; do
    sbatch --nodelist="${NODES[$i]}" --export=ALL,SHARD_INDEX=$i create.sh
    echo "Submitted shard $i → ${NODES[$i]}"
done