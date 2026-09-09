#!/usr/bin/env python3
"""Check which conversations had injection positions dropped due to failed retries."""

import json
from pathlib import Path
import sys

def main():
    root_dir = Path(__file__).resolve().parent.parent
    state_path = root_dir / 'DiagFlowBench_Dataset' / 'batch_injection_state.json'
    conv_dir = root_dir / 'DiagFlowBench_Dataset' / 'conversations'

    if not state_path.exists():
        print("ERROR: batch_injection_state.json not found.")
        sys.exit(1)

    state = json.loads(state_path.read_text(encoding="utf-8"))
    plan = state.get('injection_plan', {})
    
    dropped = []
    total_planned = 0
    total_actual = 0
    
    for conv_stem, conv_plan in plan.items():
        planned_k = conv_plan.get('k', 0)
        total_planned += planned_k
        
        mixed_file = conv_dir / f"{conv_stem}_mixed.json"
        if mixed_file.exists():
            mixed_data = json.loads(mixed_file.read_text(encoding="utf-8"))
            actual_k = len(mixed_data.get('injection_log', []))
            total_actual += actual_k
            
            if actual_k < planned_k:
                diff = planned_k - actual_k
                dropped.append((conv_stem, diff, planned_k, actual_k))
                
    print(f"\n============================================================")
    print(f"  Injection Attrition Report")
    print(f"============================================================")
    print(f"  Total planned injections: {total_planned}")
    print(f"  Total successful injections: {total_actual}")
    print(f"  Total dropped injections: {total_planned - total_actual}")
    print(f"  Conversations with dropped injections: {len(dropped)}\n")
    
    for stem, diff, p, a in dropped:
        print(f"  {stem}: planned {p}, got {a} (dropped {diff})")

if __name__ == "__main__":
    main()
