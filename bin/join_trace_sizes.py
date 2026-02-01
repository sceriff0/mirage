#!/usr/bin/env python3
"""join_trace_sizes.py — Merge Nextflow trace with input size metadata.

Usage:
    python bin/join_trace_sizes.py .trace/trace.txt .trace/input_sizes.csv .trace/merged.csv

This script joins the Nextflow execution trace (containing runtime, memory usage, etc.)
with the input size metadata collected during pipeline execution. The resulting merged
dataset enables analysis of resource usage as a function of input file size.
"""

import pandas as pd
import sys


def parse_mem(val):
    """Convert memory strings (e.g., '3.2 GB') to numeric bytes."""
    if pd.isna(val):
        return None
    val = str(val).strip()
    if not val or val == '-':
        return None

    units = {'B': 1, 'KB': 1e3, 'MB': 1e6, 'GB': 1e9, 'TB': 1e12}
    val_upper = val.upper()

    for unit, factor in units.items():
        if val_upper.endswith(unit):
            try:
                return float(val[:len(val)-len(unit)].strip()) * factor
            except ValueError:
                return None

    # Try parsing as plain number
    try:
        return float(val)
    except ValueError:
        return None


def main():
    if len(sys.argv) != 4:
        print("Usage: join_trace_sizes.py <trace.txt> <input_sizes.csv> <output.csv>")
        print("Example: python bin/join_trace_sizes.py .trace/trace.txt .trace/input_sizes.csv .trace/merged.csv")
        sys.exit(1)

    trace_file = sys.argv[1]
    sizes_file = sys.argv[2]
    output_file = sys.argv[3]

    # Read trace file (tab-separated)
    print(f"Reading trace: {trace_file}")
    trace = pd.read_csv(trace_file, sep='\t')

    # Read sizes file (comma-separated)
    print(f"Reading sizes: {sizes_file}")
    sizes = pd.read_csv(sizes_file)

    # Extract sample_id from the trace 'tag' column
    # Tag format is typically "sample_id" or "sample_id - description"
    if 'tag' in trace.columns:
        trace['sample_id'] = trace['tag'].str.split(' - ').str[0].str.strip()
    elif 'name' in trace.columns:
        # Fallback: extract from name if tag not available
        trace['sample_id'] = trace['name'].str.extract(r'\(([^)]+)\)')[0]

    # Merge on process and sample_id
    merged = trace.merge(sizes, on=['process', 'sample_id'], how='left')

    # Convert memory columns to numeric bytes
    mem_columns = ['peak_rss', 'peak_vmem', 'memory', 'rss', 'vmem']
    for col in mem_columns:
        if col in merged.columns:
            merged[f'{col}_bytes'] = merged[col].apply(parse_mem)

    # Convert duration columns to seconds if present
    duration_columns = ['duration', 'realtime']
    for col in duration_columns:
        if col in merged.columns:
            merged[f'{col}_sec'] = merged[col].apply(parse_duration)

    # Write output
    merged.to_csv(output_file, index=False)
    print(f"Merged {len(merged)} rows -> {output_file}")

    # Print summary statistics
    if 'bytes' in merged.columns:
        matched = merged['bytes'].notna().sum()
        print(f"  - {matched}/{len(merged)} rows have input size data")


def parse_duration(val):
    """Convert duration strings to seconds."""
    if pd.isna(val):
        return None
    val = str(val).strip()
    if not val or val == '-':
        return None

    # Handle formats like "1h 30m 45s", "5m 30s", "45s", "1.5s"
    import re

    total_seconds = 0.0

    # Try parsing as milliseconds (format: "123ms")
    ms_match = re.search(r'([\d.]+)\s*ms', val)
    if ms_match:
        total_seconds += float(ms_match.group(1)) / 1000

    # Parse hours
    h_match = re.search(r'([\d.]+)\s*h', val)
    if h_match:
        total_seconds += float(h_match.group(1)) * 3600

    # Parse minutes
    m_match = re.search(r'([\d.]+)\s*m(?!s)', val)  # m but not ms
    if m_match:
        total_seconds += float(m_match.group(1)) * 60

    # Parse seconds
    s_match = re.search(r'([\d.]+)\s*s\b', val)
    if s_match:
        total_seconds += float(s_match.group(1))

    # If no units found, try parsing as plain seconds
    if total_seconds == 0.0:
        try:
            return float(val)
        except ValueError:
            return None

    return total_seconds if total_seconds > 0 else None


if __name__ == '__main__':
    main()
