#!/usr/bin/env python3
"""
Memory monitoring utility for vLLM CPU memory leak detection.
Can be run as a separate process to monitor target process.
"""
import psutil
import time
import argparse
import csv
import sys
from datetime import datetime

def monitor_process_memory(pid, interval=1.0, output_file=None, duration=None):
    """
    Monitor memory usage of a specific process
    
    Args:
        pid: Process ID to monitor
        interval: Monitoring interval in seconds
        output_file: CSV file to save results
        duration: Maximum monitoring duration in seconds (None for infinite)
    """
    try:
        process = psutil.Process(pid)
        print(f"Monitoring process {pid}: {process.name()}")
    except psutil.NoSuchProcess:
        print(f"Process {pid} not found")
        return
    
    # Setup CSV output
    csv_writer = None
    csv_file = None
    if output_file:
        csv_file = open(output_file, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            'timestamp', 'rss_mb', 'vms_mb', 'percent_memory', 
            'num_threads', 'cpu_percent', 'system_available_mb', 'system_used_percent'
        ])
    
    start_time = time.time()
    try:
        while True:
            current_time = time.time()
            if duration and (current_time - start_time) > duration:
                break
                
            try:
                # Process memory info
                memory_info = process.memory_info()
                memory_percent = process.memory_percent()
                cpu_percent = process.cpu_percent()
                num_threads = process.num_threads()
                
                # System memory info
                sys_memory = psutil.virtual_memory()
                
                rss_mb = memory_info.rss / 1024 / 1024
                vms_mb = memory_info.vms / 1024 / 1024
                sys_available_mb = sys_memory.available / 1024 / 1024
                sys_used_percent = sys_memory.percent
                
                timestamp = datetime.now().isoformat()
                
                # Print to console
                print(f"[{timestamp}] RSS: {rss_mb:.1f}MB, VMS: {vms_mb:.1f}MB, "
                      f"CPU: {cpu_percent:.1f}%, Threads: {num_threads}, "
                      f"System: {sys_used_percent:.1f}% used")
                
                # Write to CSV
                if csv_writer:
                    csv_writer.writerow([
                        timestamp, rss_mb, vms_mb, memory_percent,
                        num_threads, cpu_percent, sys_available_mb, sys_used_percent
                    ])
                    csv_file.flush()
                    
            except psutil.NoSuchProcess:
                print(f"Process {pid} terminated")
                break
            except Exception as e:
                print(f"Error monitoring process: {e}")
                break
                
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")
    finally:
        if csv_file:
            csv_file.close()
            print(f"Results saved to {output_file}")

def find_vllm_processes():
    """Find all processes that might be vLLM related"""
    vllm_processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if proc.info['cmdline']:
                cmdline = ' '.join(proc.info['cmdline'])
                if 'test_distN.py' in cmdline or 'vllm' in cmdline.lower():
                    vllm_processes.append({
                        'pid': proc.info['pid'],
                        'name': proc.info['name'], 
                        'cmdline': cmdline[:100] + '...' if len(cmdline) > 100 else cmdline
                    })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return vllm_processes

def main():
    parser = argparse.ArgumentParser(description='Monitor process memory usage')
    parser.add_argument('--pid', type=int, help='Process ID to monitor')
    parser.add_argument('--find-vllm', action='store_true', help='Find vLLM related processes')
    parser.add_argument('--interval', type=float, default=1.0, help='Monitoring interval in seconds')
    parser.add_argument('--output', type=str, help='Output CSV file')
    parser.add_argument('--duration', type=float, help='Monitoring duration in seconds')
    
    args = parser.parse_args()
    
    if args.find_vllm:
        processes = find_vllm_processes()
        if processes:
            print("Found vLLM related processes:")
            for proc in processes:
                print(f"  PID {proc['pid']}: {proc['name']} - {proc['cmdline']}")
        else:
            print("No vLLM related processes found")
        return
    
    if not args.pid:
        print("Please specify --pid or use --find-vllm to find processes")
        return
    
    monitor_process_memory(
        pid=args.pid,
        interval=args.interval,
        output_file=args.output,
        duration=args.duration
    )

if __name__ == '__main__':
    main()
