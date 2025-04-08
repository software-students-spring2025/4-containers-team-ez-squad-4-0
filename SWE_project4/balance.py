#!/usr/bin/env python3
"""
Dataset Balancer - Limits each class to 1000 WAV files and removes Zone.Identifier files

This script:
1. Limits each class in your voice command dataset to 1000 files max
2. Removes Windows Zone.Identifier files
3. Provides a summary of the balancing operation
"""

import os
import shutil
import random
import argparse
from collections import Counter
from pathlib import Path


def balance_dataset(data_dir, max_files=1000, dry_run=False):
    """
    Balance the dataset by limiting each class to the specified number of files.
    
    Args:
        data_dir (str): Path to the dataset directory
        max_files (int): Maximum number of files to keep per class
        dry_run (bool): If True, only print what would be done without actually modifying files
    
    Returns:
        dict: Statistics about the operation
    """
    if not os.path.isdir(data_dir):
        raise ValueError(f"Directory not found: {data_dir}")
    
    print(f"{'DRY RUN - ' if dry_run else ''}Processing dataset in: {data_dir}")
    print(f"Target: Max {max_files} files per class")
    print("-" * 50)
    
    # Statistics
    stats = {
        "initial_counts": {},
        "removed_counts": {},
        "final_counts": {},
        "zone_identifiers_removed": 0
    }
    
    # Process each subdirectory (class)
    for class_name in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, class_name)
        
        # Skip if not a directory
        if not os.path.isdir(class_dir):
            continue
        
        # Get all WAV files in this class
        wav_files = [f for f in os.listdir(class_dir) if f.lower().endswith('.wav')]
        
        # Also find Zone.Identifier files
        zone_files = [f for f in os.listdir(class_dir) if f.endswith(':Zone.Identifier')]
        
        # Store initial counts
        stats["initial_counts"][class_name] = len(wav_files)
        
        # Remove Zone.Identifier files
        for zone_file in zone_files:
            zone_path = os.path.join(class_dir, zone_file)
            if not dry_run:
                try:
                    os.remove(zone_path)
                    stats["zone_identifiers_removed"] += 1
                except Exception as e:
                    print(f"Error removing {zone_path}: {e}")
            else:
                print(f"Would remove: {zone_path}")
                stats["zone_identifiers_removed"] += 1
        
        # Determine how many files to remove
        excess = len(wav_files) - max_files
        if excess <= 0:
            print(f"Class '{class_name}' has {len(wav_files)} files - within limit")
            stats["removed_counts"][class_name] = 0
            stats["final_counts"][class_name] = len(wav_files)
            continue
        
        # Randomly select files to remove
        files_to_remove = random.sample(wav_files, excess)
        
        print(f"Class '{class_name}' has {len(wav_files)} files - removing {excess}")
        stats["removed_counts"][class_name] = excess
        
        # Remove excess files
        if not dry_run:
            # Create a backup directory for removed files (optional)
            backup_dir = os.path.join(data_dir, f"_backup_{class_name}")
            os.makedirs(backup_dir, exist_ok=True)
            
            for file_name in files_to_remove:
                file_path = os.path.join(class_dir, file_name)
                backup_path = os.path.join(backup_dir, file_name)
                
                try:
                    # Move file to backup instead of deleting
                    shutil.move(file_path, backup_path)
                except Exception as e:
                    print(f"Error moving {file_path}: {e}")
        else:
            print(f"  Would remove {excess} files from '{class_name}'")
        
        # Update final count
        remaining = len(wav_files) - excess
        stats["final_counts"][class_name] = remaining
    
    return stats


def print_summary(stats):
    """Print a summary of the balancing operation"""
    print("\n" + "=" * 60)
    print("DATASET BALANCING SUMMARY")
    print("=" * 60)
    
    # Print class-specific stats
    print("\nClass Statistics:")
    print(f"{'Class':<15} {'Initial':<10} {'Removed':<10} {'Final':<10}")
    print("-" * 45)
    
    total_initial = 0
    total_removed = 0
    total_final = 0
    
    for class_name in sorted(stats["initial_counts"].keys()):
        initial = stats["initial_counts"][class_name]
        removed = stats["removed_counts"].get(class_name, 0)
        final = stats["final_counts"][class_name]
        
        print(f"{class_name:<15} {initial:<10} {removed:<10} {final:<10}")
        
        total_initial += initial
        total_removed += removed
        total_final += final
    
    # Print totals
    print("-" * 45)
    print(f"{'TOTAL':<15} {total_initial:<10} {total_removed:<10} {total_final:<10}")
    
    # Print Zone.Identifier stats
    print(f"\nZone.Identifier files removed: {stats['zone_identifiers_removed']}")
    
    # Print overall reduction
    if total_initial > 0:
        reduction_pct = (total_removed / total_initial) * 100
        print(f"\nDataset reduced by {reduction_pct:.1f}% ({total_removed} files)")
    
    print("=" * 60)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Balance voice command dataset by limiting each class to a maximum number of files")
    parser.add_argument("--data_dir", "-d", type=str, default="dataset", 
                        help="Path to the dataset directory (default: 'dataset')")
    parser.add_argument("--max_files", "-m", type=int, default=1000, 
                        help="Maximum number of files to keep per class (default: 1000)")
    parser.add_argument("--dry_run", "-n", action="store_true",
                        help="Dry run: only print what would be done without modifying files")
    
    args = parser.parse_args()
    
    try:
        stats = balance_dataset(args.data_dir, args.max_files, args.dry_run)
        print_summary(stats)
        
        if args.dry_run:
            print("\nThis was a dry run. No files were modified.")
            print("Run without --dry_run to apply the changes.")
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())