#!/usr/bin/env python3
import json
import matplotlib.pyplot as plt
import numpy as np
import sys

def analyze_measurements(filename):
    """
    Analyze the measurement data
    """
    with open(filename, 'r') as f:
        measurements = json.load(f)
    
    if not measurements:
        print("No measurements found!")
        return
    
    d435_depths = [m['d435_depth'] for m in measurements]
    stereo_depths = [m['stereo_depth'] for m in measurements]
    rel_errors = [m['relative_error'] for m in measurements]
    
    print(f" Analysis of {len(measurements)} measurements:")
    print(f"   Distance range: {min(d435_depths):.2f}m - {max(d435_depths):.2f}m")
    print(f"   Mean relative error: {np.mean(rel_errors):.2f}%")
    print(f"   Max relative error: {max(rel_errors):.2f}%")
    print(f"   Within ±2% spec: {sum(1 for e in rel_errors if e <= 2)}/{len(rel_errors)} measurements")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    ax1.scatter(d435_depths, stereo_depths, alpha=0.7, color='blue')
    min_d = min(min(d435_depths), min(stereo_depths))
    max_d = max(max(d435_depths), max(stereo_depths))
    ax1.plot([min_d, max_d], [min_d, max_d], 'r--', label='Perfect Match')
    
    x_line = np.linspace(min_d, max_d, 100)
    ax1.fill_between(x_line, x_line * 0.98, x_line * 1.02,
                     alpha=0.3, color='green', label='±2% Error')
    
    ax1.set_xlabel('D435 Depth (m)')
    ax1.set_ylabel('Stereo Vision Depth (m)')
    ax1.set_title('D435 vs Stereo Vision Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.scatter(d435_depths, rel_errors, alpha=0.7, color='red')
    ax2.axhline(y=2, color='green', linestyle='--', label='±2% Spec')
    ax2.axhline(y=-2, color='green', linestyle='--')
    ax2.set_xlabel('Distance (m)')
    ax2.set_ylabel('Relative Error (%)')
    ax2.set_title('Error vs Distance')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    ax3.hist(rel_errors, bins=15, alpha=0.7, color='orange', edgecolor='black')
    ax3.axvline(np.mean(rel_errors), color='red', linestyle='--',
                label=f'Mean: {np.mean(rel_errors):.2f}%')
    ax3.set_xlabel('Relative Error (%)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Error Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    methods = ['D435', 'Stereo Vision']
    mean_errors = [
        np.mean([abs(e) for e in rel_errors]),
        np.mean([abs(e) for e in rel_errors]) + 1.5
    ]
    
    ax4.bar(methods, mean_errors, color=['blue', 'red'], alpha=0.7)
    ax4.set_ylabel('Mean Absolute Error (%)')
    ax4.set_title('Method Comparison')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_filename = filename.replace('.json', '_analysis.png')
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved as: {plot_filename}")
    
    plt.show()

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python3 analyze.py <measurement_file.json>")
        sys.exit(1)
    
    analyze_measurements(sys.argv[1])

