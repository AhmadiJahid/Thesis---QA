#!/usr/bin/env python3
"""
Results Visualization and Analysis Script

Analyzes run results from runs/<component>/ directories, generates visualizations,
and creates a presentable HTML report for presentation.

Router runs are saved under:
  - runs/average_zero_shot/<run_id>/  (when using --prompt_file prompt_zero_shot.md)
  - runs/average_few_shot/<run_id>/   (default few-shot prompt)

Usage:
    python scripts/analyze_runs.py --component average_zero_shot   # plots -> reports/average_zero_shot/
    python scripts/analyze_runs.py --component average_few_shot   # plots -> reports/average_few_shot/
    python scripts/analyze_runs.py --component router [--output-dir reports/]
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import argparse

try:
    import matplotlib.pyplot as plt
    import matplotlib
    import seaborn as sns
    import pandas as pd
    from jinja2 import Template
except ImportError as e:
    print(f"Error: Missing required package. Install with: pip install matplotlib seaborn pandas jinja2")
    print(f"Missing: {e.name}")
    sys.exit(1)

# Set style for academic presentations
try:
    matplotlib.style.use('seaborn-v0_8-paper')
except OSError:
    # Fallback to available style
    matplotlib.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10


def get_model_short_name(model_id):
    """Extract short model name from full model ID."""
    if not model_id or model_id == 'N/A':
        return 'N/A'
    # Extract model name (e.g., "Qwen/Qwen2.5-1.5B-Instruct" -> "Qwen2.5-1.5B")
    parts = model_id.split('/')
    if len(parts) > 1:
        model_name = parts[-1]
    else:
        model_name = model_id
    # Remove common suffixes
    for suffix in ['-Instruct', '-Chat', '-Base']:
        if model_name.endswith(suffix):
            model_name = model_name[:-len(suffix)]
    return model_name


def load_run_data(runs_dir, component="router", skip_archived=True):
    """
    Load all run data from runs/<component>/ directory.
    
    Args:
        runs_dir: Path to runs directory
        component: Component name (e.g., "router")
        skip_archived: If True, skip folders named "archived"
    
    Returns:
        List of dicts with run data
    """
    component_dir = Path(runs_dir) / component
    if not component_dir.exists():
        print(f"Warning: {component_dir} does not exist")
        return []
    
    runs_data = []
    
    for run_dir in component_dir.iterdir():
        # Skip archived folders (check if "archived" is in folder name)
        if skip_archived and "archived" in run_dir.name.lower():
            print(f"Skipping archived folder: {run_dir.name}")
            continue
        
        # Skip if not a directory
        if not run_dir.is_dir():
            continue
        
        # Load metrics: prefer metrics.json, fall back to metrics_aggregated.json (multi-run average)
        metrics_file = run_dir / "metrics.json"
        aggregated_file = run_dir / "metrics_aggregated.json"
        config_file = run_dir / "config.json"
        detailed_file = run_dir / "detailed_results.json"
        detailed_run0_file = run_dir / "detailed_results_run_0.json"

        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
        elif aggregated_file.exists():
            with open(aggregated_file, 'r') as f:
                raw = json.load(f)
            # Normalize to same shape as metrics.json for plotting (use _mean as primary value)
            metrics = {
                "overall_accuracy": raw.get("overall_accuracy_mean", raw.get("overall_accuracy", 0)),
                "overall_accuracy_std": raw.get("overall_accuracy_std", 0),
                "hop_1_accuracy": raw.get("hop_1_accuracy_mean", raw.get("hop_1_accuracy", 0)),
                "hop_1_accuracy_std": raw.get("hop_1_accuracy_std", 0),
                "hop_2_accuracy": raw.get("hop_2_accuracy_mean", raw.get("hop_2_accuracy", 0)),
                "hop_2_accuracy_std": raw.get("hop_2_accuracy_std", 0),
                "hop_3_accuracy": raw.get("hop_3_accuracy_mean", raw.get("hop_3_accuracy", 0)),
                "hop_3_accuracy_std": raw.get("hop_3_accuracy_std", 0),
                "total_questions": raw.get("total_questions"),
                "correct_predictions": raw.get("correct_predictions"),
                "num_runs": raw.get("num_runs"),
            }
            metrics = {k: v for k, v in metrics.items() if v is not None}
        else:
            print(f"Warning: neither metrics.json nor metrics_aggregated.json found, skipping {run_dir.name}")
            continue

        try:
            config = {}
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config = json.load(f)

            detailed_results = []
            for fpath in (detailed_file, detailed_run0_file):
                if fpath.exists():
                    with open(fpath, 'r') as f:
                        detailed_results = json.load(f)
                    break
            
            model_id = config.get('model_id', 'N/A')
            model_short = get_model_short_name(model_id)
            
            runs_data.append({
                'run_id': run_dir.name,
                'metrics': metrics,
                'config': config,
                'detailed_results': detailed_results,
                'run_dir': run_dir,
                'model_id': model_id,
                'model_short': model_short
            })
            
        except Exception as e:
            print(f"Error loading {run_dir.name}: {e}")
            continue
    
    # Sort by run_id (chronological if using timestamp format)
    runs_data.sort(key=lambda x: x['run_id'])
    
    return runs_data


def plot_overall_accuracy(runs_data, output_dir):
    """Plot overall accuracy comparison across runs."""
    run_ids = [r['run_id'] for r in runs_data]
    accuracies = [r['metrics']['overall_accuracy'] * 100 for r in runs_data]
    model_names = [r.get('model_short', 'N/A') for r in runs_data]
    
    # Create labels with run_id and model name
    labels = [f"{run_id}\n({model})" for run_id, model in zip(run_ids, model_names)]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(range(len(run_ids)), accuracies, color=sns.color_palette("husl", len(run_ids)))
    
    # Add value labels on bars
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.2f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Run ID (Model)', fontsize=12)
    ax.set_ylabel('Overall Accuracy (%)', fontsize=12)
    ax.set_title('Overall Accuracy Comparison Across Runs', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(run_ids)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'overall_accuracy.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return 'overall_accuracy.png'


def plot_per_hop_accuracy(runs_data, output_dir):
    """Plot per-hop accuracy trends across runs."""
    run_ids = [r['run_id'] for r in runs_data]
    model_names = [r.get('model_short', 'N/A') for r in runs_data]
    
    hop_1_acc = [r['metrics'].get('hop_1_accuracy', 0) * 100 for r in runs_data]
    hop_2_acc = [r['metrics'].get('hop_2_accuracy', 0) * 100 for r in runs_data]
    hop_3_acc = [r['metrics'].get('hop_3_accuracy', 0) * 100 for r in runs_data]
    
    # Create labels with run_id and model name
    labels = [f"{run_id}\n({model})" for run_id, model in zip(run_ids, model_names)]
    
    fig, ax = plt.subplots(figsize=(14, 6))
    x = range(len(run_ids))
    width = 0.25
    
    bars1 = ax.bar([i - width for i in x], hop_1_acc, width, label='1-hop', alpha=0.8)
    bars2 = ax.bar(x, hop_2_acc, width, label='2-hop', alpha=0.8)
    bars3 = ax.bar([i + width for i in x], hop_3_acc, width, label='3-hop', alpha=0.8)
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%',
                       ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Run ID (Model)', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Per-Hop Accuracy Comparison Across Runs', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.set_ylim(0, 100)
    ax.legend(loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'per_hop_accuracy.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return 'per_hop_accuracy.png'


def plot_confusion_matrix(runs_data, output_dir):
    """Plot confusion matrices for each run."""
    confusion_matrices = []
    
    for run in runs_data:
        run_id = run['run_id']
        model_short = run.get('model_short', 'N/A')
        detailed_results = run.get('detailed_results', [])
        
        if not detailed_results:
            continue
        
        # Build confusion matrix
        cm = defaultdict(lambda: defaultdict(int))
        for result in detailed_results:
            # Handle different key names (expected vs expected_hop)
            expected = result.get('expected') or result.get('expected_hop', 0)
            predicted = result.get('predicted') or result.get('predicted_hop', 0)
            cm[expected][predicted] += 1
        
        # Convert to matrix format (1, 2, 3)
        matrix = [[cm[i][j] for j in [1, 2, 3]] for i in [1, 2, 3]]
        
        # Calculate max value for colormap scaling
        max_val = max(max(row) for row in matrix) if matrix and any(any(row) for row in matrix) else 1
        
        # Plot with red colormap (shades of red)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(matrix, annot=True, fmt='d', cmap='Reds', ax=ax,
                   xticklabels=['Predicted 1', 'Predicted 2', 'Predicted 3'],
                   yticklabels=['Actual 1', 'Actual 2', 'Actual 3'],
                   cbar_kws={'label': 'Count'}, vmin=0, vmax=max_val)
        
        ax.set_title(f'Confusion Matrix - {run_id}\n({model_short})', fontsize=14, fontweight='bold')
        ax.set_xlabel('Predicted Hop Count', fontsize=12)
        ax.set_ylabel('Actual Hop Count', fontsize=12)
        
        plt.tight_layout()
        filename = f'confusion_matrix_{run_id}.png'
        plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        confusion_matrices.append({
            'run_id': run_id,
            'model_short': model_short,
            'filename': filename,
            'matrix': matrix
        })
    
    return confusion_matrices


def plot_error_patterns(runs_data, output_dir):
    """Plot error pattern summary (most common misclassifications)."""
    error_patterns = defaultdict(int)
    
    for run in runs_data:
        detailed_results = run.get('detailed_results', [])
        for result in detailed_results:
            if not result.get('correct', True):
                # Handle different key names
                expected = result.get('expected') or result.get('expected_hop', 0)
                predicted = result.get('predicted') or result.get('predicted_hop', 0)
                pattern = f"{expected}-hop → {predicted}-hop"
                error_patterns[pattern] += 1
    
    if not error_patterns:
        return None
    
    # Sort by frequency
    sorted_patterns = sorted(error_patterns.items(), key=lambda x: x[1], reverse=True)
    patterns, counts = zip(*sorted_patterns) if sorted_patterns else ([], [])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(range(len(patterns)), counts, color=sns.color_palette("husl", len(patterns)))
    
    # Add value labels
    for i, (bar, count) in enumerate(zip(bars, counts)):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2.,
               f' {count}',
               ha='left', va='center', fontsize=10)
    
    ax.set_yticks(range(len(patterns)))
    ax.set_yticklabels(patterns)
    ax.set_xlabel('Error Count', fontsize=12)
    ax.set_title('Error Pattern Summary (All Runs)', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'error_patterns.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return 'error_patterns.png'


def create_summary_table(runs_data):
    """Create summary statistics table."""
    table_data = []
    
    for run in runs_data:
        metrics = run['metrics']
        config = run.get('config', {})
        
        table_data.append({
            'Run ID': run['run_id'],
            'Overall Accuracy': f"{metrics.get('overall_accuracy', 0)*100:.2f}%",
            '1-hop Accuracy': f"{metrics.get('hop_1_accuracy', 0)*100:.2f}%",
            '2-hop Accuracy': f"{metrics.get('hop_2_accuracy', 0)*100:.2f}%",
            '3-hop Accuracy': f"{metrics.get('hop_3_accuracy', 0)*100:.2f}%",
            'Model': config.get('model_id', 'N/A'),
            'Total Questions': metrics.get('total_questions', 0),
        })
    
    return table_data


def generate_html_report(runs_data, plots, output_dir):
    """Generate HTML report with embedded plots."""
    
    summary_table = create_summary_table(runs_data)
    
    # Create HTML table from summary
    table_html = "<table border='1' style='border-collapse: collapse; width: 100%; margin: 20px 0;'>\n"
    table_html += "<tr><th>Run ID</th><th>Overall Accuracy</th><th>1-hop</th><th>2-hop</th><th>3-hop</th><th>Model</th><th>Total Questions</th></tr>\n"
    
    for row in summary_table:
        table_html += "<tr>"
        for key in ['Run ID', 'Overall Accuracy', '1-hop Accuracy', '2-hop Accuracy', '3-hop Accuracy', 'Model', 'Total Questions']:
            table_html += f"<td>{row[key]}</td>"
        table_html += "</tr>\n"
    
    table_html += "</table>"
    
    # Confusion matrices HTML
    confusion_html = ""
    for cm_info in plots.get('confusion_matrices', []):
        run_id = cm_info['run_id']
        model_short = cm_info.get('model_short', 'N/A')
        confusion_html += f"""
        <div style='margin: 30px 0;'>
            <h3>Confusion Matrix - {run_id} ({model_short})</h3>
            <img src='{cm_info['filename']}' alt='Confusion Matrix {run_id}' style='max-width: 100%;'>
        </div>
        """
    
    html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Router Component Analysis Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            line-height: 1.6;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
        }}
        img {{
            max-width: 100%;
            height: auto;
            margin: 20px 0;
            border: 1px solid #ddd;
            border-radius: 4px;
        }}
        table {{
            font-size: 14px;
        }}
        th {{
            background-color: #3498db;
            color: white;
            padding: 10px;
            text-align: left;
        }}
        td {{
            padding: 8px;
            border: 1px solid #ddd;
        }}
        tr:nth-child(even) {{
            background-color: #f2f2f2;
        }}
        .section {{
            margin: 40px 0;
            padding: 20px;
            background-color: #f9f9f9;
            border-radius: 5px;
        }}
    </style>
</head>
<body>
    <h1>Router Component Performance Analysis Report</h1>
    <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    <p><strong>Total Runs Analyzed:</strong> {len(runs_data)}</p>
    
    <div class="section">
        <h2>Summary Statistics</h2>
        {table_html}
    </div>
    
    <div class="section">
        <h2>Overall Accuracy Comparison</h2>
        <img src='{plots.get('overall_accuracy', '')}' alt='Overall Accuracy Comparison'>
    </div>
    
    <div class="section">
        <h2>Per-Hop Accuracy Trends</h2>
        <img src='{plots.get('per_hop_accuracy', '')}' alt='Per-Hop Accuracy Comparison'>
    </div>
    
    <div class="section">
        <h2>Error Pattern Summary</h2>
        <img src='{plots.get('error_patterns', '')}' alt='Error Patterns'>
    </div>
    
    <div class="section">
        <h2>Confusion Matrices</h2>
        {confusion_html}
    </div>
    
    <div class="section">
        <h2>Notes</h2>
        <ul>
            <li>Runs in "archived" folders are excluded from analysis</li>
            <li>All accuracy values are percentages</li>
            <li>Confusion matrices show actual vs predicted hop counts</li>
        </ul>
    </div>
</body>
</html>
"""
    
    with open(output_dir / 'report.html', 'w', encoding='utf-8') as f:
        f.write(html_template)
    
    print(f"HTML report saved to: {output_dir / 'report.html'}")


def main():
    parser = argparse.ArgumentParser(description='Analyze run results and generate visualizations')
    parser.add_argument('--runs-dir', type=str, default='runs',
                       help='Path to runs directory (default: runs)')
    parser.add_argument('--component', type=str, default='router',
                       help='Folder under runs/ to analyze: router, average_zero_shot, average_few_shot, etc. (default: router)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for plots and report (default: reports/<component>)')
    parser.add_argument('--skip-archived', action='store_true', default=True,
                       help='Skip folders named "archived" (default: True)')
    
    args = parser.parse_args()

    # Output dir: default reports/<component> so average_zero_shot and average_few_shot don't overwrite
    if args.output_dir is None:
        output_dir = Path("reports") / args.component
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading run data from {args.runs_dir}/{args.component}/...")
    runs_data = load_run_data(args.runs_dir, args.component, args.skip_archived)
    
    if not runs_data:
        print("No run data found!")
        return
    
    print(f"Found {len(runs_data)} runs")
    
    # Generate plots
    print("Generating visualizations...")
    plots = {}
    
    plots['overall_accuracy'] = plot_overall_accuracy(runs_data, output_dir)
    print(f"  [OK] Overall accuracy plot saved")
    
    plots['per_hop_accuracy'] = plot_per_hop_accuracy(runs_data, output_dir)
    print(f"  [OK] Per-hop accuracy plot saved")
    
    plots['confusion_matrices'] = plot_confusion_matrix(runs_data, output_dir)
    print(f"  [OK] Confusion matrices saved ({len(plots['confusion_matrices'])} matrices)")
    
    error_plot = plot_error_patterns(runs_data, output_dir)
    if error_plot:
        plots['error_patterns'] = error_plot
        print(f"  [OK] Error patterns plot saved")
    
    # Generate HTML report
    print("Generating HTML report...")
    generate_html_report(runs_data, plots, output_dir)
    
    print(f"\n[OK] Analysis complete! Output saved to: {output_dir}")
    print(f"  - Open {output_dir / 'report.html'} in a browser to view the report")


if __name__ == '__main__':
    main()
