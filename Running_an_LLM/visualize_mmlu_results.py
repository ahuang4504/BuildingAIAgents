"""
MMLU Model Comparison Visualization Script

This script creates 6 essential visualizations to compare models:
1. Timing Comparison (Wall/CPU/GPU time + cycles)
2. Overall Accuracy
3. Subject Accuracy Heatmap
4. Mistake Overlap Analysis
5. Mistake Correlation Heatmap (answers: do models make same mistakes?)
6. Question Difficulty Distribution

Usage:
    python visualize_mmlu_results.py
"""

import json
import os
import glob
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from scipy.stats import pearsonr

# ============================================================================
# CONFIGURATION
# ============================================================================

# Style settings
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Figure sizes
SINGLE_FIG = (10, 6)
WIDE_FIG = (14, 6)
TALL_FIG = (10, 8)

# Font sizes
TITLE_SIZE = 14
LABEL_SIZE = 12
TICK_SIZE = 10

# Model name mapping for cleaner display
MODEL_DISPLAY_NAMES = {
    'allenai/OLMo-2-0425-1B-Instruct': 'OLMo-1B',
    'meta-llama/Llama-3.2-1B-Instruct': 'Llama-3.2-1B',
    'google/gemma-2-2b-it': 'Gemma-2-2B'
}

# Model colors
MODEL_COLORS = {
    'OLMo-1B': '#FF6B6B',      # Red
    'Llama-3.2-1B': '#4ECDC4', # Teal
    'Gemma-2-2B': '#45B7D1'    # Blue
}

OUTPUT_DIR = Path('visualizations')

# ============================================================================
# DATA LOADING AND PROCESSING
# ============================================================================

def load_all_results():
    """Load all JSON result files and deduplicate by model"""
    json_files = glob.glob('mmlu_multi_model_results_*.json')

    if not json_files:
        raise FileNotFoundError("No MMLU result files found. Run llama_mmlu_eval.py first.")

    print(f"Found {len(json_files)} result files")

    # Store latest result per model
    model_results = {}

    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)

        # Extract model data
        for model_name, model_data in data['models'].items():
            # Use timestamp to keep only latest
            timestamp = data['timestamp']

            if model_name not in model_results:
                model_results[model_name] = (timestamp, model_data)
            else:
                # Keep latest timestamp
                if timestamp > model_results[model_name][0]:
                    model_results[model_name] = (timestamp, model_data)

    # Extract just the model data (drop timestamps)
    results = {model: data for model, (ts, data) in model_results.items()}

    print(f"\nLoaded {len(results)} unique models:")
    for model in results.keys():
        display_name = MODEL_DISPLAY_NAMES.get(model, model)
        print(f"  - {display_name}")

    return results


def extract_timing_data(results):
    """Extract timing metrics for all models"""
    timing_data = []

    for model_name, data in results.items():
        display_name = MODEL_DISPLAY_NAMES.get(model_name, model_name)
        timing = data['timing']

        # Extract wall time, CPU time, GPU time
        eval_phase = timing['phases'].get('evaluation', {})
        wall_time = eval_phase.get('wall_time', 0)
        cpu_time = eval_phase.get('total_cpu_time', 0)

        # GPU active time = wall_time × gpu_active_ratio
        gpu_active_time = 0
        gpu_cycles = 0
        if 'powermetrics' in timing:
            # Calculate actual GPU active time (not just wall time)
            gpu_active_ratio = timing['powermetrics'].get('gpu_active_ratio', {}).get('mean', 0) / 100.0
            gpu_active_time = wall_time * gpu_active_ratio

            if 'gpu_cycles' in timing['powermetrics']:
                gpu_cycles = timing['powermetrics']['gpu_cycles'].get('total_approx', 0)

        timing_data.append({
            'model': display_name,
            'wall_time': wall_time,
            'cpu_time': cpu_time,
            'gpu_active_time': gpu_active_time,  # CHANGED from gpu_time
            'gpu_cycles': gpu_cycles / 1e9  # Convert to billions
        })

    return pd.DataFrame(timing_data)


def extract_accuracy_data(results):
    """Extract overall and subject-level accuracy"""
    overall_acc = []
    subject_acc = []

    for model_name, data in results.items():
        display_name = MODEL_DISPLAY_NAMES.get(model_name, model_name)

        # Overall accuracy
        overall_acc.append({
            'model': display_name,
            'accuracy': data['overall_accuracy']
        })

        # Subject accuracy
        for subject_result in data['subject_results']:
            subject_acc.append({
                'model': display_name,
                'subject': subject_result['subject'],
                'accuracy': subject_result['accuracy']
            })

    return pd.DataFrame(overall_acc), pd.DataFrame(subject_acc)


def build_question_correctness_matrix(results):
    """
    Build a binary matrix: models × questions (1=correct, 0=wrong)

    Returns:
        - correctness_df: DataFrame with models as columns, questions as rows
        - question_metadata: List of dicts with question info (subject, text, correct_answer)
    """
    # Collect all questions in order (assuming same across models)
    first_model = list(results.keys())[0]
    question_metadata = []

    for subject_result in results[first_model]['subject_results']:
        subject = subject_result['subject']
        for q in subject_result['questions']:
            question_metadata.append({
                'subject': subject,
                'question': q['question'],
                'correct_answer': q['correct_answer']
            })

    total_questions = len(question_metadata)

    # Build correctness matrix
    correctness = {}

    for model_name, data in results.items():
        display_name = MODEL_DISPLAY_NAMES.get(model_name, model_name)
        model_correctness = []

        for subject_result in data['subject_results']:
            for q in subject_result['questions']:
                model_correctness.append(1 if q['is_correct'] else 0)

        correctness[display_name] = model_correctness

    correctness_df = pd.DataFrame(correctness)

    print(f"\nBuilt correctness matrix: {total_questions} questions × {len(correctness)} models")

    return correctness_df, question_metadata


# ============================================================================
# VISUALIZATION 1: TIMING COMPARISON
# ============================================================================

def plot_timing_comparison(timing_df):
    """Create grouped bar chart comparing Wall/CPU/GPU Active time"""
    fig, ax = plt.subplots(figsize=WIDE_FIG)

    models = timing_df['model']
    x = np.arange(len(models))
    width = 0.25

    # Plot bars
    bars1 = ax.bar(x - width, timing_df['wall_time'], width,
                   label='Wall Time', color='#3498db', alpha=0.8)
    bars2 = ax.bar(x, timing_df['cpu_time'], width,
                   label='CPU Time', color='#e74c3c', alpha=0.8)
    bars3 = ax.bar(x + width, timing_df['gpu_active_time'], width,
                   label='GPU Active Time', color='#2ecc71', alpha=0.8)

    # Add GPU cycles as text annotations above GPU Active Time bars
    for i, (bar, cycles) in enumerate(zip(bars3, timing_df['gpu_cycles'])):
        if cycles > 0:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                    f'{cycles:.1f}B cycles',
                    ha='center', va='bottom', fontsize=9, style='italic')

    # Formatting
    ax.set_xlabel('Model', fontsize=LABEL_SIZE, fontweight='bold')
    ax.set_ylabel('Time (seconds)', fontsize=LABEL_SIZE, fontweight='bold')
    ax.set_title('Timing Comparison: Wall Time vs CPU Time vs GPU Active Time',
                 fontsize=TITLE_SIZE, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=TICK_SIZE)
    ax.legend(fontsize=LABEL_SIZE)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '1_timing_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Created: 1_timing_comparison.png")


# ============================================================================
# VISUALIZATION 2: OVERALL ACCURACY
# ============================================================================

def plot_overall_accuracy(overall_acc_df):
    """Create bar chart for overall accuracy"""
    fig, ax = plt.subplots(figsize=SINGLE_FIG)

    # Sort by accuracy
    overall_acc_df = overall_acc_df.sort_values('accuracy', ascending=False)

    # Get colors for each model
    colors = [MODEL_COLORS.get(model, '#95a5a6') for model in overall_acc_df['model']]

    bars = ax.bar(overall_acc_df['model'], overall_acc_df['accuracy'],
                  color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    # Add percentage labels on bars
    for bar, acc in zip(bars, overall_acc_df['accuracy']):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{acc:.2f}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Add baseline for random guessing (25%)
    ax.axhline(y=25, color='red', linestyle='--', linewidth=2,
               label='Random Guessing (25%)', alpha=0.7)

    # Formatting
    ax.set_xlabel('Model', fontsize=LABEL_SIZE, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=LABEL_SIZE, fontweight='bold')
    ax.set_title('Overall MMLU Accuracy Comparison',
                 fontsize=TITLE_SIZE, fontweight='bold', pad=20)
    ax.set_ylim(0, max(overall_acc_df['accuracy']) + 10)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '2_overall_accuracy.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Created: 2_overall_accuracy.png")


# ============================================================================
# VISUALIZATION 3: SUBJECT ACCURACY HEATMAP
# ============================================================================

def plot_subject_accuracy_heatmap(subject_acc_df):
    """Create heatmap: models × subjects"""
    # Pivot to get models as rows, subjects as columns
    heatmap_data = subject_acc_df.pivot(index='model', columns='subject', values='accuracy')

    # Sort subjects alphabetically
    heatmap_data = heatmap_data.sort_index(axis=1)

    fig, ax = plt.subplots(figsize=WIDE_FIG)

    # Create heatmap
    sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='RdYlGn',
                cbar_kws={'label': 'Accuracy (%)'},
                linewidths=0.5, linecolor='gray',
                vmin=0, vmax=100, ax=ax)

    # Formatting
    ax.set_xlabel('Subject', fontsize=LABEL_SIZE, fontweight='bold')
    ax.set_ylabel('Model', fontsize=LABEL_SIZE, fontweight='bold')
    ax.set_title('Subject-Wise Accuracy Heatmap',
                 fontsize=TITLE_SIZE, fontweight='bold', pad=20)

    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right', fontsize=TICK_SIZE)
    plt.yticks(rotation=0, fontsize=TICK_SIZE)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '3_subject_accuracy_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Created: 3_subject_accuracy_heatmap.png")


# ============================================================================
# VISUALIZATION 4: MISTAKE OVERLAP ANALYSIS
# ============================================================================

def plot_mistake_overlap(correctness_df, question_metadata):
    """
    Create stacked horizontal bar showing how many questions:
    - All 3 models got correct
    - 2 models got correct
    - 1 model got correct
    - 0 models got correct
    """
    # Count correct answers per question
    correct_counts = correctness_df.sum(axis=1)

    # Count questions in each category
    all_correct = (correct_counts == 3).sum()
    two_correct = (correct_counts == 2).sum()
    one_correct = (correct_counts == 1).sum()
    none_correct = (correct_counts == 0).sum()

    total_questions = len(correctness_df)

    # Create data for stacked bar
    categories = ['All Correct', '2 Correct', '1 Correct', 'None Correct']
    counts = [all_correct, two_correct, one_correct, none_correct]
    percentages = [c / total_questions * 100 for c in counts]
    colors_map = ['#2ecc71', '#f39c12', '#e67e22', '#e74c3c']

    fig, ax = plt.subplots(figsize=WIDE_FIG)

    # Create horizontal stacked bar
    left = 0
    for category, count, pct, color in zip(categories, counts, percentages, colors_map):
        ax.barh(0, count, left=left, color=color, alpha=0.8,
                edgecolor='black', linewidth=1.5)

        # Add label in the middle of segment
        if count > 0:
            ax.text(left + count/2, 0, f'{category}\n{count} ({pct:.1f}%)',
                    ha='center', va='center', fontsize=11, fontweight='bold',
                    color='white' if category == 'None Correct' else 'black')

        left += count

    # Formatting
    ax.set_xlim(0, total_questions)
    ax.set_ylim(-0.5, 0.5)
    ax.set_xlabel('Number of Questions', fontsize=LABEL_SIZE, fontweight='bold')
    ax.set_title('Question Difficulty: How Many Models Got Each Question Correct?',
                 fontsize=TITLE_SIZE, fontweight='bold', pad=20)
    ax.set_yticks([])
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '4_mistake_overlap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Created: 4_mistake_overlap.png")

    # Print statistics
    print(f"\n{'='*70}")
    print("MISTAKE OVERLAP STATISTICS")
    print(f"{'='*70}")
    print(f"All 3 models correct:  {all_correct:4d} questions ({all_correct/total_questions*100:5.1f}%)")
    print(f"2 models correct:      {two_correct:4d} questions ({two_correct/total_questions*100:5.1f}%)")
    print(f"1 model correct:       {one_correct:4d} questions ({one_correct/total_questions*100:5.1f}%)")
    print(f"0 models correct:      {none_correct:4d} questions ({none_correct/total_questions*100:5.1f}%)")
    print(f"{'='*70}\n")


# ============================================================================
# VISUALIZATION 5: MISTAKE CORRELATION HEATMAP
# ============================================================================

def plot_mistake_correlation(correctness_df):
    """
    Create correlation matrix showing if models make similar mistakes
    High correlation = models make same mistakes (systematic)
    Low correlation = models make different mistakes (random)
    """
    # Calculate correlation matrix
    corr_matrix = correctness_df.corr()

    fig, ax = plt.subplots(figsize=SINGLE_FIG)

    # Create heatmap
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='Reds',
                cbar_kws={'label': 'Correlation'},
                linewidths=2, linecolor='black',
                vmin=0, vmax=1, ax=ax, square=True)

    # Formatting
    ax.set_xlabel('Model', fontsize=LABEL_SIZE, fontweight='bold')
    ax.set_ylabel('Model', fontsize=LABEL_SIZE, fontweight='bold')
    ax.set_title('Mistake Correlation Matrix\n(Do models make mistakes on the same questions?)',
                 fontsize=TITLE_SIZE, fontweight='bold', pad=20)

    plt.xticks(rotation=45, ha='right', fontsize=TICK_SIZE)
    plt.yticks(rotation=0, fontsize=TICK_SIZE)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '5_mistake_correlation_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Created: 5_mistake_correlation_heatmap.png")

    # Print correlation statistics
    print(f"\n{'='*70}")
    print("MISTAKE CORRELATION ANALYSIS")
    print(f"{'='*70}")
    print("Correlation coefficients (1.0 = identical mistakes, 0.0 = independent):\n")

    models = list(correctness_df.columns)
    for i in range(len(models)):
        for j in range(i+1, len(models)):
            corr = corr_matrix.iloc[i, j]
            print(f"{models[i]:15s} ↔ {models[j]:15s}: {corr:.3f}")

            # Interpret correlation
            if corr > 0.7:
                interpretation = "SYSTEMATIC - Models make very similar mistakes"
            elif corr > 0.5:
                interpretation = "MODERATE - Some shared patterns"
            elif corr > 0.3:
                interpretation = "WEAK - Mostly different mistakes"
            else:
                interpretation = "RANDOM - Independent mistake patterns"

            print(f"    → {interpretation}\n")

    print(f"{'='*70}\n")


# ============================================================================
# VISUALIZATION 6: QUESTION DIFFICULTY DISTRIBUTION
# ============================================================================

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("\n" + "="*70)
    print("MMLU MODEL COMPARISON VISUALIZATION")
    print("="*70 + "\n")

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Load data
    print("Loading data...")
    results = load_all_results()

    # Extract metrics
    print("\nExtracting metrics...")
    timing_df = extract_timing_data(results)
    overall_acc_df, subject_acc_df = extract_accuracy_data(results)
    correctness_df, question_metadata = build_question_correctness_matrix(results)

    # Generate visualizations
    print("\nGenerating visualizations...")
    print("-" * 70)

    plot_timing_comparison(timing_df)
    plot_overall_accuracy(overall_acc_df)
    plot_subject_accuracy_heatmap(subject_acc_df)
    plot_mistake_overlap(correctness_df, question_metadata)
    plot_mistake_correlation(correctness_df)

    print("-" * 70)
    print(f"\n✅ 5 visualizations saved to: {OUTPUT_DIR.absolute()}/")
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
