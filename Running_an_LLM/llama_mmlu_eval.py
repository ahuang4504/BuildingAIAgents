"""
Llama 3.2-1B MMLU Evaluation Script (Laptop Optimized with Quantization)

This script evaluates Llama 3.2-1B on the MMLU benchmark.
Optimized for laptops with 4-bit or 8-bit quantization to reduce memory usage.

Quantization options:
- 4-bit: ~1.5 GB VRAM/RAM (default for laptop)
- 8-bit: ~2.5 GB VRAM/RAM
- No quantization: ~5 GB VRAM/RAM

Usage:
1. Install: pip install transformers torch datasets accelerate tqdm bitsandbytes
2. Login: huggingface-cli login
3. Run: python llama_mmlu_eval_quantized.py

Set QUANTIZATION_BITS below to choose quantization level.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
import json
from tqdm.auto import tqdm
import os
from datetime import datetime
import sys
import platform
import argparse
import time
import resource
import gc
import subprocess
import signal
import atexit
import re
from contextlib import contextmanager
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Multi-Model MMLU Evaluation')
parser.add_argument('--verbose', action='store_true', help='Print each question and answer')
args = parser.parse_args()

# ============================================================================
# CONFIGURATION - Modify these settings
# ============================================================================

# Multi-model evaluation
MODELS_TO_TEST = [
    # "meta-llama/Llama-3.2-1B-Instruct",
    "google/gemma-2-2b-it",
    # "allenai/OLMo-2-0425-1B-Instruct"
]

# Legacy single model name (for backwards compatibility)
MODEL_NAME = MODELS_TO_TEST[0]

# GPU settings
# If True, will attempt to use the best available GPU (CUDA for NVIDIA, MPS for Apple Silicon)
# If False, will always use CPU regardless of available hardware
USE_GPU = True  # Set to False to force CPU-only execution

MAX_NEW_TOKENS = 1

# Verbose mode - print each question, answer, and correctness
# Can be overridden by --verbose CLI flag
VERBOSE = args.verbose if args.verbose else False

# Quantization settings
# Options: 4, 8, or None (default is None for full precision)
#
# To enable quantization, change QUANTIZATION_BITS to one of the following:
#   QUANTIZATION_BITS = 4   # 4-bit quantization: ~1.5 GB memory (most memory efficient)
#   QUANTIZATION_BITS = 8   # 8-bit quantization: ~2.5 GB memory (balanced quality/memory)
#   QUANTIZATION_BITS = None  # No quantization: ~5 GB memory (full precision, best quality)
#
# Notes:
# - Quantization requires the 'bitsandbytes' package: pip install bitsandbytes
# - Quantization only works with CUDA (NVIDIA GPUs), not with Apple Metal (MPS)
# - If using Apple Silicon, quantization will be automatically disabled

QUANTIZATION_BITS = None  # Change to 4 or 8 to enable quantization

# MMLU Subjects to test (10 subjects for comprehensive evaluation)
MMLU_SUBJECTS = [
    "astronomy", "business_ethics",
    "clinical_knowledge", "college_biology", "college_chemistry",
    "college_computer_science", "college_mathematics", "college_medicine",
    "college_physics", "computer_security"
]

# Additional subjects available (uncomment to add more):
# MMLU_SUBJECTS += [
#     "conceptual_physics", "econometrics", "electrical_engineering",
    # "formal_logic", "global_facts", "high_school_biology",
    # "high_school_chemistry", "high_school_computer_science",
    # "high_school_european_history", "high_school_geography",
    # "high_school_government_and_politics", "high_school_macroeconomics",
    # "high_school_mathematics", "high_school_microeconomics",
    # "high_school_physics", "high_school_psychology", "high_school_statistics",
    # "high_school_us_history", "high_school_world_history", "human_aging",
    # "human_sexuality", "international_law", "jurisprudence",
    # "logical_fallacies", "machine_learning", "management", "marketing",
    # "medical_genetics", "miscellaneous", "moral_disputes", "moral_scenarios",
    # "nutrition", "philosophy", "prehistory", "professional_accounting",
    # "professional_law", "professional_medicine", "professional_psychology",
    # "public_relations", "security_studies", "sociology", "us_foreign_policy",
    # "virology", "world_religions"
# ]


# ============================================================================
# TIMING UTILITIES
# ============================================================================

class EvaluationTimer:
    """Robust timing for ML model evaluation with MPS support"""

    def __init__(self, device):
        self.device = device
        self.is_mps = (str(device) == "mps")
        self.phase_times = {}
        self.inference_times = []  # Per-question inference times

    def get_snapshot(self):
        """Capture current timing and resource state"""
        usage = resource.getrusage(resource.RUSAGE_SELF)
        return {
            'wall': time.perf_counter(),
            'user_cpu': usage.ru_utime,
            'sys_cpu': usage.ru_stime
        }

    def calculate_diff(self, start, end):
        """Calculate elapsed times from snapshots"""
        wall_time = end['wall'] - start['wall']
        user_cpu = end['user_cpu'] - start['user_cpu']
        sys_cpu = end['sys_cpu'] - start['sys_cpu']
        total_cpu = user_cpu + sys_cpu

        return {
            'wall_time': wall_time,
            'user_cpu_time': user_cpu,
            'system_cpu_time': sys_cpu,
            'total_cpu_time': total_cpu,
            'cpu_utilization': (total_cpu / wall_time * 100) if wall_time > 0 else 0
        }

    @contextmanager
    def phase(self, name):
        """Context manager for timing a phase (e.g., model loading)"""
        if self.is_mps:
            torch.mps.synchronize()

        start = self.get_snapshot()
        try:
            yield
        finally:
            if self.is_mps:
                torch.mps.synchronize()
            end = self.get_snapshot()
            self.phase_times[name] = self.calculate_diff(start, end)

    def time_inference(self, func, *args, **kwargs):
        """Time a single inference with MPS synchronization"""
        if self.is_mps:
            torch.mps.synchronize()

        start = time.perf_counter()
        result = func(*args, **kwargs)

        if self.is_mps:
            torch.mps.synchronize()

        elapsed = time.perf_counter() - start
        self.inference_times.append(elapsed)
        return result

    def get_summary(self):
        """Get complete timing summary with statistics"""
        import statistics

        summary = {'phases': self.phase_times}

        if self.inference_times:
            summary['inference'] = {
                'count': len(self.inference_times),
                'mean': statistics.mean(self.inference_times),
                'median': statistics.median(self.inference_times),
                'min': min(self.inference_times),
                'max': max(self.inference_times),
                'stdev': statistics.stdev(self.inference_times) if len(self.inference_times) > 1 else 0
            }

            # Add percentiles if we have enough samples
            if len(self.inference_times) >= 10:
                import math
                sorted_times = sorted(self.inference_times)
                p95_idx = max(0, math.ceil(len(sorted_times) * 0.95) - 1)
                p99_idx = max(0, math.ceil(len(sorted_times) * 0.99) - 1)
                summary['inference']['p95'] = sorted_times[p95_idx]
                summary['inference']['p99'] = sorted_times[p99_idx]

        return summary


# ============================================================================
# POWERMETRICS PROFILING
# ============================================================================

class PowerMetricsMonitor:
    """Monitor GPU/CPU metrics using macOS powermetrics subprocess"""

    def __init__(self):
        self.process = None
        self.log_file = None
        self.metrics = {
            'gpu_active_ratio': [],
            'gpu_freq_mhz': [],
            'gpu_power_mw': [],
            'e_cluster_freq': [],
            'p_cluster_freq': [],
            'cycles': [],
            'instructions': []
        }
        self.running = False
        self.start_time = None
        self.end_time = None

    def start(self):
        """Start powermetrics monitoring in background"""
        if self.running:
            return

        # Create log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = f"powermetrics_{timestamp}.log"

        try:
            # Start powermetrics subprocess
            # Requires sudo - will prompt for password before Python script starts
            cmd = [
                'sudo', 'powermetrics',
                '--samplers', 'gpu_power,cpu_power,tasks',
                '-i', '1000',  # Sample every 1000ms
                '-o', self.log_file
            ]

            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,  # Suppress "Second underflow" warnings
                text=True
            )

            self.running = True
            self.start_time = time.perf_counter()

            # Register cleanup handler
            atexit.register(self.stop)

            print(f"✓ PowerMetrics monitoring started (log: {self.log_file})")
            return True

        except Exception as e:
            print(f"⚠️  PowerMetrics failed to start: {e}")
            return False

    def stop(self):
        """Stop monitoring and parse results"""
        if not self.running:
            return None

        self.running = False
        self.end_time = time.perf_counter()

        # Stop powermetrics gracefully
        if self.process:
            try:
                # Send SIGINT (Ctrl+C) to powermetrics
                self.process.send_signal(signal.SIGINT)
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # Force kill if it doesn't stop
                self.process.kill()
                self.process.wait()
            except Exception:
                pass

        # Parse the log file
        if self.log_file and os.path.exists(self.log_file):
            self._parse_log()
            summary = self._summarize()
            return summary

        return None

    def _parse_log(self):
        """Parse powermetrics log file"""
        try:
            with open(self.log_file, 'r') as f:
                for line in f:
                    # GPU Active Residency (HW active residency)
                    match = re.search(r'GPU HW active residency:\s+([\d.]+)%', line)
                    if match:
                        self.metrics['gpu_active_ratio'].append(float(match.group(1)))

                    # GPU Frequency (HW active frequency)
                    match = re.search(r'GPU HW active frequency:\s+([\d.]+)\s+MHz', line)
                    if match:
                        self.metrics['gpu_freq_mhz'].append(float(match.group(1)))

                    # GPU Power
                    match = re.search(r'GPU Power:\s+([\d.]+)\s+mW', line)
                    if match:
                        self.metrics['gpu_power_mw'].append(float(match.group(1)))

                    # E-Cluster Frequency
                    match = re.search(r'E-Cluster HW active frequency:\s+([\d.]+)\s+MHz', line)
                    if match:
                        self.metrics['e_cluster_freq'].append(float(match.group(1)))

                    # P-Cluster Frequency
                    match = re.search(r'P-Cluster HW active frequency:\s+([\d.]+)\s+MHz', line)
                    if match:
                        self.metrics['p_cluster_freq'].append(float(match.group(1)))

                    # CPU Cycles
                    match = re.search(r'cycles:\s+([\d.]+)([KMG]?)', line)
                    if match:
                        value = float(match.group(1))
                        suffix = match.group(2)
                        multipliers = {'K': 1e3, 'M': 1e6, 'G': 1e9, '': 1}
                        self.metrics['cycles'].append(value * multipliers.get(suffix, 1))

                    # CPU Instructions
                    match = re.search(r'instructions:\s+([\d.]+)([KMG]?)', line)
                    if match:
                        value = float(match.group(1))
                        suffix = match.group(2)
                        multipliers = {'K': 1e3, 'M': 1e6, 'G': 1e9, '': 1}
                        self.metrics['instructions'].append(value * multipliers.get(suffix, 1))

        except Exception as e:
            print(f"⚠️  Error parsing powermetrics log: {e}")

    def _summarize(self):
        """Calculate summary statistics including cycle calculations"""
        import statistics

        summary = {}

        for key, values in self.metrics.items():
            if values:
                summary[key] = {
                    'mean': statistics.mean(values),
                    'median': statistics.median(values),
                    'min': min(values),
                    'max': max(values),
                    'stdev': statistics.stdev(values) if len(values) > 1 else 0,
                    'total': sum(values)
                }

        # Calculate GPU cycles (approximate)
        # Formula: GPU_Cycles ≈ GPU_Frequency × Time × (Active_Ratio / 100)
        if self.start_time and self.end_time:
            duration_sec = self.end_time - self.start_time
            summary['evaluation_duration_sec'] = duration_sec

            # GPU Cycle Calculation
            if 'gpu_freq_mhz' in summary and 'gpu_active_ratio' in summary:
                avg_freq_hz = summary['gpu_freq_mhz']['mean'] * 1e6  # Convert MHz to Hz
                avg_active_ratio = summary['gpu_active_ratio']['mean'] / 100.0  # Convert % to ratio

                # Approximate GPU cycles = frequency * time * activity ratio
                gpu_cycles_approx = avg_freq_hz * duration_sec * avg_active_ratio

                summary['gpu_cycles'] = {
                    'total_approx': gpu_cycles_approx,
                    'formula': 'avg_freq_hz × duration_sec × avg_active_ratio',
                    'note': 'Approximate - based on sampled frequency and activity',
                    'avg_freq_hz': avg_freq_hz,
                    'duration_sec': duration_sec,
                    'avg_active_ratio': avg_active_ratio
                }

            # CPU Cycle Summary (already captured in 'cycles' metric)
            if 'cycles' in summary:
                total_cpu_cycles = summary['cycles']['total']
                summary['cpu_cycles'] = {
                    'total': total_cpu_cycles,
                    'per_second': total_cpu_cycles / duration_sec if duration_sec > 0 else 0,
                    'note': 'Total CPU cycles from powermetrics task sampling'
                }

        return summary


# ============================================================================
# DEVICE DETECTION
# ============================================================================

def detect_device():
    """Detect the best available device (CUDA, MPS, or CPU)"""

    # If GPU is disabled, always use CPU
    if not USE_GPU:
        return "cpu"

    # Check for CUDA
    if torch.cuda.is_available():
        return "cuda"

    # Check for Apple Silicon with Metal
    if torch.backends.mps.is_available():
        # Check if we're actually on Apple ARM
        is_apple_arm = platform.system() == "Darwin" and platform.processor() == "arm"

        if is_apple_arm:
            # Metal is available but incompatible with quantization
            if QUANTIZATION_BITS is not None:
                print("\n" + "="*70)
                print("ERROR: Metal and Quantization Conflict")
                print("="*70)
                print("Metal Performance Shaders (MPS) is incompatible with quantization.")
                print(f"You have USE_GPU = True and QUANTIZATION_BITS = {QUANTIZATION_BITS}")
                print("")
                print("Please choose one of the following options:")
                print("  1. Set USE_GPU = False to use CPU with quantization")
                print("  2. Set QUANTIZATION_BITS = None to use Metal without quantization")
                print("="*70 + "\n")
                sys.exit(1)
            return "mps"

    # Default to CPU
    return "cpu"




def check_environment():
    global QUANTIZATION_BITS
    """Check environment and dependencies"""
    print("="*70)
    print("Environment Check")
    print("="*70)

    # Check if in Colab
    try:
        import google.colab
        print("✓ Running in Google Colab")
        in_colab = True
    except:
        print("✓ Running locally (not in Colab)")
        in_colab = False

    # Check system info
    print(f"✓ Platform: {platform.system()} ({platform.machine()})")
    if platform.system() == "Darwin":
        print(f"✓ Processor: {platform.processor()}")

    # Detect and set device
    device = detect_device()

    # Check device
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✓ GPU Available: {gpu_name}")
        print(f"✓ GPU Memory: {gpu_memory:.2f} GB")
    elif device == "mps":
        print("✓ Apple Metal (MPS) Available")
        print("✓ Using Metal Performance Shaders for GPU acceleration")
    else:
        print("⚠️  No GPU detected - running on CPU")
       
    # Check quantization support

    if QUANTIZATION_BITS is not None:
        try:
            import bitsandbytes
            print(f"✓ bitsandbytes installed - {QUANTIZATION_BITS}-bit quantization available")
        except ImportError:
            print(f"❌ bitsandbytes NOT installed - cannot use quantization")
            sys.exit(1)
        if device == 'mps':
            print(f"❌ Apple METAL is incompatible with quantization")
            print("✓ Quantization disabled - loading full precision model")
            QUANTIZATION_BITS = None
            sys.exit(1)
    else:
        print("✓ Quantization disabled - loading full precision model")
    
    # Check HF authentication
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        print("✓ Hugging Face token loaded from .env file")
    else:
        # Fallback to CLI token
        try:
            from huggingface_hub import HfFolder
            token = HfFolder.get_token()
            if token:
                print("✓ Hugging Face authenticated via CLI")
            else:
                print("⚠️  No Hugging Face token found")
                print("Add HF_TOKEN to .env file or run: huggingface-cli login")
        except:
            print("⚠️  Could not check Hugging Face authentication")
    
    # Print configuration
    print("\n" + "="*70)
    print("Configuration")
    print("="*70)
    print(f"Models: {len(MODELS_TO_TEST)} models to evaluate")
    for i, model in enumerate(MODELS_TO_TEST, 1):
        print(f"  {i}. {model}")
    print(f"Device: {device}")
    if QUANTIZATION_BITS is not None:
        print(f"Quantization: {QUANTIZATION_BITS}-bit")
        if QUANTIZATION_BITS == 4:
            print(f"Expected memory: ~1.5 GB")
        elif QUANTIZATION_BITS == 8:
            print(f"Expected memory: ~2.5 GB")
    else:
        print(f"Quantization: None (full precision)")
        if device == "cuda":
            print(f"Expected memory: ~2.5 GB (FP16)")
        elif device == "mps":
            print(f"Expected memory: ~2.5 GB (FP16)")
        else:
            print(f"Expected memory: ~5 GB (FP32)")
    print(f"Number of subjects: {len(MMLU_SUBJECTS)}")

    print("="*70 + "\n")
    return in_colab, device


def get_quantization_config():
    """Create quantization config based on settings"""
    if QUANTIZATION_BITS is None:
        return None
    
    if QUANTIZATION_BITS == 4:
        # 4-bit quantization (most memory efficient)
        config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,  # Double quantization for extra compression
            bnb_4bit_quant_type="nf4"  # NormalFloat4 - better for LLMs
        )
        print("Using 4-bit quantization (NF4 + double quant)")
        print("Memory usage: ~1.5 GB")
    elif QUANTIZATION_BITS == 8:
        # 8-bit quantization (balanced)
        config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False
        )
        print("Using 8-bit quantization")
        print("Memory usage: ~2.5 GB")
    else:
        raise ValueError(f"Invalid QUANTIZATION_BITS: {QUANTIZATION_BITS}. Use 4, 8, or None")
    
    return config


def load_model_and_tokenizer(device, model_name=None):
    """Load model with optional quantization"""
    if model_name is None:
        model_name = model_name  # Fallback to original

    # Force garbage collection before loading new model
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()

    print(f"\nLoading model {model_name}...")
    print(f"Device: {device}")

    try:
        # Get HF token from environment
        hf_token = os.getenv("HF_TOKEN")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        print("✓ Tokenizer loaded")

        # Get quantization config
        quant_config = get_quantization_config()

        # Load model
        print("Loading model (this may take 2-3 minutes)...")

        if quant_config is not None:
            # Quantized model loading (only works with CUDA)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quant_config,
                device_map="auto",
                low_cpu_mem_usage=True,
                token=hf_token
            )
        else:
            # Non-quantized model loading
            if device == "cuda":
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    dtype=torch.float16,
                    device_map="auto",
                    low_cpu_mem_usage=True,
                    token=hf_token
                )
            elif device == "mps":
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    dtype=torch.float16,
                    low_cpu_mem_usage=True,
                    token=hf_token
                )
                model = model.to(device)
            else:  # CPU
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    dtype=torch.float32,
                    low_cpu_mem_usage=True,
                    token=hf_token
                )
                model = model.to(device)

        model.eval()

        # Print model info
        print("✓ Model loaded successfully!")
        print(f"  Model device: {next(model.parameters()).device}")
        print(f"  Model dtype: {next(model.parameters()).dtype}")

        # Check memory usage
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated(0) / 1e9
            memory_reserved = torch.cuda.memory_reserved(0) / 1e9
            print(f"  GPU Memory: {memory_allocated:.2f} GB allocated, {memory_reserved:.2f} GB reserved")

            # Check if using quantization
            if quant_config is not None:
                print(f"  Quantization: {QUANTIZATION_BITS}-bit active")
        elif device == "mps":
            print(f"  Running on Apple Metal (MPS)")

        return model, tokenizer
        
    except Exception as e:
        print(f"\n❌ Error loading model: {e}")
        print("\nPossible causes:")
        print("1. No Hugging Face token - Run: huggingface-cli login")
        print("2. Llama license not accepted - Visit: https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct")
        print("3. bitsandbytes not installed - Run: pip install bitsandbytes")
        print("4. Out of memory - Try 4-bit quantization or smaller model")
        raise


def format_mmlu_prompt(question, choices):
    """Format MMLU question as multiple choice"""
    choice_labels = ["A", "B", "C", "D"]
    prompt = f"{question}\n\n"
    for label, choice in zip(choice_labels, choices):
        prompt += f"{label}. {choice}\n"
    prompt += "\nAnswer:"
    return prompt


def get_model_prediction(model, tokenizer, prompt, timer=None):
    """Get model's prediction with optional timing"""
    # Prepare inputs (not part of timed inference)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    def _inference():
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False,
                temperature=1.0
            )
        return outputs

    # Use timer if provided (for per-question timing)
    if timer:
        outputs = timer.time_inference(_inference)
    else:
        outputs = _inference()

    # Decode output
    generated_text = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:],
        skip_special_tokens=True
    )

    # Extract answer
    answer = generated_text.strip()[:1].upper()
    if answer not in ["A", "B", "C", "D"]:
        for char in generated_text.upper():
            if char in ["A", "B", "C", "D"]:
                answer = char
                break
        else:
            answer = "A"

    # Clear tensors to free memory
    del inputs, outputs

    return answer


def evaluate_subject(model, tokenizer, subject, timer=None, verbose=False):
    """Evaluate model on a specific MMLU subject with optional timing"""
    print(f"\n{'='*70}")
    print(f"Evaluating subject: {subject}")
    print(f"{'='*70}")

    try:
        dataset = load_dataset("cais/mmlu", subject, split="test")
    except Exception as e:
        print(f"❌ Error loading subject {subject}: {e}")
        return None

    correct = 0
    total = 0
    question_details = []  # Store per-question results

    for example in tqdm(dataset, desc=f"Testing {subject}", leave=True):
        question = example["question"]
        choices = example["choices"]
        correct_answer_idx = example["answer"]
        correct_answer = ["A", "B", "C", "D"][correct_answer_idx]

        prompt = format_mmlu_prompt(question, choices)
        predicted_answer = get_model_prediction(model, tokenizer, prompt, timer=timer)

        is_correct = predicted_answer == correct_answer
        if is_correct:
            correct += 1
        total += 1

        # Store per-question details
        question_details.append({
            "question": question,
            "choices": {
                "A": choices[0],
                "B": choices[1],
                "C": choices[2],
                "D": choices[3]
            },
            "predicted_answer": predicted_answer,
            "correct_answer": correct_answer,
            "is_correct": is_correct
        })

        # Verbose output
        if verbose:
            status = "✓" if is_correct else "✗"
            print(f"\n{status} Q{total}: {question}")
            print(f"   Choices: {', '.join([f'{lbl}={ch}' for lbl, ch in zip(['A','B','C','D'], choices)])}")
            print(f"   Model answer: {predicted_answer} | Correct: {correct_answer}")

    accuracy = (correct / total * 100) if total > 0 else 0
    print(f"✓ Result: {correct}/{total} correct = {accuracy:.2f}%")

    # Clear dataset from memory
    del dataset
    gc.collect()

    # Clear MPS cache after each subject to prevent memory buildup
    if str(model.device) == "mps":
        torch.mps.empty_cache()

    return {
        "subject": subject,
        "correct": correct,
        "total": total,
        "accuracy": accuracy,
        "questions": question_details
    }


def print_comparative_summary(all_model_results, device):
    """Print comparison of all models with detailed timing breakdown"""
    print("\n" + "="*70)
    print("COMPARATIVE EVALUATION SUMMARY")
    print("="*70)
    print(f"Device: {device.upper()}")
    print(f"Subjects tested: {len(MMLU_SUBJECTS)}")
    print(f"Verbose mode: {'ON' if VERBOSE else 'OFF'}")
    print("="*70)

    for model_name, data in all_model_results.items():
        timing = data['timing']

        print(f"\n{'─'*70}")
        print(f"Model: {model_name}")
        print(f"{'─'*70}")

        # Accuracy
        print(f"Accuracy: {data['accuracy']:.2f}% ({data['total_correct']}/{data['total_questions']})")

        # Phase timing breakdown
        print(f"\nPhase Timing:")
        for phase_name, phase_data in timing['phases'].items():
            print(f"  {phase_name.replace('_', ' ').title()}:")
            print(f"    Wall time:    {phase_data['wall_time']:7.2f}s  ({phase_data['wall_time']/60:.2f} min)")
            print(f"    User CPU:     {phase_data['user_cpu_time']:7.2f}s")
            print(f"    System CPU:   {phase_data['system_cpu_time']:7.2f}s")
            print(f"    Total CPU:    {phase_data['total_cpu_time']:7.2f}s")
            print(f"    CPU usage:    {phase_data['cpu_utilization']:7.1f}%")

        # PowerMetrics GPU/CPU data if available
        if 'powermetrics' in timing:
            pm = timing['powermetrics']

            print(f"\n  {'─'*66}")
            print(f"  GPU METRICS (Apple Silicon MPS)")
            print(f"  {'─'*66}")

            # GPU Time
            if 'evaluation_duration_sec' in pm:
                duration = pm['evaluation_duration_sec']
                print(f"    GPU Time:     {duration:7.2f} sec  ({duration/60:7.2f} min)")

            # GPU Cycles (Approximate)
            if 'gpu_cycles' in pm:
                gc = pm['gpu_cycles']
                print(f"\n    GPU Cycles (Approximate):")
                print(f"      Total:      {gc['total_approx']/1e9:7.2f} billion cycles")
                print(f"      Formula:    {gc['formula']}")
                print(f"      Avg Freq:   {gc['avg_freq_hz']/1e6:7.0f} MHz")
                print(f"      Duration:   {gc['duration_sec']:7.2f} sec")
                print(f"      Activity:   {gc['avg_active_ratio']*100:7.1f}%")
                print(f"      Note:       {gc['note']}")

            # GPU Frequency & Activity
            print(f"\n    GPU Frequency & Activity:")
            if 'gpu_freq_mhz' in pm:
                s = pm['gpu_freq_mhz']
                print(f"      Frequency:  {s['mean']:7.0f} MHz (avg)  {s['max']:7.0f} MHz (peak)")

            if 'gpu_active_ratio' in pm:
                s = pm['gpu_active_ratio']
                print(f"      Active:     {s['mean']:7.1f}% (avg)  {s['max']:7.1f}% (peak)")

            # GPU Power
            if 'gpu_power_mw' in pm:
                s = pm['gpu_power_mw']
                print(f"\n    GPU Power:")
                print(f"      Average:    {s['mean']/1000:7.2f} W")
                print(f"      Peak:       {s['max']/1000:7.2f} W")

            print(f"\n  {'─'*66}")
            print(f"  CPU METRICS")
            print(f"  {'─'*66}")

            # CPU Time (from phase timing)
            if 'evaluation' in timing['phases']:
                cpu_time = timing['phases']['evaluation']['total_cpu_time']
                print(f"    CPU Time:     {cpu_time:7.2f} sec  ({cpu_time/60:7.2f} min)")

            # CPU Cycles
            if 'cpu_cycles' in pm:
                cc = pm['cpu_cycles']
                print(f"\n    CPU Cycles:")
                print(f"      Total:      {cc['total']/1e9:7.2f} billion cycles")
                print(f"      Per Second: {cc['per_second']/1e9:7.2f} billion cycles/sec")
                print(f"      Note:       {cc['note']}")
            elif 'cycles' in pm:
                # Fallback if cpu_cycles summary not available
                s = pm['cycles']
                print(f"\n    CPU Cycles:")
                print(f"      Total:      {s['total']/1e9:7.2f} billion cycles")

            # CPU Instructions & IPC
            if 'instructions' in pm:
                s = pm['instructions']
                print(f"\n    CPU Instructions:")
                print(f"      Total:      {s['total']/1e9:7.2f} billion instructions")

                # Calculate IPC if both available
                if 'cycles' in pm:
                    ipc = pm['instructions']['mean'] / pm['cycles']['mean']
                    print(f"      IPC:        {ipc:7.2f}")

            # CPU Cluster Frequencies
            print(f"\n    CPU Cluster Frequencies:")
            if 'e_cluster_freq' in pm:
                s = pm['e_cluster_freq']
                print(f"      E-Cluster:  {s['mean']:7.0f} MHz (avg)  {s['max']:7.0f} MHz (peak)")

            if 'p_cluster_freq' in pm:
                s = pm['p_cluster_freq']
                print(f"      P-Cluster:  {s['mean']:7.0f} MHz (avg)  {s['max']:7.0f} MHz (peak)")

            print(f"  {'─'*66}")

        # Per-question inference statistics
        if 'inference' in timing:
            inf = timing['inference']
            print(f"\nPer-Question Inference Stats ({inf['count']} questions):")
            print(f"  Mean:         {inf['mean']*1000:7.2f}ms")
            print(f"  Median:       {inf['median']*1000:7.2f}ms")
            print(f"  Min:          {inf['min']*1000:7.2f}ms")
            print(f"  Max:          {inf['max']*1000:7.2f}ms")
            print(f"  Std Dev:      {inf['stdev']*1000:7.2f}ms")
            if 'p95' in inf:
                print(f"  95th %ile:    {inf['p95']*1000:7.2f}ms")
            if 'p99' in inf:
                print(f"  99th %ile:    {inf['p99']*1000:7.2f}ms")

            # Throughput
            if 'evaluation' in timing['phases']:
                total_wall = timing['phases']['evaluation']['wall_time']
                if total_wall > 0:
                    throughput = inf['count'] / total_wall
                    print(f"  Throughput:   {throughput:.2f} questions/sec")

    print(f"\n{'='*70}")


def save_results(all_model_results):
    """Save results for all models to JSON file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"mmlu_multi_model_results_{timestamp}.json"

    output_data = {
        "timestamp": timestamp,
        "models": {}
    }

    for model_name, data in all_model_results.items():
        output_data["models"][model_name] = {
            "model": model_name,
            "overall_accuracy": data["accuracy"],
            "total_correct": data["total_correct"],
            "total_questions": data["total_questions"],
            "subject_results": data["results"],
            "timing": {
                "phases": data["timing"]["phases"],
                "inference_stats": data["timing"].get("inference", {}),
                "powermetrics": data["timing"].get("powermetrics", {})
            }
        }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n✓ Results saved to: {output_file}")
    return output_file


def main():
    """Main evaluation function with robust timing"""
    # Disable gradient computation globally (we're not training)
    torch.set_grad_enabled(False)

    print("\n" + "="*70)
    print("Multi-Model MMLU Evaluation")
    print("="*70 + "\n")

    # Check environment
    in_colab, device = check_environment()

    # Initialize powermetrics monitor if on MPS
    powermetrics = None
    if device == "mps":
        print("\n" + "="*70)
        print("GPU Profiling Setup")
        print("="*70)
        print("PowerMetrics requires sudo access for GPU profiling.")
        print("You will be prompted for your password once.")
        print("")
        # Pre-validate sudo access
        try:
            subprocess.run(['sudo', '-v'], check=True)
            powermetrics = PowerMetricsMonitor()
        except Exception as e:
            print(f"⚠️  PowerMetrics unavailable: {e}")
            print("Continuing without GPU profiling...")
        print("="*70 + "\n")

    # Store results for all models
    all_model_results = {}

    # Loop through each model
    for model_name in MODELS_TO_TEST:
        print(f"\n{'='*70}")
        print(f"EVALUATING MODEL: {model_name}")
        print(f"{'='*70}\n")

        # Create timer for this model
        timer = EvaluationTimer(device)

        # Time model loading phase
        with timer.phase("model_loading"):
            model, tokenizer = load_model_and_tokenizer(device, model_name)

        # Evaluate on all subjects
        results = []
        total_correct = 0
        total_questions = 0

        # Start powermetrics ONLY for evaluation phase
        if powermetrics:
            powermetrics.start()

        # Time evaluation phase (includes all subjects)
        with timer.phase("evaluation"):
            for i, subject in enumerate(MMLU_SUBJECTS, 1):
                print(f"\nProgress: {i}/{len(MMLU_SUBJECTS)} subjects")
                result = evaluate_subject(model, tokenizer, subject, timer=timer, verbose=VERBOSE)
                if result:
                    results.append(result)
                    total_correct += result["correct"]
                    total_questions += result["total"]

        # Stop powermetrics and get results
        pm_summary = None
        if powermetrics:
            pm_summary = powermetrics.stop()

        # Get timing summary
        timing_summary = timer.get_summary()

        # Add powermetrics data to timing summary
        if pm_summary:
            timing_summary['powermetrics'] = pm_summary

        # Store results
        overall_accuracy = (total_correct / total_questions * 100) if total_questions > 0 else 0
        all_model_results[model_name] = {
            "results": results,
            "total_correct": total_correct,
            "total_questions": total_questions,
            "accuracy": overall_accuracy,
            "timing": timing_summary
        }

        # Aggressive memory cleanup between models
        del model, tokenizer, timer
        gc.collect()

        if device == "mps":
            torch.mps.empty_cache()
        elif device == "cuda":
            torch.cuda.empty_cache()

        # Force another garbage collection after cache clear
        gc.collect()

        print(f"\n✓ Memory cleaned up after {model_name}")

    # Print comparative summary
    print_comparative_summary(all_model_results, device)

    # Save results
    output_file = save_results(all_model_results)

    print("\n✅ Evaluation complete!")
    return output_file


if __name__ == "__main__":
    try:
        output_file = main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Evaluation interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
