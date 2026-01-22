Q4: Time the code using the time shell command line function. Compare the timings for the following setups:

I decided to use the 'time' CLI tool. Of the following reported numbers, user refers to the time executing the Python code like with model inference, system refers to time on I/O and system calls like loading files, downloading model weights, and disk I/O, and, total which is the total elapsed real world time.

1. Using GPU and no quantization.
   50.14s user 10.00s system 62% cpu 1:36.45 total
   (total time includes GPU work + I/O so we expect to see the gap between CPU usage time and total decrease when only using CPU below)

This approach is by far the fastest and most efficient since the GPU is doing the heavy inference work. The CPU is mainly responsible for preparing data and delegating tasks.

2. Using CPU and no quantization.
   112.83s user 14.14s system 95% cpu 2:13.53 total

With this approach, we can observe that all the computations are being done on the CPU (user is now ~112 sec. and CPU is at 95% usage). There is also only 6s of gap between the total CPU runtime and the total overall wall-clock time.

3. Using CPU and 4-bit quantization.
   83.23s user 26.19s system 18% cpu 9:42.41 total

When using CPU with 4-bit quantization, we see a massive spike in the gap between CPU runtime and total overall wall-clock time to around 473 sec. with only 18% CPU usage, which is contrary to what we expect to see. I asked Claude what was happening and its insights were that this waiting is most likely caused by either memory swapping (quantized model thrashes RAM and disk), bitsandbytes working inefficiently on CPU (designed for CUDA), lock contention, or dequantization overhead (converting 4-bit to float32 on CPU).

THIS SHOWS THAT 4-BIT QUANTIZATION IS ONLY FOR CUDA GPUS AND IT BEING USED ON CPU MAKES PERFORMANCE SO MUCH WORSE.

INCLUDES TIME TO LOAD MODELS ON TOP OF TASK EXECUTION. THE MODEL IS PROBABLY CACHED LOCALLY THOUGH.

Q5: Sample outputs that show the effectiveness of the code

# Model:meta-llama/Llama-3.2-1B-Instruct

# Multi-Model MMLU Evaluation

======================================================================
Environment Check
======================================================================
✓ Running locally (not in Colab)
✓ Platform: Darwin (arm64)
✓ Processor: arm
✓ Apple Metal (MPS) Available
✓ Using Metal Performance Shaders for GPU acceleration
✓ Quantization disabled - loading full precision model
✓ Hugging Face token loaded from .env file

======================================================================
Configuration
======================================================================
Models: 1 models to evaluate

1. meta-llama/Llama-3.2-1B-Instruct
   Device: mps
   Quantization: None (full precision)
   Expected memory: ~2.5 GB (FP16)
   Number of subjects: 10
   ======================================================================

======================================================================
GPU Profiling Setup
======================================================================
PowerMetrics requires sudo access for GPU profiling.
You will be prompted for your password once.

# Password:

======================================================================
EVALUATING MODEL: meta-llama/Llama-3.2-1B-Instruct
======================================================================

Loading model meta-llama/Llama-3.2-1B-Instruct...
Device: mps
✓ Tokenizer loaded
Loading model (this may take 2-3 minutes)...
✓ Model loaded successfully!
Model device: mps:0
Model dtype: torch.float16
Running on Apple Metal (MPS)
✓ PowerMetrics monitoring started (log: powermetrics_20260120_145555.log)

Progress: 1/10 subjects

======================================================================
Evaluating subject: astronomy
======================================================================
Testing astronomy: 0%| | 0/15The following generation flags are not valid and may be ignored: ['top_p']. Set `TRANSFORMERS_VERBOSITY=info` for more details.
Testing astronomy: 100%|█| 152/
✓ Result: 76/152 correct = 50.00%

Progress: 2/10 subjects

======================================================================
Evaluating subject: business_ethics
======================================================================
Testing business_ethics: 100%|█
✓ Result: 45/100 correct = 45.00%

Progress: 3/10 subjects

======================================================================
Evaluating subject: clinical_knowledge
======================================================================
Testing clinical_knowledge: 100
✓ Result: 143/265 correct = 53.96%

Progress: 4/10 subjects

======================================================================
Evaluating subject: college_biology
======================================================================
Testing college_biology: 100%|█
✓ Result: 75/144 correct = 52.08%

Progress: 5/10 subjects

======================================================================
Evaluating subject: college_chemistry
======================================================================
Testing college_chemistry: 100%
✓ Result: 35/100 correct = 35.00%

Progress: 6/10 subjects

======================================================================
Evaluating subject: college_computer_science
======================================================================
Testing college_computer_scienc
✓ Result: 26/100 correct = 26.00%

Progress: 7/10 subjects

======================================================================
Evaluating subject: college_mathematics
======================================================================
Testing college_mathematics: 10
✓ Result: 24/100 correct = 24.00%

Progress: 8/10 subjects

======================================================================
Evaluating subject: college_medicine
======================================================================
Testing college_medicine: 100%|
✓ Result: 79/173 correct = 45.66%

Progress: 9/10 subjects

======================================================================
Evaluating subject: college_physics
======================================================================
Testing college_physics: 100%|█
✓ Result: 17/102 correct = 16.67%

Progress: 10/10 subjects

======================================================================
Evaluating subject: computer_security
======================================================================
Testing computer_security: 100%
✓ Result: 58/100 correct = 58.00%

✓ Memory cleaned up after meta-llama/Llama-3.2-1B-Instruct

======================================================================
COMPARATIVE EVALUATION SUMMARY
======================================================================
Device: MPS
Subjects tested: 10
Verbose mode: OFF
======================================================================

──────────────────────────────────────────────────────────────────────
Model: meta-llama/Llama-3.2-1B-Instruct
──────────────────────────────────────────────────────────────────────
Accuracy: 43.26% (578/1336)

Phase Timing:
Model Loading:
Wall time: 6.56s (0.11 min)
User CPU: 1.23s
System CPU: 2.31s
Total CPU: 3.54s
CPU usage: 54.0%
Evaluation:
Wall time: 419.83s (7.00 min)
User CPU: 222.19s
System CPU: 22.20s
Total CPU: 244.38s
CPU usage: 58.2%

──────────────────────────────────────────────────────────────────
GPU METRICS (Apple Silicon MPS)
──────────────────────────────────────────────────────────────────
GPU Time: 419.83 sec ( 7.00 min)

    GPU Cycles (Approximate):
      Total:       204.27 billion cycles
      Formula:    avg_freq_hz × duration_sec × avg_active_ratio
      Avg Freq:       800 MHz
      Duration:    419.83 sec
      Activity:      60.8%
      Note:       Approximate - based on sampled frequency and activity

    GPU Frequency & Activity:
      Frequency:      800 MHz (avg)     1275 MHz (peak)
      Active:        60.8% (avg)    100.0% (peak)

    GPU Power:
      Average:       1.51 W
      Peak:          3.17 W

──────────────────────────────────────────────────────────────────
CPU METRICS
──────────────────────────────────────────────────────────────────
CPU Time: 244.38 sec ( 4.07 min)

    CPU Cluster Frequencies:
      E-Cluster:     1085 MHz (avg)     2693 MHz (peak)
      P-Cluster:     1793 MHz (avg)     2165 MHz (peak)

──────────────────────────────────────────────────────────────────

Per-Question Inference Stats (1336 questions):
Mean: 305.16ms
Median: 275.83ms
Min: 85.33ms
Max: 3537.32ms
Std Dev: 200.99ms
95th %ile: 493.73ms
99th %ile: 750.55ms
Throughput: 3.18 questions/sec

======================================================================

✓ Results saved to: mmlu_multi_model_results_20260120_150300.json

✅ Evaluation complete!

# Model: google/gemma-2-2b-it

# Multi-Model MMLU Evaluation

======================================================================
Environment Check
======================================================================
✓ Running locally (not in Colab)
✓ Platform: Darwin (arm64)
✓ Processor: arm
✓ Apple Metal (MPS) Available
✓ Using Metal Performance Shaders for GPU acceleration
✓ Quantization disabled - loading full precision model
✓ Hugging Face token loaded from .env file

======================================================================
Configuration
======================================================================
Models: 1 models to evaluate

1. google/gemma-2-2b-it
   Device: mps
   Quantization: None (full precision)
   Expected memory: ~2.5 GB (FP16)
   Number of subjects: 10
   ======================================================================

======================================================================
GPU Profiling Setup
======================================================================
PowerMetrics requires sudo access for GPU profiling.
You will be prompted for your password once.

# Password:

======================================================================
EVALUATING MODEL: google/gemma-2-2b-it
======================================================================

Loading model google/gemma-2-2b-it...
Device: mps
✓ Tokenizer loaded
Loading model (this may take 2-3 minutes)...
Loading checkpoint shards: 100%
✓ Model loaded successfully!
Model device: mps:0
Model dtype: torch.float16
Running on Apple Metal (MPS)
✓ PowerMetrics monitoring started (log: powermetrics_20260121_140237.log)

Progress: 1/10 subjects

======================================================================
Evaluating subject: astronomy
======================================================================
Testing astronomy: 100%|█| 152/
✓ Result: 80/152 correct = 52.63%

Progress: 2/10 subjects

======================================================================
Evaluating subject: business_ethics
======================================================================
Testing business_ethics: 100%|█
✓ Result: 47/100 correct = 47.00%

Progress: 3/10 subjects

======================================================================
Evaluating subject: clinical_knowledge
======================================================================
Testing clinical_knowledge: 100
✓ Result: 136/265 correct = 51.32%

Progress: 4/10 subjects

======================================================================
Evaluating subject: college_biology
======================================================================
Testing college_biology: 100%|█
✓ Result: 95/144 correct = 65.97%

Progress: 5/10 subjects

======================================================================
Evaluating subject: college_chemistry
======================================================================
Testing college_chemistry: 100%
✓ Result: 38/100 correct = 38.00%

Progress: 6/10 subjects

======================================================================
Evaluating subject: college_computer_science
======================================================================
Testing college_computer_scienc
✓ Result: 40/100 correct = 40.00%

Progress: 7/10 subjects

======================================================================
Evaluating subject: college_mathematics
======================================================================
Testing college_mathematics: 10
✓ Result: 25/100 correct = 25.00%

Progress: 8/10 subjects

======================================================================
Evaluating subject: college_medicine
======================================================================
Testing college_medicine: 100%|
✓ Result: 89/173 correct = 51.45%

Progress: 9/10 subjects

======================================================================
Evaluating subject: college_physics
======================================================================
Testing college_physics: 100%|█
✓ Result: 35/102 correct = 34.31%

Progress: 10/10 subjects

======================================================================
Evaluating subject: computer_security
======================================================================
Testing computer_security: 100%
✓ Result: 65/100 correct = 65.00%

✓ Memory cleaned up after google/gemma-2-2b-it

======================================================================
COMPARATIVE EVALUATION SUMMARY
======================================================================
Device: MPS
Subjects tested: 10
Verbose mode: OFF
======================================================================

──────────────────────────────────────────────────────────────────────
Model: google/gemma-2-2b-it
──────────────────────────────────────────────────────────────────────
Accuracy: 48.65% (650/1336)

Phase Timing:
Model Loading:
Wall time: 26.82s (0.45 min)
User CPU: 2.49s
System CPU: 6.48s
Total CPU: 8.97s
CPU usage: 33.5%
Evaluation:
Wall time: 827.17s (13.79 min)
User CPU: 317.67s
System CPU: 60.54s
Total CPU: 378.21s
CPU usage: 45.7%

──────────────────────────────────────────────────────────────────
GPU METRICS (Apple Silicon MPS)
──────────────────────────────────────────────────────────────────
GPU Time: 827.17 sec ( 13.79 min)

    GPU Cycles (Approximate):
      Total:       423.16 billion cycles
      Formula:    avg_freq_hz × duration_sec × avg_active_ratio
      Avg Freq:       717 MHz
      Duration:    827.17 sec
      Activity:      71.4%
      Note:       Approximate - based on sampled frequency and activity

    GPU Frequency & Activity:
      Frequency:      717 MHz (avg)     1231 MHz (peak)
      Active:        71.4% (avg)    100.0% (peak)

    GPU Power:
      Average:       1.58 W
      Peak:          3.15 W

──────────────────────────────────────────────────────────────────
CPU METRICS
──────────────────────────────────────────────────────────────────
CPU Time: 378.21 sec ( 6.30 min)

    CPU Cluster Frequencies:
      E-Cluster:     1440 MHz (avg)     2720 MHz (peak)
      P-Cluster:     1413 MHz (avg)     2158 MHz (peak)

──────────────────────────────────────────────────────────────────

Per-Question Inference Stats (1336 questions):
Mean: 609.18ms
Median: 543.27ms
Min: 243.52ms
Max: 6765.79ms
Std Dev: 454.90ms
95th %ile: 1040.76ms
99th %ile: 2381.10ms
Throughput: 1.62 questions/sec

======================================================================

✓ Results saved to: mmlu_multi_model_results_20260121_141630.json

✅ Evaluation complete!

# Model: allenai/OLMo-2-0425-1B-Instruct

# Multi-Model MMLU Evaluation

======================================================================
Environment Check
======================================================================
✓ Running locally (not in Colab)
✓ Platform: Darwin (arm64)
✓ Processor: arm
✓ Apple Metal (MPS) Available
✓ Using Metal Performance Shaders for GPU acceleration
✓ Quantization disabled - loading full precision model
✓ Hugging Face token loaded from .env file

======================================================================
Configuration
======================================================================
Models: 1 models to evaluate

1. allenai/OLMo-2-0425-1B-Instruct
   Device: mps
   Quantization: None (full precision)
   Expected memory: ~2.5 GB (FP16)
   Number of subjects: 10
   ======================================================================

======================================================================
GPU Profiling Setup
======================================================================
PowerMetrics requires sudo access for GPU profiling.
You will be prompted for your password once.

# Password:

======================================================================
EVALUATING MODEL: allenai/OLMo-2-0425-1B-Instruct
======================================================================

Loading model allenai/OLMo-2-0425-1B-Instruct...
Device: mps
✓ Tokenizer loaded
Loading model (this may take 2-3 minutes)...
✓ Model loaded successfully!
Model device: mps:0
Model dtype: torch.float16
Running on Apple Metal (MPS)
✓ PowerMetrics monitoring started (log: powermetrics_20260120_144658.log)

Progress: 1/10 subjects

======================================================================
Evaluating subject: astronomy
======================================================================
Testing astronomy: 100%|█| 152/
✓ Result: 47/152 correct = 30.92%

Progress: 2/10 subjects

======================================================================
Evaluating subject: business_ethics
======================================================================
Testing business_ethics: 100%|█
✓ Result: 45/100 correct = 45.00%

Progress: 3/10 subjects

======================================================================
Evaluating subject: clinical_knowledge
======================================================================
Testing clinical_knowledge: 100
✓ Result: 102/265 correct = 38.49%

Progress: 4/10 subjects

======================================================================
Evaluating subject: college_biology
======================================================================
Testing college_biology: 100%|█
✓ Result: 53/144 correct = 36.81%

Progress: 5/10 subjects

======================================================================
Evaluating subject: college_chemistry
======================================================================
Testing college_chemistry: 100%
✓ Result: 27/100 correct = 27.00%

Progress: 6/10 subjects

======================================================================
Evaluating subject: college_computer_science
======================================================================
Testing college_computer_scienc
✓ Result: 24/100 correct = 24.00%

Progress: 7/10 subjects

======================================================================
Evaluating subject: college_mathematics
======================================================================
Testing college_mathematics: 10
✓ Result: 32/100 correct = 32.00%

Progress: 8/10 subjects

======================================================================
Evaluating subject: college_medicine
======================================================================
Testing college_medicine: 100%|
✓ Result: 50/173 correct = 28.90%

Progress: 9/10 subjects

======================================================================
Evaluating subject: college_physics
======================================================================
Testing college_physics: 100%|█
✓ Result: 19/102 correct = 18.63%

Progress: 10/10 subjects

======================================================================
Evaluating subject: computer_security
======================================================================
Testing computer_security: 100%
✓ Result: 40/100 correct = 40.00%

✓ Memory cleaned up after allenai/OLMo-2-0425-1B-Instruct

======================================================================
COMPARATIVE EVALUATION SUMMARY
======================================================================
Device: MPS
Subjects tested: 10
Verbose mode: OFF
======================================================================

──────────────────────────────────────────────────────────────────────
Model: allenai/OLMo-2-0425-1B-Instruct
──────────────────────────────────────────────────────────────────────
Accuracy: 32.86% (439/1336)

Phase Timing:
Model Loading:
Wall time: 13.40s (0.22 min)
User CPU: 1.27s
System CPU: 3.44s
Total CPU: 4.71s
CPU usage: 35.2%
Evaluation:
Wall time: 443.64s (7.39 min)
User CPU: 214.57s
System CPU: 24.86s
Total CPU: 239.43s
CPU usage: 54.0%

──────────────────────────────────────────────────────────────────
GPU METRICS (Apple Silicon MPS)
──────────────────────────────────────────────────────────────────
GPU Time: 443.64 sec ( 7.39 min)

    GPU Cycles (Approximate):
      Total:       232.65 billion cycles
      Formula:    avg_freq_hz × duration_sec × avg_active_ratio
      Avg Freq:       784 MHz
      Duration:    443.64 sec
      Activity:      66.9%
      Note:       Approximate - based on sampled frequency and activity

    GPU Frequency & Activity:
      Frequency:      784 MHz (avg)     1276 MHz (peak)
      Active:        66.9% (avg)    100.0% (peak)

    GPU Power:
      Average:       1.60 W
      Peak:          3.06 W

──────────────────────────────────────────────────────────────────
CPU METRICS
──────────────────────────────────────────────────────────────────
CPU Time: 239.43 sec ( 3.99 min)

    CPU Cluster Frequencies:
      E-Cluster:     1279 MHz (avg)     2721 MHz (peak)
      P-Cluster:     1648 MHz (avg)     2106 MHz (peak)

──────────────────────────────────────────────────────────────────

Per-Question Inference Stats (1336 questions):
Mean: 324.24ms
Median: 295.77ms
Min: 118.81ms
Max: 3909.82ms
Std Dev: 219.27ms
95th %ile: 515.25ms
99th %ile: 758.50ms
Throughput: 3.01 questions/sec

======================================================================

✓ Results saved to: mmlu_multi_model_results_20260120_145427.json

✅ Evaluation complete!

Example of Per Question Logging for the Verbose Flag
(in the generated log file)

Model: google/gemma-2-2b-it
Subject: Astronomy
"questions": [
    {
        "question": "What is true for a type-Ia (\"type one-a\") supernova?",
        "choices": {
        "A": "This type occurs in binary systems.",
        "B": "This type occurs in young galaxies.",
        "C": "This type produces gamma-ray bursts.",
        "D": "This type produces high amounts of X-rays."
        },
        "predicted_answer": "A",
        "correct_answer": "A",
        "is_correct": true
    },
    {
        "question": "If you know both the actual brightness of an object and its apparent brightness from your location then with no other information you can estimate:",
        "choices": {
        "A": "Its speed relative to you",
        "B": "Its composition",
        "C": "Its size",
        "D": "Its distance from you"
        },
        "predicted_answer": "D",
        "correct_answer": "D",
        "is_correct": true
    },
    {
        "question": "Why is the sky blue?",
        "choices": {
        "A": "Because the molecules that compose the Earth's atmosphere have a blue-ish color.",
        "B": "Because the sky reflects the color of the Earth's oceans.",
        "C": "Because the atmosphere preferentially scatters short wavelengths.",
        "D": "Because the Earth's atmosphere preferentially absorbs all other colors."
        },
        "predicted_answer": "C",
        "correct_answer": "C",
        "is_correct": true
    },
    {
        "question": "You\u2019ve made a scientific theory that there is an attractive force between all objects. When will your theory be proven to be correct?",
        "choices": {
        "A": "The first time you drop a bowling ball and it falls to the ground proving your hypothesis.",
        "B": "After you\u2019ve repeated your experiment many times.",
        "C": "You can never prove your theory to be correct only \u201cyet to be proven wrong\u201d.",
        "D": "When you and many others have tested the hypothesis."
        },
        "predicted_answer": "D",
        "correct_answer": "C",
        "is_correct": false
    },
    {
        "question": "Which of the following is/are true?",
        "choices": {
        "A": "Titan is the only outer solar system moon with a thick atmosphere",
        "B": "Titan is the only outer solar system moon with evidence for recent geologic activity",
        "C": "Titan's atmosphere is composed mostly of hydrocarbons",
        "D": "A and D"
        },
        "predicted_answer": "D",
        "correct_answer": "D",
        "is_correct": true
    },
]
