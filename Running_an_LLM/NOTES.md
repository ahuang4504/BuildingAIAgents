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