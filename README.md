# Lab 2: Matrix Multiplication and Image Blur

**Class:** EEL 4930 – Introduction to GPU Computing (Fall 2026)  
**Name:** Ilan Danial  

---

## 1. Environment & Hardware Specifications

- **Compute Node:** `c0604a-s22.ufhpc`
- **GPU Model:** NVIDIA L4 (23,034 MiB)
- **NVIDIA Driver:** 580.178.04
- **CUDA Toolkit:** Release 13.2, V13.2.78
- **Host Compiler:** GCC 14.2.0

---

## 2. Build & Run Instructions

### Build:
```bash
module load gcc cuda
make clean
make
```
Compiler flags (from the Makefile): `nvcc -O3 -std=c++17 -Xcompiler -Wall,-Wextra`

### Run:
- **Interactive single run (arguments are M K N):**
  ```bash
  ./lab2 31 17 50
  ```
- **Automated batch submission (builds, runs the sanitizer, all five public tests, and the timing case):**
  ```bash
  sbatch run.slurm
  ```
- **Compute Sanitizer:**
  ```bash
  compute-sanitizer --tool memcheck ./lab2 31 17 50
  ```

---

## 3. Part A · Correctness Test Matrix

| Test Case (M, K, N) | Grid (blocks x × y) | Verification |
| :--- | :--- | :---: |
| **1, 1, 1** | 1 × 1 | **PASS** |
| **3, 5, 2** | 1 × 1 | **PASS** |
| **16, 16, 16** | 1 × 1 | **PASS** |
| **31, 17, 50** | 4 × 2 | **PASS** |
| **65, 33, 129** | 9 × 5 | **PASS** |

Grid shown for the 16 × 16 block used in the correctness run: `grid.x = ceil(N / 16)`, `grid.y = ceil(M / 16)`.

**Compute Sanitizer (31, 17, 50):** `ERROR SUMMARY: 0 errors`

---

## 4. Part A · Timing Study


| Block Shape | Threads / Block | M, K, N | Median Kernel Time |
| :--- | :---: | :--- | :---: |
| **16 × 16** | 256 | 1024, 1024, 1024 | 1.18067 ms |
| **32 × 8** | 256 | 1024, 1024, 1024 | 1.17043 ms |

Timing method: Ran each block shape 10 times per M K N test, the timing was taken with Cuda events and we take the median of those runs. They are only timed before and after the launch of the kernel.

### Observations:

There is close to no difference because the threads are assigned to warps in nearly the same way. Both shapes have the same number of threads and warps, and in both a warp reads neighboring elements of B and the same element of A. The memory access patterns are consistent, so changing the shape of the block gives near nothing new to exploit, and the times end up nearly the same.

---

## 5. Problems Encountered & Resolution

When doing cat on the output from my job I was running through all of them rather than just the most recent.

---

## 6. AI Disclosure

I used Claude Code, I used it for double checking assignment requirements, configuring the slurm submission headers, and debugging my compiler issues, and was also used to format (not write) the readme file.
I verified my code by running it on HiPerGator.

---

## 7. Files Submitted

- `main.cu` — CUDA matmul kernel, CPU reference, elementwise verifier, host pipeline, timing study
- `Makefile` — builds `lab2` with nvcc
- `run.slurm` — HiPerGator batch script: build, sanitizer, five public tests, timing case
- `Part_A.txt` — slurm output log for job 46731253 (environment, build, tests, sanitizer, timing)
- `README.md` — this file
