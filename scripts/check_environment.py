"""Environment diagnostic script for Defect-GAN deployment and company GPU evaluation."""

import os
import platform
import shutil
import subprocess
import sys


def print_header(title: str) -> None:
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def check_system():
    print_header("1. HOST SYSTEM & PYTHON")
    print(f"Operating System: {platform.system()} {platform.release()} ({platform.version()})")
    print(f"Platform Architecture: {platform.machine()}")
    print(f"Python Executable: {sys.executable}")
    print(f"Python Version: {platform.python_version()}")


def check_pytorch_cuda():
    print_header("2. PYTORCH & CUDA ACCELERATION")
    try:
        import torch
        print(f"PyTorch Version: {torch.__version__}")
        cuda_avail = torch.cuda.is_available()
        print(f"CUDA Available: {cuda_avail}")

        if cuda_avail:
            print(f"CUDA Version (PyTorch built with): {torch.version.cuda}")
            print(f"cuDNN Version: {torch.backends.cudnn.version()}")
            print(f"cuDNN Enabled: {torch.backends.cudnn.enabled}")
            device_count = torch.cuda.device_count()
            print(f"NVIDIA GPU Count: {device_count}")

            for i in range(device_count):
                props = torch.cuda.get_device_properties(i)
                total_mem_gb = props.total_memory / (1024 ** 3)
                print(f"\n  [GPU {i}] {props.name}")
                print(f"    Compute Capability: {props.major}.{props.minor}")
                print(f"    Total Dedicated VRAM: {total_mem_gb:.2f} GB")
                print(f"    Multi-Processors: {props.multi_processor_count}")

                # Check free memory
                try:
                    free_mem, total_mem = torch.cuda.mem_get_info(i)
                    print(f"    Currently Free VRAM: {free_mem / (1024 ** 3):.2f} GB / {total_mem / (1024 ** 3):.2f} GB")
                except Exception as e:
                    print(f"    Could not query free memory: {e}")

                # Check AMP support
                bf16_support = torch.cuda.is_bf16_supported() if hasattr(torch.cuda, "is_bf16_supported") else False
                print(f"    FP16 AMP Supported: True")
                print(f"    BF16 AMP Supported: {bf16_support}")
        else:
            print("  [INFO] No CUDA GPU detected by PyTorch.")
            print("  If running inside Docker on the company GPU machine, ensure:")
            print("  1. NVIDIA drivers are installed on the host.")
            print("  2. nvidia-container-toolkit is installed.")
            print("  3. Container is launched with '--gpus all'.")

    except ImportError:
        print("[ERROR] PyTorch is not installed in this environment.")


def check_nvidia_smi():
    print_header("3. NVIDIA-SMI & DRIVER STATUS")
    smi_path = shutil.which("nvidia-smi")
    if smi_path:
        print(f"nvidia-smi located at: {smi_path}")
        try:
            res = subprocess.run([smi_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=10)
            print(res.stdout)
        except Exception as e:
            print(f"Failed to execute nvidia-smi: {e}")
    else:
        print("nvidia-smi not found in system PATH.")


def recommend_training_parameters():
    print_header("4. TRAINING PARAMETER RECOMMENDATIONS")
    try:
        import torch
        if not torch.cuda.is_available():
            print("Environment: CPU ONLY")
            print("  - Status: READY FOR SMOKE TESTING & CODE VERIFICATION ONLY.")
            print("  - Action: DO NOT run full GAN training on CPU.")
            print("  - Use command: python -m src.train --config configs/smoke_test.yaml --dry-run")
            return

        props = torch.cuda.get_device_properties(0)
        vram_gb = props.total_memory / (1024 ** 3)
        print(f"Evaluated GPU: {props.name} with {vram_gb:.2f} GB VRAM")

        if vram_gb >= 22.0:
            print("  Classification: HIGH VRAM (>= 24 GB, e.g., RTX 3090, 4090, A5000, A6000, A100)")
            print("  - Native 1024x1024 baseline: FULLY SUPPORTED")
            print("  - Recommended batch_size: 1 or 2")
            print("  - Recommended gradient_accumulation_steps: 2 or 4 (effective batch size: 4)")
            print("  - Mixed precision (AMP): fp16 or bf16")
            print("  - num_workers: 4")
        elif vram_gb >= 15.0:
            print("  Classification: MEDIUM-HIGH VRAM (16 GB, e.g., RTX 4080, V100, T4-16G, RTX 3080-16G)")
            print("  - Native 1024x1024 baseline: SUPPORTED with careful gradient management")
            print("  - Recommended batch_size: 1")
            print("  - Recommended gradient_accumulation_steps: 4 (effective batch size: 4)")
            print("  - Mixed precision (AMP): fp16 (MANDATORY to avoid OOM)")
            print("  - num_workers: 2")
        elif vram_gb >= 10.0:
            print("  Classification: MEDIUM VRAM (12 GB, e.g., RTX 3060 12GB, RTX 3080 10GB, RTX 4070)")
            print("  - Native 1024x1024 baseline: MARGINAL (May encounter OOM during cycle backprop)")
            print("  - Recommended batch_size: 1 with gradient_accumulation_steps: 4")
            print("  - Mixed precision (AMP): MANDATORY fp16")
            print("  - Alternative: Fallback to 512x512 patch configuration if OOM occurs")
        else:
            print("  Classification: LOW VRAM (< 10 GB, e.g., 6GB - 8GB GPUs)")
            print("  - Native 1024x1024 baseline: HIGH RISK OF OOM")
            print("  - Recommended Action: Train on 512x512 patches or perform CPU smoke tests.")

    except Exception as e:
        print(f"Error computing recommendations: {e}")


def main():
    check_system()
    check_pytorch_cuda()
    check_nvidia_smi()
    recommend_training_parameters()
    print("\n" + "=" * 60)
    print("  Environment check completed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
