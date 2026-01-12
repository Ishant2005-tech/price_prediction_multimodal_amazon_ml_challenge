# Price Prediction Multimodal - Amazon ML Challenge 2025

This repository contains the solution code for the **Amazon ML Challenge 2025 (Price Prediction)**, where we predict product prices based on multimodal data (images + textual descriptions).

## Achievements

**Amazon ML Challenge 2025: Placed within top 1,000 teams among 80K+ participants.**

*   **Final Score (SMAPE):** **49.77**
*   **Rank:** Top 1000

![Team Position Screenshot](static/team_position.png)


## Technical Approach

Our solution leverages the **Qwen2.5-VL-3B-Instruct** Vision-Language Model (VLM), fine-tuned using **QLoRA** on a dataset of **75,000+ multimodal samples**. We orchestrated a distributed multi-GPU training and inference pipeline using **DeepSpeed** and **LLaMA Factory**.

### Model & Architecture
*   **Base Model:** `Qwen/Qwen2.5-VL-3B-Instruct`
*   **Fine-tuning Method:** QLoRA (Quantized Low-Rank Adaptation)
*   **Precision:** 4-bit Normal Float (NF4) quantization for memory efficiency.
*   **Frameworks:** PyTorch, LLaMA Factory, DeepSpeed (Stage 2).

### Data Pipeline
1.  **Image Preprocessing:**
    *   Images are downloaded and resized to **128x128** pixels.
    *   Padding is applied to maintain aspect ratio (black background).
    *   Images are converted to RGB pattern.
2.  **Dataset Formatting:**
    *   Data is converted to **ShareGPT** multimodal format.
    *   Structure: `<image>\nProduct Details: {catalog_content}` -> `price`.

### Training Configuration
We utilized **DeepSpeed ZeRO-2** for memory optimization across GPUs.

| Hyperparameter | Value |
| :--- | :--- |
| **Learning Rate** | `1e-5` |
| **Scheduler** | Cosine (Warmup: 10 steps) |
| **Optimizer** | `adamw_bnb_8bit` |
| **Batch Size** | 4 (per device) |
| **Gradient Accumulation** | 16 steps |
| **Epochs** | 2 |
| **LoRA Rank** | 32 |
| **LoRA Alpha** | 64 |
| **LoRA Dropout** | 0.1 |
| **Target Modules** | `all` |
| **Max Length** | 1024 tokens |

### Inference Pipeline (`VL_inference.py`)
To handle the large dataset efficiently, we built a robust distributed inference script:
*   **Multi-GPU Support:** Uses `deepspeed.init_inference` to parallelize across available GPUs (e.g., 2x T4).
*   **Data Parallelism:** Splits the test dataset into chunks for simultaneous processing.
*   **Robust Extraction:** Implements regex-based post-processing to accurately extract prices from the model's textual output.
*   **Error Handling:** Includes fallback mechanisms for failed image loads or empty predictions.

## Project Structure

*   **`price_prediction_VLM.ipynb`**
    *   Complete development workflow.
    *   Data cleaning, image downloading, and preprocessing.
    *   LLaMA Factory configuration and training loop.
    *   Model merging and export.
*   **`VL_inference.py`**
    *   Standalone script for high-throughput inference.
    *   Optimized for T4 GPUs with DeepSpeed.

##  Usage

### 1. Prerequisites
Ensure you have a GPU-enabled environment (e.g., Google Colab, Kaggle, or local CUDA machine).

```bash
pip install torch transformers deepspeed pandas pillow accelerate peft bitsandbytes
pip install "deepspeed>=0.10.0,<=0.16.9"
```

### 2. Training
The training process is documented in `price_prediction_VLM.ipynb`. It uses LLaMA Factory for streamlined training.

### 3. Running Inference
To generate predictions using the trained model:

```bash
# Run with DeepSpeed on all available GPUs
deepspeed --num_gpus=2 VL_inference.py
```

*Note: You may need to adjust paths in the script (e.g., `MERGED_MODEL_DIR`) to point to your specific directories.*

## Author

Nitish Biswas   
nitishbiswas066@gmail.com