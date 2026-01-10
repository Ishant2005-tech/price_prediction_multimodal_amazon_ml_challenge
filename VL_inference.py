#!/usr/bin/env python3
"""
DeepSpeed Multi-GPU Inference for Qwen2.5-VL-3B on 2x T4 GPUs with Images
"""

import os
import sys

# Set environment variables early
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

import deepspeed
import torch
import pandas as pd
import re
import logging
from tqdm import tqdm
import gc
import json
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from PIL import Image
import torch.distributed as dist

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Directories and Config
MERGED_MODEL_DIR = "/content/drive/MyDrive/amazon_ml_challenge_VL_75000/Output/merged_model"
TEST_JSON = '/content/drive/MyDrive/amazon_ml_challenge_VL_75000/Output/price_prediction_test.json'
SUBMISSION_CSV = '/content/drive/MyDrive/amazon_ml_challenge_VL_75000/Output/submission.csv'
BATCH_SIZE = 8

def extract_price_from_text(text):
    """Enhanced price extraction with debugging"""
    try:
        # Print what we're receiving for debugging
        
        
        # Isolate the part after the last 'assistant'
        if 'assistant' in text:
            text = text.split('assistant')[-1].strip()  # Take only after the last 'assistant'
        else:
            text = text.strip()  # Fallback if no marker
        
        text_clean = text.lower()
        
        # More comprehensive price patterns (kept your original, but removed redundant integer-only from floats)
        patterns = [
            r'\$?(\d+\.\d{2})\b',           # $19.99 or 19.99
            r'\$?(\d+\.\d+)\b',             # $19.9 or 19.9
            r'price[:\s]*\$?(\d+\.\d+)',    # "price: $19.99"
            r'cost[:\s]*\$?(\d+\.\d+)',     # "cost: $19.99"  
            r'USD[:\s]*(\d+\.\d+)',         # "USD: 19.99"
            r'(\d+\.\d+)\s*USD',            # "19.99 USD"
            r'(\d+\.\d+)\s*dollars?',       # "19.99 dollars"
            r'(\d+)\s*dollars?',            # "19 dollars"
        ]
        
        # Find ALL matches across patterns
        all_matches = []
        for i, pattern in enumerate(patterns):
            matches = re.findall(pattern, text_clean)
            if matches:
                all_matches.extend(matches)
                
        
        if all_matches:
            price = float(all_matches[-1])  # Take the LAST match as the prediction
            
            return price
        
        print(f"âŒ No price pattern matched for: '{text_clean}'")
        return 0.0
        
    except Exception as e:
        print(f"âŒ Error extracting price: {e}")
        return 0.0


def load_image_safely(image_path):
    """Load image with proper error handling and return PIL Image object"""
    try:
        if os.path.exists(image_path):
            image = Image.open(image_path).convert('RGB')
            # Resize if too large to prevent OOM
            if image.size[0] > 1280 or image.size[1] > 1280:
                image.thumbnail((1280, 1280), Image.Resampling.LANCZOS)
            return image
        else:
            logger.warning(f"Image not found: {image_path}")
            return None
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {e}")
        return None

def main():
    # Initialize distributed processing
    deepspeed.init_distributed()
    
    local_rank = int(os.getenv('LOCAL_RANK', '0'))
    world_size = int(os.getenv('WORLD_SIZE', '1'))
    
    # Set device
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    
    logger.info(f"Starting DeepSpeed inference on rank {local_rank} of {world_size} GPUs")
    
    try:
        # Load processor for VL model
        processor = AutoProcessor.from_pretrained(
            MERGED_MODEL_DIR, 
            trust_remote_code=True,
            use_fast=True,  # Explicitly set to avoid warnings
            min_pixels=128*128,
            max_pixels=128*128*4   # Reduced for T4 GPUs
        )
        processor.tokenizer.padding_side = "left"
        # Load VL model
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MERGED_MODEL_DIR,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True
        )
        
        # Initialize DeepSpeed inference engine
        # Replace the try-except for ds_engine
        try:
            ds_engine = deepspeed.init_inference(
                model,
                tensor_parallel={"tp_size": world_size},
                dtype=torch.float16,
                replace_with_kernel_inject=False,
                mp_size=world_size  # Note: mp_size is deprecated, but keep if needed
            )
            model = ds_engine.module
            logger.info(f"DeepSpeed initialized successfully on rank {local_rank}")
        except Exception as e:
            logger.warning(f"DeepSpeed init failed: {e}, falling back to single GPU per process")
            model = model.to(device)  # Load full model on this rank's GPU
        
        
        model.eval()
        
        # Log GPU info (on rank 0 only)
        if local_rank == 0:
            logger.info(f"Number of GPUs: {world_size}")
            for i in range(world_size):
                if i < torch.cuda.device_count():
                    logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        
        # Load test data (JSON format with images)
        with open(TEST_JSON, 'r') as f:
            test_data = json.load(f)
        logger.info(f"Loaded {len(test_data)} test samples")
        
        # Split data evenly across GPUs
        chunk_size = len(test_data) // world_size
        start_idx = local_rank * chunk_size
        end_idx = (local_rank + 1) * chunk_size if local_rank < world_size - 1 else len(test_data)
        local_data = test_data[start_idx:end_idx]
        logger.info(f"Rank {local_rank} processing samples {start_idx} to {end_idx-1} ({len(local_data)} samples)")
        
        local_predictions = []
        local_sample_ids = []
        
        # Process batches
        for i in tqdm(range(0, len(local_data), BATCH_SIZE), desc=f"Rank {local_rank} Inference"):
            try:
                batch = local_data[i:i+BATCH_SIZE]
                batch_messages = []
                batch_images = []  # This will store PIL Images, not paths
                
                for entry in batch:
                    try:
                        # Extract data with error handling
                        system_msg = entry.get("messages", [{}])[0]
                        user_msg = entry.get("messages", [{}, {}])[1] 
                        image_path = entry.get("images", [None])[0]
                        sample_id = entry.get("sample_id", f"unknown_{i}")
                        
                        # ðŸ”¥ CRITICAL FIX: Load actual PIL Image objects
                        image = load_image_safely(image_path) if image_path else None
                        
                        # Create messages for VL model
                        if image is not None:
                            new_user_content = [
                                {"type": "image", "image": image},  # PIL Image object
                                {"type": "text", "text": user_msg.get("content", "").replace("<image>", "").strip()}
                            ]
                            # Add PIL Image to batch for processor
                            batch_images.append(image)  # Single PIL Image, not list
                        else:
                            new_user_content = [
                                {"type": "text", "text": user_msg.get("content", "").replace("<image>", "").strip()}
                            ]
                            batch_images.append(None)  # No image
                        
                        new_messages = [
                            {
                                "role": "system",
                                "content": [{"type": "text", "text": system_msg.get("content", "")}]
                            },
                            {"role": "user", "content": new_user_content}
                        ]
                        
                        batch_messages.append(new_messages)
                        local_sample_ids.append(sample_id)
                        
                    except Exception as e:
                        logger.error(f"Error processing entry: {e}")
                        # Add fallback
                        batch_messages.append([
                            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
                            {"role": "user", "content": [{"type": "text", "text": "Predict a price."}]}
                        ])
                        batch_images.append(None)
                        local_sample_ids.append(f"error_{i}")
                
                # Apply chat template
                prompts = []
                for msg in batch_messages:
                    prompt = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
                    prompts.append(prompt)
                
                # ðŸ”¥ CRITICAL FIX: Process with proper image format
                # Filter out None images and structure properly
                 images_list = [img for img in batch_images if img is not None] 
                
                
                if images_list:
                    # For VL processor, pass images as a flat list
                    inputs = processor(
                        text=prompts,
                        images=images_list,  # Now correct format: List[List[PIL.Image]]
                        padding=True,
                        truncation=True,
                        max_length=2048,
                        return_tensors="pt"
                    ).to(device)
                    
                else:
                    # Text-only processing when no images
                    inputs = processor(
                        text=prompts,
                        padding=True,
                        truncation=True,
                        max_length=2048,
                        return_tensors="pt"
                    ).to(device)
                
                # Generate predictions
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=50,
                        do_sample=False,
                        temperature=0.0,
                        eos_token_id=processor.tokenizer.eos_token_id,
                        pad_token_id=processor.tokenizer.eos_token_id,
                        use_cache=True
                    )
                # After model.generate
                processor.tokenizer.padding_side = "right" 
                actual_lengths = inputs.attention_mask.sum(dim=1).tolist()# List of effective input lengths per sample
                
                generated_ids = [outputs[i, actual_lengths[i]:] for i in range(outputs.shape[0])]
                generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
                # # Decode generated text
                # generated_texts = processor.batch_decode(
                #     outputs[:, inputs['input_ids'].shape[1]:], 
                #     skip_special_tokens=True
                # )
                
                # Extract prices
                batch_predictions = [extract_price_from_text(text) for text in generated_texts]
                local_predictions.extend(batch_predictions)
                
                # Periodic cleanup
                if i % 20 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()
                
            except Exception as e:
                logger.error(f"Error in batch {i}: {e}")
                # Add fallback predictions
                fallback_size = min(BATCH_SIZE, len(local_data) - i)
                local_predictions.extend([0.0] * fallback_size)
        
        # Save local results
        temp_file = f'/content/drive/MyDrive/amazon_ml_challenge_VL_75000/Output/predictions_rank_{local_rank}.csv'
        local_result_df = pd.DataFrame({
            'sample_id': local_sample_ids, 
            'price': local_predictions
        })
        local_result_df.to_csv(temp_file, index=False)
        logger.info(f"Rank {local_rank} saved {len(local_predictions)} predictions")
        
        # Synchronize all ranks
        if dist.is_initialized():
            dist.barrier()
        
        # Combine results on rank 0
        if local_rank == 0:
            all_dfs = []
            for r in range(world_size):
                temp_file = f'/content/drive/MyDrive/amazon_ml_challenge_VL_75000/Output/predictions_rank_{r}.csv'
                if os.path.exists(temp_file):
                    df = pd.read_csv(temp_file)
                    all_dfs.append(df)
                    try:
                        os.remove(temp_file)
                    except:
                        pass
                else:
                    logger.warning(f"Missing results from rank {r}")
            
            if all_dfs:
                submission_df = pd.concat(all_dfs).sort_values('sample_id').reset_index(drop=True)
                submission_df.to_csv(SUBMISSION_CSV, index=False)
                logger.info(f"Submission saved to {SUBMISSION_CSV}")
                
                # Print statistics
                logger.info(f"Total predictions: {len(submission_df)}")
                logger.info(f"Mean price: ${submission_df['price'].mean():.2f}")
                logger.info(f"Median price: ${submission_df['price'].median():.2f}")
                logger.info(f"Min: ${submission_df['price'].min():.2f}, Max: ${submission_df['price'].max():.2f}")
            else:
                logger.error("No valid predictions found!")
    
    except Exception as e:
        logger.error(f"Fatal error in main: {e}")
        raise
    
    finally:
        # Proper cleanup
        if dist.is_initialized():
            dist.destroy_process_group()

if __name__ == "__main__":
    main()
