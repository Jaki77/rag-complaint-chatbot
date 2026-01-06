"""
Stratified sampling utilities for creating balanced dataset samples.
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from sklearn.model_selection import train_test_split


def create_stratified_sample(
    df: pd.DataFrame,
    sample_size: int = 12000,
    stratify_col: str = 'Product_standardized',
    random_state: int = 42
) -> pd.DataFrame:
    """
    Create a stratified sample from the dataset.
    
    Args:
        df: Input DataFrame with cleaned complaints
        sample_size: Target sample size (default: 12,000)
        stratify_col: Column to use for stratification
        random_state: Random seed for reproducibility
    
    Returns:
        Stratified sample DataFrame
    """
    print(f"Creating stratified sample of size {sample_size:,}...")
    
    # Check if sample size is larger than dataset
    if sample_size > len(df):
        print(f"Warning: Sample size ({sample_size}) > Dataset size ({len(df)}). Using full dataset.")
        return df.copy()
    
    # Calculate proportions for stratification
    product_counts = df[stratify_col].value_counts()
    total_complaints = len(df)
    
    print("\nOriginal distribution:")
    for product, count in product_counts.items():
        percentage = (count / total_complaints) * 100
        print(f"  {product}: {count:,} complaints ({percentage:.1f}%)")
    
    # Calculate target counts per category
    proportions = product_counts / total_complaints
    target_counts = (proportions * sample_size).round().astype(int)
    
    # Adjust to ensure total equals sample_size
    total_target = target_counts.sum()
    if total_target != sample_size:
        # Adjust the largest category
        diff = sample_size - total_target
        largest_category = target_counts.idxmax()
        target_counts[largest_category] += diff
    
    print("\nTarget sample distribution:")
    for product, target in target_counts.items():
        print(f"  {product}: {target:,} complaints")
    
    # Perform stratified sampling
    sampled_dfs = []
    
    for product, target_count in target_counts.items():
        product_df = df[df[stratify_col] == product]
        
        if len(product_df) >= target_count:
            # Sample if we have enough
            product_sample = product_df.sample(
                n=target_count, 
                random_state=random_state
            )
        else:
            # Use all if not enough
            print(f"Warning: Not enough {product} complaints ({len(product_df)} < {target_count})")
            product_sample = product_df.copy()
        
        sampled_dfs.append(product_sample)
    
    # Combine all samples
    sampled_df = pd.concat(sampled_dfs, ignore_index=True)
    
    # Shuffle the sample
    sampled_df = sampled_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    print(f"\nFinal sample size: {len(sampled_df):,}")
    print("\nFinal sample distribution:")
    final_counts = sampled_df[stratify_col].value_counts()
    for product, count in final_counts.items():
        percentage = (count / len(sampled_df)) * 100
        print(f"  {product}: {count:,} complaints ({percentage:.1f}%)")
    
    return sampled_df


def analyze_sample_quality(
    original_df: pd.DataFrame,
    sampled_df: pd.DataFrame,
    stratify_col: str = 'Product_standardized'
) -> Dict[str, Any]:
    """
    Analyze how well the sample represents the original data.
    
    Returns:
        Dictionary with quality metrics
    """
    original_counts = original_df[stratify_col].value_counts(normalize=True)
    sample_counts = sampled_df[stratify_col].value_counts(normalize=True)
    
    # Calculate distribution similarity
    similarity_scores = {}
    for product in original_counts.index:
        original_pct = original_counts[product] * 100
        sample_pct = sample_counts.get(product, 0) * 100
        diff = abs(original_pct - sample_pct)
        similarity_scores[product] = {
            'original_pct': original_pct,
            'sample_pct': sample_pct,
            'difference': diff
        }
    
    # Calculate overall mean absolute difference
    mean_diff = np.mean([s['difference'] for s in similarity_scores.values()])
    
    return {
        'similarity_scores': similarity_scores,
        'mean_absolute_difference': mean_diff,
        'total_samples': len(sampled_df),
        'coverage': len(sampled_df) / len(original_df) * 100
    }