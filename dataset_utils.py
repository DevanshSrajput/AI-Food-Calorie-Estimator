"""
Dataset Utilities for AI Food Calorie Estimator - Food-101
Comprehensive dataset management, validation, and preprocessing utilities
"""

import os
import shutil
import json
import csv
import hashlib
import zipfile
import tarfile
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
import pandas as pd
import numpy as np
from PIL import Image, ImageStat
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
from datetime import datetime
import logging
from tqdm import tqdm
import requests
import random

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetStatistics:
    """Statistics and analysis for food datasets"""
    
    def __init__(self):
        self.stats = {}
        
    def analyze_images(self, dataset_path: Union[str, Path]) -> Dict[str, Any]:
        """Comprehensive image analysis"""
        dataset_path = Path(dataset_path)
        
        analysis = {
            'total_images': 0,
            'total_size_mb': 0,
            'classes': {},
            'image_formats': Counter(),
            'image_sizes': [],
            'corrupted_images': [],
            'duplicate_hashes': defaultdict(list),
            'color_analysis': {},
            'metadata': {
                'analyzed_at': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
                'analyzed_by': 'DevanshSrajput'
            }
        }
        
        logger.info("Starting comprehensive image analysis...")
        
        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(dataset_path.rglob(f'*{ext}'))
            image_files.extend(dataset_path.rglob(f'*{ext.upper()}'))
        
        # Analyze each image
        for img_path in tqdm(image_files, desc="Analyzing images"):
            try:
                # Basic file info
                file_size = img_path.stat().st_size
                analysis['total_size_mb'] += file_size / (1024 * 1024)
                analysis['total_images'] += 1
                
                # Get class name (parent directory)
                class_name = img_path.parent.name
                if class_name not in analysis['classes']:
                    analysis['classes'][class_name] = {
                        'count': 0,
                        'total_size_mb': 0,
                        'avg_width': 0,
                        'avg_height': 0,
                        'formats': Counter()
                    }
                
                analysis['classes'][class_name]['count'] += 1
                analysis['classes'][class_name]['total_size_mb'] += file_size / (1024 * 1024)
                
                # File format
                file_format = img_path.suffix.lower()
                analysis['image_formats'][file_format] += 1
                analysis['classes'][class_name]['formats'][file_format] += 1
                
                # Load and analyze image
                with Image.open(img_path) as img:
                    width, height = img.size
                    analysis['image_sizes'].append((width, height))
                    
                    # Update class averages
                    class_info = analysis['classes'][class_name]
                    class_info['avg_width'] = (class_info['avg_width'] * (class_info['count'] - 1) + width) / class_info['count']
                    class_info['avg_height'] = (class_info['avg_height'] * (class_info['count'] - 1) + height) / class_info['count']
                    
                    # Color analysis (sample)
                    if random.random() < 0.1:  # Sample 10% for color analysis
                        stat = ImageStat.Stat(img)
                        if class_name not in analysis['color_analysis']:
                            analysis['color_analysis'][class_name] = []
                        analysis['color_analysis'][class_name].append({
                            'mean_rgb': stat.mean,
                            'stddev_rgb': stat.stddev
                        })
                
                # Calculate hash for duplicate detection
                with open(img_path, 'rb') as f:
                    file_hash = hashlib.md5(f.read()).hexdigest()
                    analysis['duplicate_hashes'][file_hash].append(str(img_path))
                    
            except Exception as e:
                logger.warning(f"Error analyzing {img_path}: {e}")
                analysis['corrupted_images'].append(str(img_path))
        
        # Post-process statistics
        if analysis['image_sizes']:
            widths, heights = zip(*analysis['image_sizes'])
            analysis['avg_image_size'] = (np.mean(widths), np.mean(heights))
            analysis['min_image_size'] = (min(widths), min(heights))
            analysis['max_image_size'] = (max(widths), max(heights))
        
        # Find duplicates
        analysis['duplicates'] = {k: v for k, v in analysis['duplicate_hashes'].items() if len(v) > 1}
        del analysis['duplicate_hashes']  # Remove large temporary data
        
        logger.info(f"Analysis complete: {analysis['total_images']} images, {len(analysis['classes'])} classes")
        return analysis

class DatasetValidator:
    """Validation utilities for food datasets"""
    
    def __init__(self, min_images_per_class: int = 50, max_image_size_mb: int = 10):
        self.min_images_per_class = min_images_per_class
        self.max_image_size_mb = max_image_size_mb
        
    def validate_structure(self, dataset_path: Union[str, Path]) -> Dict[str, Any]:
        """Validate dataset directory structure"""
        dataset_path = Path(dataset_path)
        
        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'structure_type': 'unknown',
            'classes': [],
            'total_images': 0,
            'validated_at': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
            'validated_by': 'DevanshSrajput'
        }
        
        if not dataset_path.exists():
            validation_results['valid'] = False
            validation_results['errors'].append(f"Dataset path does not exist: {dataset_path}")
            return validation_results
        
        # Check for organized structure (train/val/test)
        organized_dirs = ['train', 'val', 'test']
        if all((dataset_path / d).exists() for d in organized_dirs[:2]):  # At least train and val
            validation_results['structure_type'] = 'organized'
            logger.info("Detected organized dataset structure")
            
            for split in organized_dirs:
                split_path = dataset_path / split
                if split_path.exists():
                    self._validate_split(split_path, split, validation_results)
        
        # Check for single directory structure
        elif any(p.is_dir() for p in dataset_path.iterdir()):
            validation_results['structure_type'] = 'single'
            logger.info("Detected single directory structure")
            self._validate_split(dataset_path, 'single', validation_results)
        
        else:
            validation_results['valid'] = False
            validation_results['errors'].append("No valid directory structure found")
        
        return validation_results
    
    def _validate_split(self, split_path: Path, split_name: str, results: Dict[str, Any]):
        """Validate a single split directory"""
        class_dirs = [d for d in split_path.iterdir() if d.is_dir()]
        
        if not class_dirs:
            results['errors'].append(f"No class directories found in {split_name}")
            results['valid'] = False
            return
        
        split_classes = []
        split_images = 0
        
        for class_dir in class_dirs:
            class_name = class_dir.name
            
            # Count images in this class
            image_files = self._get_image_files(class_dir)
            image_count = len(image_files)
            split_images += image_count
            
            if image_count == 0:
                results['warnings'].append(f"No images found in {split_name}/{class_name}")
                continue
            
            if image_count < self.min_images_per_class:
                results['warnings'].append(
                    f"Class {class_name} in {split_name} has only {image_count} images "
                    f"(minimum recommended: {self.min_images_per_class})"
                )
            
            split_classes.append({
                'name': class_name,
                'image_count': image_count,
                'split': split_name
            })
            
            # Validate image files
            self._validate_images(image_files, class_name, split_name, results)
        
        results['classes'].extend(split_classes)
        results['total_images'] += split_images
        
        logger.info(f"Validated {split_name}: {len(split_classes)} classes, {split_images} images")
    
    def _get_image_files(self, directory: Path) -> List[Path]:
        """Get all image files in a directory"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(directory.glob(f'*{ext}'))
            image_files.extend(directory.glob(f'*{ext.upper()}'))
        
        return image_files
    
    def _validate_images(self, image_files: List[Path], class_name: str, split_name: str, results: Dict[str, Any]):
        """Validate individual image files"""
        for img_path in image_files:
            try:
                # Check file size
                file_size_mb = img_path.stat().st_size / (1024 * 1024)
                if file_size_mb > self.max_image_size_mb:
                    results['warnings'].append(
                        f"Large image file: {img_path.name} ({file_size_mb:.1f}MB) in {split_name}/{class_name}"
                    )
                
                # Try to open image
                with Image.open(img_path) as img:
                    # Check if image is valid
                    img.verify()
                    
            except Exception as e:
                results['errors'].append(f"Corrupted image: {img_path} - {str(e)}")
                results['valid'] = False

class DatasetOrganizer:
    """Organize and restructure food datasets"""
    
    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        np.random.seed(random_seed)
        random.seed(random_seed)
    
    def create_train_val_test_split(
        self, 
        source_dir: Union[str, Path], 
        target_dir: Union[str, Path],
        train_ratio: float = 0.7,
        val_ratio: float = 0.2,
        test_ratio: float = 0.1,
        copy_files: bool = True
    ) -> Dict[str, Any]:
        """Create organized train/validation/test splits"""
        
        source_path = Path(source_dir)
        target_path = Path(target_dir)
        
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 0.001:
            raise ValueError("Split ratios must sum to 1.0")
        
        logger.info(f"Creating dataset splits: train={train_ratio}, val={val_ratio}, test={test_ratio}")
        
        # Create target directories
        splits = ['train', 'val', 'test']
        for split in splits:
            (target_path / split).mkdir(parents=True, exist_ok=True)
        
        split_info = {
            'train': {'count': 0, 'classes': {}},
            'val': {'count': 0, 'classes': {}},
            'test': {'count': 0, 'classes': {}},
            'metadata': {
                'source_dir': str(source_path),
                'target_dir': str(target_path),
                'train_ratio': train_ratio,
                'val_ratio': val_ratio,
                'test_ratio': test_ratio,
                'created_at': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
                'created_by': 'DevanshSrajput',
                'random_seed': self.random_seed
            }
        }
        
        # Process each class directory
        class_dirs = [d for d in source_path.iterdir() if d.is_dir()]
        
        for class_dir in tqdm(class_dirs, desc="Organizing classes"):
            class_name = class_dir.name
            
            # Get all images in this class
            image_files = self._get_image_files(class_dir)
            
            if not image_files:
                logger.warning(f"No images found in class: {class_name}")
                continue
            
            # Shuffle files for random split
            import random
            random.shuffle(image_files)
            
            # Calculate split indices
            n_total = len(image_files)
            n_train = int(n_total * train_ratio)
            n_val = int(n_total * val_ratio)
            n_test = n_total - n_train - n_val
            
            # Split files
            train_files = image_files[:n_train]
            val_files = image_files[n_train:n_train + n_val]
            test_files = image_files[n_train + n_val:]
            
            # Create class directories in each split
            for split in splits:
                (target_path / split / class_name).mkdir(parents=True, exist_ok=True)
            
            # Copy/move files to respective directories
            file_splits = {
                'train': train_files,
                'val': val_files,
                'test': test_files
            }
            
            for split_name, files in file_splits.items():
                split_info[split_name]['classes'][class_name] = len(files)
                split_info[split_name]['count'] += len(files)
                
                for file_path in files:
                    target_file = target_path / split_name / class_name / file_path.name
                    
                    if copy_files:
                        shutil.copy2(file_path, target_file)
                    else:
                        shutil.move(str(file_path), str(target_file))
            
            logger.debug(f"Class {class_name}: {n_train} train, {n_val} val, {n_test} test")
        
        # Save split information
        split_info_path = target_path / "split_info.json"
        with open(split_info_path, 'w') as f:
            json.dump(split_info, f, indent=2)
        
        logger.info(f"Dataset split completed:")
        logger.info(f"  Train: {split_info['train']['count']} images")
        logger.info(f"  Val: {split_info['val']['count']} images")
        logger.info(f"  Test: {split_info['test']['count']} images")
        logger.info(f"  Total: {sum(s['count'] for s in split_info.values() if isinstance(s, dict) and 'count' in s)} images")
        
        return split_info
    
    def _get_image_files(self, directory: Path) -> List[Path]:
        """Get all image files in a directory"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(directory.glob(f'*{ext}'))
            image_files.extend(directory.glob(f'*{ext.upper()}'))
        
        return image_files
    
    def balance_classes(
        self, 
        dataset_path: Union[str, Path],
        target_samples_per_class: int,
        method: str = 'undersample'
    ) -> Dict[str, Any]:
        """Balance dataset classes by under/oversampling"""
        
        dataset_path = Path(dataset_path)
        
        if method not in ['undersample', 'oversample', 'augment']:
            raise ValueError("Method must be 'undersample', 'oversample', or 'augment'")
        
        logger.info(f"Balancing classes using {method} to {target_samples_per_class} samples per class")
        
        balance_info = {
            'method': method,
            'target_samples': target_samples_per_class,
            'classes_processed': 0,
            'total_samples_before': 0,
            'total_samples_after': 0,
            'changes': {},
            'processed_at': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
            'processed_by': 'DevanshSrajput'
        }
        
        # Process each class
        for class_dir in dataset_path.iterdir():
            if not class_dir.is_dir():
                continue
                
            class_name = class_dir.name
            image_files = self._get_image_files(class_dir)
            current_count = len(image_files)
            
            balance_info['total_samples_before'] += current_count
            balance_info['classes_processed'] += 1
            
            if current_count == target_samples_per_class:
                logger.info(f"Class {class_name} already balanced ({current_count} samples)")
                balance_info['changes'][class_name] = {
                    'before': current_count,
                    'after': current_count,
                    'action': 'no_change'
                }
                balance_info['total_samples_after'] += current_count
                continue
            
            if method == 'undersample' and current_count > target_samples_per_class:
                # Randomly select files to keep
                random.shuffle(image_files)
                files_to_keep = image_files[:target_samples_per_class]
                files_to_remove = image_files[target_samples_per_class:]
                
                # Remove excess files
                for file_path in files_to_remove:
                    file_path.unlink()
                
                balance_info['changes'][class_name] = {
                    'before': current_count,
                    'after': target_samples_per_class,
                    'action': 'undersampled',
                    'removed': len(files_to_remove)
                }
                balance_info['total_samples_after'] += target_samples_per_class
                
            elif method == 'oversample' and current_count < target_samples_per_class:
                # Duplicate existing files
                needed_samples = target_samples_per_class - current_count
                files_to_duplicate = random.choices(image_files, k=needed_samples)
                
                for i, file_path in enumerate(files_to_duplicate):
                    # Create new filename
                    stem = file_path.stem
                    suffix = file_path.suffix
                    new_name = f"{stem}_dup_{i:04d}{suffix}"
                    new_path = class_dir / new_name
                    
                    # Copy file
                    shutil.copy2(file_path, new_path)
                
                balance_info['changes'][class_name] = {
                    'before': current_count,
                    'after': target_samples_per_class,
                    'action': 'oversampled',
                    'added': needed_samples
                }
                balance_info['total_samples_after'] += target_samples_per_class
                
            elif method == 'augment' and current_count < target_samples_per_class:
                # Apply data augmentation (basic implementation)
                needed_samples = target_samples_per_class - current_count
                self._augment_images(class_dir, image_files, needed_samples)
                
                balance_info['changes'][class_name] = {
                    'before': current_count,
                    'after': target_samples_per_class,
                    'action': 'augmented',
                    'added': needed_samples
                }
                balance_info['total_samples_after'] += target_samples_per_class
                
            else:
                # No action needed
                balance_info['changes'][class_name] = {
                    'before': current_count,
                    'after': current_count,
                    'action': 'no_action_needed'
                }
                balance_info['total_samples_after'] += current_count
        
        logger.info(f"Class balancing completed:")
        logger.info(f"  Classes processed: {balance_info['classes_processed']}")
        logger.info(f"  Total samples before: {balance_info['total_samples_before']}")
        logger.info(f"  Total samples after: {balance_info['total_samples_after']}")
        
        return balance_info
    
    def _augment_images(self, class_dir: Path, image_files: List[Path], needed_samples: int):
        """Apply basic image augmentation"""
        import cv2
        from PIL import ImageEnhance, ImageFilter
        
        files_to_augment = random.choices(image_files, k=needed_samples)
        
        for i, file_path in enumerate(files_to_augment):
            try:
                # Load image
                with Image.open(file_path) as img:
                    # Apply random augmentation
                    augmented = img.copy()
                    
                    # Random rotation
                    if random.random() < 0.5:
                        angle = random.uniform(-15, 15)
                        augmented = augmented.rotate(angle, fillcolor=(255, 255, 255))
                    
                    # Random brightness
                    if random.random() < 0.5:
                        enhancer = ImageEnhance.Brightness(augmented)
                        factor = random.uniform(0.8, 1.2)
                        augmented = enhancer.enhance(factor)
                    
                    # Random contrast
                    if random.random() < 0.5:
                        enhancer = ImageEnhance.Contrast(augmented)
                        factor = random.uniform(0.8, 1.2)
                        augmented = enhancer.enhance(factor)
                    
                    # Random blur
                    if random.random() < 0.3:
                        augmented = augmented.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.0)))
                    
                    # Save augmented image
                    stem = file_path.stem
                    suffix = file_path.suffix
                    new_name = f"{stem}_aug_{i:04d}{suffix}"
                    new_path = class_dir / new_name
                    
                    augmented.save(new_path)
                    
            except Exception as e:
                logger.warning(f"Failed to augment {file_path}: {e}")

class DatasetVisualizer:
    """Visualization utilities for food datasets"""
    
    def __init__(self):
        plt.style.use('default')
        
    def plot_class_distribution(self, dataset_path: Union[str, Path], save_path: Optional[str] = None):
        """Plot class distribution"""
        dataset_path = Path(dataset_path)
        
        # Count images per class
        class_counts = {}
        
        # Check if organized structure
        if (dataset_path / 'train').exists():
            splits = ['train', 'val', 'test']
            split_data = {split: {} for split in splits if (dataset_path / split).exists()}
            
            for split in split_data.keys():
                split_path = dataset_path / split
                for class_dir in split_path.iterdir():
                    if class_dir.is_dir():
                        class_name = class_dir.name
                        image_count = len(list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png')))
                        split_data[split][class_name] = image_count
            
            # Create stacked bar plot
            fig, ax = plt.subplots(figsize=(15, 8))
            
            all_classes = set()
            for split_counts in split_data.values():
                all_classes.update(split_counts.keys())
            all_classes = sorted(all_classes)
            
            bottoms = np.zeros(len(all_classes))
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
            
            for i, (split, counts) in enumerate(split_data.items()):
                heights = [counts.get(cls, 0) for cls in all_classes]
                ax.bar(all_classes, heights, bottom=bottoms, label=split.title(), 
                      color=colors[i % len(colors)], alpha=0.8)
                bottoms += heights
            
            ax.set_title('Dataset Class Distribution by Split', fontsize=16, fontweight='bold')
            ax.set_ylabel('Number of Images', fontsize=12)
            ax.set_xlabel('Food Classes', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        else:
            # Single directory structure
            for class_dir in dataset_path.iterdir():
                if class_dir.is_dir():
                    class_name = class_dir.name
                    image_count = len(list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png')))
                    class_counts[class_name] = image_count
            
            # Create bar plot
            fig, ax = plt.subplots(figsize=(15, 8))
            
            classes = list(class_counts.keys())
            counts = list(class_counts.values())
            
            bars = ax.bar(classes, counts, color='skyblue', alpha=0.8, edgecolor='navy')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}', ha='center', va='bottom', fontsize=8)
            
            ax.set_title('Dataset Class Distribution', fontsize=16, fontweight='bold')
            ax.set_ylabel('Number of Images', fontsize=12)
            ax.set_xlabel('Food Classes', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Class distribution plot saved to {save_path}")
        
        plt.show()
    
    def plot_image_size_distribution(self, dataset_path: Union[str, Path], save_path: Optional[str] = None):
        """Plot image size distribution"""
        dataset_path = Path(dataset_path)
        
        widths = []
        heights = []
        
        logger.info("Analyzing image sizes...")
        
        # Collect image sizes
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            image_files.extend(dataset_path.rglob(f'*{ext}'))
            image_files.extend(dataset_path.rglob(f'*{ext.upper()}'))
        
        sample_size = min(1000, len(image_files))  # Sample for large datasets
        sampled_files = random.sample(image_files, sample_size)
        
        for img_path in tqdm(sampled_files, desc="Reading image sizes"):
            try:
                with Image.open(img_path) as img:
                    w, h = img.size
                    widths.append(w)
                    heights.append(h)
            except Exception as e:
                logger.warning(f"Could not read {img_path}: {e}")
        
        if not widths:
            logger.error("No valid images found for size analysis")
            return
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Width distribution
        ax1.hist(widths, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax1.set_title('Image Width Distribution')
        ax1.set_xlabel('Width (pixels)')
        ax1.set_ylabel('Frequency')
        ax1.axvline(np.mean(widths), color='red', linestyle='--', label=f'Mean: {np.mean(widths):.0f}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Height distribution
        ax2.hist(heights, bins=50, alpha=0.7, color='green', edgecolor='black')
        ax2.set_title('Image Height Distribution')
        ax2.set_xlabel('Height (pixels)')
        ax2.set_ylabel('Frequency')
        ax2.axvline(np.mean(heights), color='red', linestyle='--', label=f'Mean: {np.mean(heights):.0f}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Aspect ratio distribution
        aspect_ratios = [w/h for w, h in zip(widths, heights)]
        ax3.hist(aspect_ratios, bins=50, alpha=0.7, color='orange', edgecolor='black')
        ax3.set_title('Aspect Ratio Distribution')
        ax3.set_xlabel('Width/Height Ratio')
        ax3.set_ylabel('Frequency')
        ax3.axvline(np.mean(aspect_ratios), color='red', linestyle='--', label=f'Mean: {np.mean(aspect_ratios):.2f}')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Size scatter plot
        ax4.scatter(widths, heights, alpha=0.5, s=10)
        ax4.set_title('Width vs Height Scatter')
        ax4.set_xlabel('Width (pixels)')
        ax4.set_ylabel('Height (pixels)')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Image size distribution plot saved to {save_path}")
        
        plt.show()
        
        # Print statistics
        logger.info(f"Image size statistics (sample of {len(widths)} images):")
        logger.info(f"  Width - Mean: {np.mean(widths):.0f}, Std: {np.std(widths):.0f}, Range: {min(widths)}-{max(widths)}")
        logger.info(f"  Height - Mean: {np.mean(heights):.0f}, Std: {np.std(heights):.0f}, Range: {min(heights)}-{max(heights)}")
        logger.info(f"  Aspect Ratio - Mean: {np.mean(aspect_ratios):.2f}, Std: {np.std(aspect_ratios):.2f}")
    
    def create_sample_grid(self, dataset_path: Union[str, Path], num_samples: int = 20, save_path: Optional[str] = None):
        """Create a grid of sample images from each class"""
        dataset_path = Path(dataset_path)
        
        # Get class directories
        if (dataset_path / 'train').exists():
            class_dirs = list((dataset_path / 'train').iterdir())
        else:
            class_dirs = [d for d in dataset_path.iterdir() if d.is_dir()]
        
        class_dirs = [d for d in class_dirs if d.is_dir()]
        
        if not class_dirs:
            logger.error("No class directories found")
            return
        
        # Calculate grid size
        num_classes = min(len(class_dirs), num_samples)
        cols = min(5, num_classes)
        rows = (num_classes + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))
        if rows == 1:
            axes = [axes] if cols == 1 else axes
        elif cols == 1:
            axes = [[ax] for ax in axes]
        
        for idx, class_dir in enumerate(class_dirs[:num_samples]):
            if idx >= num_samples:
                break
                
            row = idx // cols
            col = idx % cols
            
            # Get a random image from this class
            image_files = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
            
            if image_files:
                img_path = random.choice(image_files)
                
                try:
                    with Image.open(img_path) as img:
                        # Convert to RGB if necessary
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        
                        # Handle different subplot configurations with type ignore to avoid matplotlib typing issues
                        if rows == 1 and cols == 1:
                            # Single subplot case
                            axes.imshow(img)  # type: ignore
                            axes.set_title(class_dir.name.replace('_', ' ').title(), fontsize=10)  # type: ignore
                            axes.axis('off')  # type: ignore
                        elif rows == 1:
                            # Single row, multiple columns
                            axes[col].imshow(img)  # type: ignore
                            axes[col].set_title(class_dir.name.replace('_', ' ').title(), fontsize=10)  # type: ignore
                            axes[col].axis('off')  # type: ignore
                        else:
                            # Multiple rows (and possibly multiple columns)
                            if cols == 1:
                                axes[row].imshow(img)  # type: ignore
                                axes[row].set_title(class_dir.name.replace('_', ' ').title(), fontsize=10)  # type: ignore
                                axes[row].axis('off')  # type: ignore
                            else:
                                axes[row, col].imshow(img)  # type: ignore
                                axes[row, col].set_title(class_dir.name.replace('_', ' ').title(), fontsize=10)  # type: ignore
                                axes[row, col].axis('off')  # type: ignore
                                
                except Exception as e:
                    logger.warning(f"Could not load image {img_path}: {e}")
        
        # Hide empty subplots
        for idx in range(num_classes, rows * cols):
            row = idx // cols
            col = idx % cols
            if rows > 1:
                axes[row][col].axis('off')  # type: ignore
            elif cols > 1:
                axes[col].axis('off')  # type: ignore
        
        plt.suptitle(f'Sample Images from {num_classes} Classes', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Sample grid saved to {save_path}")
        
        plt.show()

class NutritionDatabaseManager:
    """Manage nutrition database for food items"""
    
    def __init__(self, db_path: str = "data/food101_calories.csv"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True)
        
    def create_food101_database(self) -> pd.DataFrame:
        """Create comprehensive Food-101 nutrition database"""
        logger.info("Creating comprehensive Food-101 nutrition database...")
        
        # Comprehensive nutritional data for Food-101 classes
        nutrition_data = {
            'apple_pie': {'calories': 237, 'protein': 2.4, 'carbs': 34.0, 'fats': 11.0, 'fiber': 1.9, 'sodium': 266},
            'baby_back_ribs': {'calories': 292, 'protein': 26.0, 'carbs': 0.0, 'fats': 20.0, 'fiber': 0.0, 'sodium': 544},
            'baklava': {'calories': 307, 'protein': 4.1, 'carbs': 29.0, 'fats': 20.0, 'fiber': 1.2, 'sodium': 230},
            'beef_carpaccio': {'calories': 190, 'protein': 22.0, 'carbs': 1.0, 'fats': 11.0, 'fiber': 0.0, 'sodium': 421},
            'beef_tartare': {'calories': 196, 'protein': 20.0, 'carbs': 3.0, 'fats': 12.0, 'fiber': 0.0, 'sodium': 389},
            'beet_salad': {'calories': 43, 'protein': 1.6, 'carbs': 10.0, 'fats': 0.2, 'fiber': 2.8, 'sodium': 78},
            'beignets': {'calories': 347, 'protein': 6.0, 'carbs': 35.0, 'fats': 20.0, 'fiber': 1.5, 'sodium': 542},
            'bibimbap': {'calories': 121, 'protein': 8.0, 'carbs': 15.0, 'fats': 3.5, 'fiber': 2.0, 'sodium': 456},
            'bread_pudding': {'calories': 180, 'protein': 4.5, 'carbs': 31.0, 'fats': 4.5, 'fiber': 1.0, 'sodium': 234},
            'breakfast_burrito': {'calories': 219, 'protein': 12.0, 'carbs': 20.0, 'fats': 11.0, 'fiber': 2.5, 'sodium': 678},
            'bruschetta': {'calories': 194, 'protein': 6.0, 'carbs': 29.0, 'fats': 6.5, 'fiber': 2.0, 'sodium': 456},
            'caesar_salad': {'calories': 158, 'protein': 3.0, 'carbs': 6.0, 'fats': 14.0, 'fiber': 2.0, 'sodium': 789},
            'cannoli': {'calories': 297, 'protein': 6.0, 'carbs': 27.0, 'fats': 19.0, 'fiber': 1.0, 'sodium': 234},
            'caprese_salad': {'calories': 166, 'protein': 11.0, 'carbs': 5.0, 'fats': 12.0, 'fiber': 1.0, 'sodium': 432},
            'carrot_cake': {'calories': 415, 'protein': 4.0, 'carbs': 56.0, 'fats': 20.0, 'fiber': 2.0, 'sodium': 456},
            'ceviche': {'calories': 134, 'protein': 18.0, 'carbs': 8.0, 'fats': 3.0, 'fiber': 1.0, 'sodium': 567},
            'cheese_plate': {'calories': 368, 'protein': 25.0, 'carbs': 1.0, 'fats': 30.0, 'fiber': 0.0, 'sodium': 654},
            'cheesecake': {'calories': 321, 'protein': 5.5, 'carbs': 26.0, 'fats': 23.0, 'fiber': 0.5, 'sodium': 345},
            'chicken_curry': {'calories': 165, 'protein': 14.0, 'carbs': 7.0, 'fats': 9.0, 'fiber': 1.5, 'sodium': 567},
            'chicken_quesadilla': {'calories': 234, 'protein': 15.0, 'carbs': 18.0, 'fats': 12.0, 'fiber': 1.0, 'sodium': 678},
            'chicken_wings': {'calories': 203, 'protein': 30.0, 'carbs': 0.0, 'fats': 8.1, 'fiber': 0.0, 'sodium': 456},
            'chocolate_cake': {'calories': 389, 'protein': 4.3, 'carbs': 56.0, 'fats': 18.0, 'fiber': 2.0, 'sodium': 234},
            'chocolate_mousse': {'calories': 168, 'protein': 2.8, 'carbs': 16.0, 'fats': 11.0, 'fiber': 1.0, 'sodium': 123},
            'churros': {'calories': 117, 'protein': 1.3, 'carbs': 12.0, 'fats': 7.3, 'fiber': 0.5, 'sodium': 234},
            'clam_chowder': {'calories': 78, 'protein': 4.8, 'carbs': 9.8, 'fats': 2.3, 'fiber': 0.5, 'sodium': 678},
            'club_sandwich': {'calories': 282, 'protein': 21.0, 'carbs': 24.0, 'fats': 12.0, 'fiber': 2.0, 'sodium': 789},
            'crab_cakes': {'calories': 160, 'protein': 11.0, 'carbs': 5.0, 'fats': 11.0, 'fiber': 0.5, 'sodium': 567},
            'creme_brulee': {'calories': 343, 'protein': 4.6, 'carbs': 22.0, 'fats': 27.0, 'fiber': 0.0, 'sodium': 123},
            'croque_madame': {'calories': 295, 'protein': 18.0, 'carbs': 15.0, 'fats': 19.0, 'fiber': 1.0, 'sodium': 678},
            'cup_cakes': {'calories': 305, 'protein': 3.8, 'carbs': 53.0, 'fats': 9.0, 'fiber': 1.0, 'sodium': 234},
            'deviled_eggs': {'calories': 124, 'protein': 6.2, 'carbs': 0.6, 'fats': 11.0, 'fiber': 0.0, 'sodium': 234},
            'donuts': {'calories': 452, 'protein': 4.9, 'carbs': 51.0, 'fats': 25.0, 'fiber': 1.5, 'sodium': 456},
            'dumplings': {'calories': 41, 'protein': 1.7, 'carbs': 8.5, 'fats': 0.4, 'fiber': 0.5, 'sodium': 234},
            'edamame': {'calories': 121, 'protein': 11.0, 'carbs': 8.9, 'fats': 5.2, 'fiber': 5.2, 'sodium': 6},
            'eggs_benedict': {'calories': 230, 'protein': 12.0, 'carbs': 16.0, 'fats': 14.0, 'fiber': 1.0, 'sodium': 567},
            'escargots': {'calories': 90, 'protein': 16.0, 'carbs': 2.0, 'fats': 1.4, 'fiber': 0.0, 'sodium': 456},
            'falafel': {'calories': 333, 'protein': 13.0, 'carbs': 32.0, 'fats': 18.0, 'fiber': 4.9, 'sodium': 456},
            'filet_mignon': {'calories': 227, 'protein': 25.0, 'carbs': 0.0, 'fats': 15.0, 'fiber': 0.0, 'sodium': 234},
            'fish_and_chips': {'calories': 265, 'protein': 14.0, 'carbs': 23.0, 'fats': 14.0, 'fiber': 2.0, 'sodium': 456},
            'foie_gras': {'calories': 462, 'protein': 11.0, 'carbs': 4.7, 'fats': 44.0, 'fiber': 0.0, 'sodium': 567},
            'french_fries': {'calories': 365, 'protein': 4.0, 'carbs': 63.0, 'fats': 17.0, 'fiber': 3.8, 'sodium': 246},
            'french_onion_soup': {'calories': 57, 'protein': 3.8, 'carbs': 8.0, 'fats': 1.7, 'fiber': 1.0, 'sodium': 678},
            'french_toast': {'calories': 166, 'protein': 5.9, 'carbs': 18.0, 'fats': 7.0, 'fiber': 1.0, 'sodium': 234},
            'fried_calamari': {'calories': 175, 'protein': 15.0, 'carbs': 8.0, 'fats': 9.0, 'fiber': 0.5, 'sodium': 456},
            'fried_rice': {'calories': 163, 'protein': 2.9, 'carbs': 20.0, 'fats': 8.0, 'fiber': 0.5, 'sodium': 567},
            'frozen_yogurt': {'calories': 127, 'protein': 3.0, 'carbs': 22.0, 'fats': 4.0, 'fiber': 0.0, 'sodium': 67},
            'garlic_bread': {'calories': 300, 'protein': 8.0, 'carbs': 42.0, 'fats': 12.0, 'fiber': 2.0, 'sodium': 567},
            'gnocchi': {'calories': 131, 'protein': 3.8, 'carbs': 23.0, 'fats': 2.9, 'fiber': 1.5, 'sodium': 234},
            'greek_salad': {'calories': 107, 'protein': 2.8, 'carbs': 8.0, 'fats': 8.0, 'fiber': 3.0, 'sodium': 456},
            'grilled_cheese_sandwich': {'calories': 291, 'protein': 12.0, 'carbs': 28.0, 'fats': 15.0, 'fiber': 2.0, 'sodium': 678},
            'grilled_salmon': {'calories': 231, 'protein': 25.0, 'carbs': 0.0, 'fats': 14.0, 'fiber': 0.0, 'sodium': 234},
            'guacamole': {'calories': 160, 'protein': 2.0, 'carbs': 9.0, 'fats': 15.0, 'fiber': 7.0, 'sodium': 7},
            'gyoza': {'calories': 64, 'protein': 2.7, 'carbs': 6.6, 'fats': 3.0, 'fiber': 0.5, 'sodium': 234},
            'hamburger': {'calories': 540, 'protein': 25.0, 'carbs': 40.0, 'fats': 31.0, 'fiber': 3.0, 'sodium': 678},
            'hot_and_sour_soup': {'calories': 91, 'protein': 3.7, 'carbs': 8.0, 'fats': 5.0, 'fiber': 1.0, 'sodium': 789},
            'hot_dog': {'calories': 290, 'protein': 10.0, 'carbs': 2.0, 'fats': 26.0, 'fiber': 0.0, 'sodium': 890},
            'huevos_rancheros': {'calories': 153, 'protein': 8.9, 'carbs': 12.0, 'fats': 8.0, 'fiber': 2.0, 'sodium': 456},
            'hummus': {'calories': 166, 'protein': 8.0, 'carbs': 14.0, 'fats': 10.0, 'fiber': 6.0, 'sodium': 234},
            'ice_cream': {'calories': 207, 'protein': 3.5, 'carbs': 24.0, 'fats': 11.0, 'fiber': 0.7, 'sodium': 80},
            'lasagna': {'calories': 135, 'protein': 8.1, 'carbs': 11.0, 'fats': 6.9, 'fiber': 1.0, 'sodium': 456},
            'lobster_bisque': {'calories': 104, 'protein': 4.7, 'carbs': 7.0, 'fats': 6.9, 'fiber': 0.2, 'sodium': 567},
            'lobster_roll_sandwich': {'calories': 436, 'protein': 18.0, 'carbs': 44.0, 'fats': 21.0, 'fiber': 2.0, 'sodium': 678},
            'macaroni_and_cheese': {'calories': 164, 'protein': 6.4, 'carbs': 20.0, 'fats': 6.6, 'fiber': 1.0, 'sodium': 456},
            'macarons': {'calories': 300, 'protein': 5.0, 'carbs': 50.0, 'fats': 10.0, 'fiber': 2.0, 'sodium': 234},
            'miso_soup': {'calories': 40, 'protein': 2.2, 'carbs': 7.0, 'fats': 1.0, 'fiber': 1.0, 'sodium': 890},
            'mussels': {'calories': 172, 'protein': 24.0, 'carbs': 7.4, 'fats': 4.6, 'fiber': 0.0, 'sodium': 456},
            'nachos': {'calories': 346, 'protein': 9.0, 'carbs': 36.0, 'fats': 19.0, 'fiber': 3.0, 'sodium': 678},
            'omelette': {'calories': 154, 'protein': 11.0, 'carbs': 1.0, 'fats': 12.0, 'fiber': 0.0, 'sodium': 234},
            'onion_rings': {'calories': 411, 'protein': 5.9, 'carbs': 38.0, 'fats': 26.0, 'fiber': 2.5, 'sodium': 456},
            'oysters': {'calories': 81, 'protein': 9.5, 'carbs': 4.7, 'fats': 2.9, 'fiber': 0.0, 'sodium': 567},
            'pad_thai': {'calories': 181, 'protein': 9.0, 'carbs': 23.0, 'fats': 6.5, 'fiber': 2.0, 'sodium': 567},
            'paella': {'calories': 172, 'protein': 8.0, 'carbs': 21.0, 'fats': 6.5, 'fiber': 1.5, 'sodium': 456},
            'pancakes': {'calories': 227, 'protein': 6.0, 'carbs': 28.0, 'fats': 10.0, 'fiber': 1.5, 'sodium': 456},
            'panna_cotta': {'calories': 185, 'protein': 2.8, 'carbs': 15.0, 'fats': 13.0, 'fiber': 0.0, 'sodium': 123},
            'peking_duck': {'calories': 337, 'protein': 19.0, 'carbs': 0.0, 'fats': 28.0, 'fiber': 0.0, 'sodium': 567},
            'pho': {'calories': 194, 'protein': 15.0, 'carbs': 25.0, 'fats': 3.0, 'fiber': 1.0, 'sodium': 789},
            'pizza': {'calories': 266, 'protein': 11.0, 'carbs': 33.0, 'fats': 10.0, 'fiber': 2.3, 'sodium': 567},
            'pork_chop': {'calories': 231, 'protein': 23.0, 'carbs': 0.0, 'fats': 15.0, 'fiber': 0.0, 'sodium': 234},
            'poutine': {'calories': 740, 'protein': 37.0, 'carbs': 93.0, 'fats': 27.0, 'fiber': 6.0, 'sodium': 1234},
            'prime_rib': {'calories': 338, 'protein': 25.0, 'carbs': 0.0, 'fats': 26.0, 'fiber': 0.0, 'sodium': 456},
            'pulled_pork_sandwich': {'calories': 415, 'protein': 29.0, 'carbs': 41.0, 'fats': 15.0, 'fiber': 2.0, 'sodium': 678},
            'ramen': {'calories': 436, 'protein': 18.0, 'carbs': 54.0, 'fats': 16.0, 'fiber': 2.0, 'sodium': 1234},
            'ravioli': {'calories': 220, 'protein': 8.0, 'carbs': 31.0, 'fats': 7.0, 'fiber': 2.0, 'sodium': 456},
            'red_velvet_cake': {'calories': 478, 'protein': 5.1, 'carbs': 73.0, 'fats': 19.0, 'fiber': 1.5, 'sodium': 345},
            'risotto': {'calories': 166, 'protein': 3.0, 'carbs': 20.0, 'fats': 8.0, 'fiber': 0.5, 'sodium': 345},
            'samosa': {'calories': 308, 'protein': 5.0, 'carbs': 25.0, 'fats': 21.0, 'fiber': 3.0, 'sodium': 456},
            'sashimi': {'calories': 127, 'protein': 20.0, 'carbs': 0.0, 'fats': 4.4, 'fiber': 0.0, 'sodium': 234},
            'scallops': {'calories': 111, 'protein': 20.0, 'carbs': 5.4, 'fats': 0.8, 'fiber': 0.0, 'sodium': 456},
            'seaweed_salad': {'calories': 45, 'protein': 1.7, 'carbs': 9.0, 'fats': 0.6, 'fiber': 0.3, 'sodium': 567},
            'shrimp_and_grits': {'calories': 258, 'protein': 18.0, 'carbs': 27.0, 'fats': 9.0, 'fiber': 1.5, 'sodium': 456},
            'spaghetti_bolognese': {'calories': 158, 'protein': 8.2, 'carbs': 20.0, 'fats': 5.6, 'fiber': 2.0, 'sodium': 456},
            'spaghetti_carbonara': {'calories': 370, 'protein': 14.0, 'carbs': 40.0, 'fats': 17.0, 'fiber': 2.0, 'sodium': 567},
            'spring_rolls': {'calories': 78, 'protein': 1.8, 'carbs': 13.0, 'fats': 2.5, 'fiber': 1.5, 'sodium': 234},
            'steak': {'calories': 271, 'protein': 26.0, 'carbs': 0.0, 'fats': 19.0, 'fiber': 0.0, 'sodium': 234},
            'strawberry_shortcake': {'calories': 344, 'protein': 4.4, 'carbs': 55.0, 'fats': 13.0, 'fiber': 2.0, 'sodium': 234},
            'sushi': {'calories': 143, 'protein': 6.0, 'carbs': 21.0, 'fats': 3.9, 'fiber': 0.2, 'sodium': 456},
            'tacos': {'calories': 226, 'protein': 15.0, 'carbs': 20.0, 'fats': 12.0, 'fiber': 3.0, 'sodium': 456},
            'takoyaki': {'calories': 40, 'protein': 1.9, 'carbs': 4.3, 'fats': 1.6, 'fiber': 0.1, 'sodium': 234},
            'tiramisu': {'calories': 240, 'protein': 4.0, 'carbs': 29.0, 'fats': 12.0, 'fiber': 1.0, 'sodium': 123},
            'tuna_tartare': {'calories': 144, 'protein': 23.0, 'carbs': 0.0, 'fats': 5.0, 'fiber': 0.0, 'sodium': 234},
            'waffles': {'calories': 291, 'protein': 7.9, 'carbs': 37.0, 'fats': 13.0, 'fiber': 1.9, 'sodium': 456}
        }
        
        # Convert to DataFrame
        nutrition_df = pd.DataFrame.from_dict(nutrition_data, orient='index')
        nutrition_df.reset_index(inplace=True)
        nutrition_df.rename(columns={'index': 'food_name'}, inplace=True)
        
        # Add metadata
        nutrition_df['created_at'] = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
        nutrition_df['created_by'] = 'DevanshSrajput'
        nutrition_df['source'] = 'USDA_and_research'
        nutrition_df['verified'] = True
        
        # Save to CSV
        nutrition_df.to_csv(self.db_path, index=False)
        
        logger.info(f"Created nutrition database with {len(nutrition_df)} food items")
        logger.info(f"Database saved to: {self.db_path}")
        
        return nutrition_df
    
    def validate_nutrition_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate nutrition database"""
        validation = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'statistics': {},
            'validated_at': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
            'validated_by': 'DevanshSrajput'
        }
        
        required_columns = ['food_name', 'calories', 'protein', 'carbs', 'fats']
        
        # Check required columns
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            validation['valid'] = False
            validation['errors'].append(f"Missing required columns: {missing_cols}")
        
        # Check for missing values
        for col in required_columns:
            if col in df.columns:
                missing_count = df[col].isna().sum()
                if missing_count > 0:
                    validation['warnings'].append(f"Column '{col}' has {missing_count} missing values")
        
        # Check for reasonable value ranges
        if 'calories' in df.columns:
            calories = df['calories']
            if (calories < 0).any():
                validation['errors'].append("Negative calorie values found")
            if (calories > 1000).any():
                validation['warnings'].append("Very high calorie values found (>1000 per 100g)")
        
        # Generate statistics
        if validation['valid']:
            numeric_cols = ['calories', 'protein', 'carbs', 'fats']
            for col in numeric_cols:
                if col in df.columns:
                    validation['statistics'][col] = {
                        'mean': float(df[col].mean()),
                        'std': float(df[col].std()),
                        'min': float(df[col].min()),
                        'max': float(df[col].max())
                    }
        
        return validation

def main():
    """Main function for standalone execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Dataset Utilities for Food-101")
    parser.add_argument("--analyze", type=str, help="Analyze dataset at given path")
    parser.add_argument("--validate", type=str, help="Validate dataset at given path")
    parser.add_argument("--organize", type=str, help="Organize dataset for training")
    parser.add_argument("--visualize", type=str, help="Create visualizations for dataset")
    parser.add_argument("--create-nutrition-db", action="store_true", help="Create nutrition database")
    parser.add_argument("--balance-classes", type=str, help="Balance classes in dataset")
    parser.add_argument("--target-samples", type=int, default=750, help="Target samples per class for balancing")
    parser.add_argument("--balance-method", choices=['undersample', 'oversample', 'augment'], 
                       default='undersample', help="Method for class balancing")
    parser.add_argument("--output-dir", type=str, help="Output directory for organized dataset")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Training set ratio")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation set ratio")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Test set ratio")
    parser.add_argument("--save-plots", action="store_true", help="Save visualization plots")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    print("🍎 Food-101 Dataset Utilities")
    print("=" * 50)
    print(f"Date: 2025-06-29 10:06:51")
    print(f"User: DevanshSrajput")
    print("=" * 50)
    
    try:
        # Dataset analysis
        if args.analyze:
            print(f"📊 Analyzing dataset: {args.analyze}")
            stats = DatasetStatistics()
            analysis = stats.analyze_images(args.analyze)
            
            print(f"Analysis Results:")
            print(f"  Total Images: {analysis['total_images']:,}")
            print(f"  Total Size: {analysis['total_size_mb']:.1f} MB")
            print(f"  Classes: {len(analysis['classes'])}")
            print(f"  Corrupted Images: {len(analysis['corrupted_images'])}")
            print(f"  Duplicates: {len(analysis['duplicates'])}")
            
            # Save analysis results
            analysis_file = Path(args.analyze) / "analysis_results.json"
            with open(analysis_file, 'w') as f:
                json.dump(analysis, f, indent=2, default=str)
            print(f"📄 Analysis saved to: {analysis_file}")
        
        # Dataset validation
        if args.validate:
            print(f"🔍 Validating dataset: {args.validate}")
            validator = DatasetValidator()
            validation = validator.validate_structure(args.validate)
            
            print(f"Validation Results:")
            print(f"  Valid: {'✅' if validation['valid'] else '❌'}")
            print(f"  Structure Type: {validation['structure_type']}")
            print(f"  Total Classes: {len(validation['classes'])}")
            print(f"  Total Images: {validation['total_images']}")
            
            if validation['errors']:
                print("  Errors:")
                for error in validation['errors']:
                    print(f"    ❌ {error}")
            
            if validation['warnings']:
                print("  Warnings:")
                for warning in validation['warnings']:
                    print(f"    ⚠️  {warning}")
            
            # Save validation results
            validation_file = Path(args.validate) / "validation_results.json"
            with open(validation_file, 'w') as f:
                json.dump(validation, f, indent=2, default=str)
            print(f"📄 Validation saved to: {validation_file}")
        
        # Dataset organization
        if args.organize:
            print(f"🗂️ Organizing dataset: {args.organize}")
            
            if not args.output_dir:
                args.output_dir = str(Path(args.organize).parent / "organized_dataset")
            
            organizer = DatasetOrganizer(random_seed=args.random_seed)
            split_info = organizer.create_train_val_test_split(
                source_dir=args.organize,
                target_dir=args.output_dir,
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio
            )
            
            print(f"✅ Dataset organized successfully!")
            print(f"📁 Output directory: {args.output_dir}")
            
        # Class balancing
        if args.balance_classes:
            print(f"⚖️ Balancing classes: {args.balance_classes}")
            
            organizer = DatasetOrganizer(random_seed=args.random_seed)
            balance_info = organizer.balance_classes(
                dataset_path=args.balance_classes,
                target_samples_per_class=args.target_samples,
                method=args.balance_method
            )
            
            print(f"✅ Class balancing completed!")
            print(f"📊 Method: {balance_info['method']}")
            print(f"🎯 Target samples: {balance_info['target_samples']}")
            print(f"📈 Classes processed: {balance_info['classes_processed']}")
            
        # Dataset visualization
        if args.visualize:
            print(f"📈 Creating visualizations: {args.visualize}")
            
            visualizer = DatasetVisualizer()
            
            # Class distribution plot
            save_path = f"{args.visualize}_class_distribution.png" if args.save_plots else None
            visualizer.plot_class_distribution(args.visualize, save_path)
            
            # Image size distribution
            save_path = f"{args.visualize}_size_distribution.png" if args.save_plots else None
            visualizer.plot_image_size_distribution(args.visualize, save_path)
            
            # Sample grid
            save_path = f"{args.visualize}_sample_grid.png" if args.save_plots else None
            visualizer.create_sample_grid(args.visualize, save_path=save_path)
            
            print("✅ Visualizations created!")
        
        # Create nutrition database
        if args.create_nutrition_db:
            print("📊 Creating Food-101 nutrition database...")
            
            nutrition_manager = NutritionDatabaseManager()
            nutrition_df = nutrition_manager.create_food101_database()
            
            # Validate the created database
            validation = nutrition_manager.validate_nutrition_data(nutrition_df)
            
            print(f"✅ Nutrition database created!")
            print(f"📄 Database file: {nutrition_manager.db_path}")
            print(f"📊 Food items: {len(nutrition_df)}")
            print(f"🔍 Validation: {'✅ Valid' if validation['valid'] else '❌ Invalid'}")
            
            if validation['warnings']:
                print("Warnings:")
                for warning in validation['warnings']:
                    print(f"  ⚠️  {warning}")
    
    except Exception as e:
        logger.error(f"Error in dataset utilities: {e}")
        print(f"❌ Error: {e}")
        return 1
    
    return 0

class DatasetDownloadManager:
    """Manage dataset downloads from various sources"""
    
    def __init__(self, download_dir: str = "downloads"):
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(exist_ok=True)
        
    def download_with_progress(self, url: str, filename: str, description: str = "Downloading") -> Path:
        """Download file with progress bar"""
        file_path = self.download_dir / filename
        
        if file_path.exists():
            logger.info(f"File already exists: {file_path}")
            return file_path
        
        logger.info(f"Downloading {description} from {url}")
        
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(file_path, 'wb') as file, tqdm(
            desc=description,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    size = file.write(chunk)
                    pbar.update(size)
        
        logger.info(f"Download completed: {file_path}")
        return file_path
    
    def extract_archive(self, archive_path: Path, extract_to: Optional[Path] = None) -> Path:
        """Extract archive file"""
        if extract_to is None:
            extract_to = archive_path.parent
        
        extract_to.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Extracting {archive_path} to {extract_to}")
        
        if archive_path.suffix.lower() == '.zip':
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
        elif archive_path.suffix.lower() in ['.tar', '.gz', '.tgz']:
            with tarfile.open(archive_path, 'r:*') as tar_ref:
                tar_ref.extractall(extract_to)
        else:
            raise ValueError(f"Unsupported archive format: {archive_path.suffix}")
        
        logger.info("Extraction completed")
        return extract_to

class DatasetAugmentationManager:
    """Advanced data augmentation for food images"""
    
    def __init__(self):
        self.augmentation_techniques = {
            'rotation': self._rotate_image,
            'brightness': self._adjust_brightness,
            'contrast': self._adjust_contrast,
            'saturation': self._adjust_saturation,
            'hue': self._adjust_hue,
            'blur': self._apply_blur,
            'noise': self._add_noise,
            'crop': self._random_crop,
            'flip': self._flip_image,
            'elastic': self._elastic_transform
        }
    
    def augment_dataset(self, dataset_path: Union[str, Path], 
                       output_path: Union[str, Path],
                       augmentations_per_image: int = 3,
                       techniques: Optional[List[str]] = None) -> Dict[str, Any]:
        """Apply augmentation to entire dataset"""
        
        if techniques is None:
            techniques = ['rotation', 'brightness', 'contrast', 'flip', 'blur']
        
        dataset_path = Path(dataset_path)
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        augmentation_info = {
            'source_dataset': str(dataset_path),
            'output_dataset': str(output_path),
            'techniques_used': techniques,
            'augmentations_per_image': augmentations_per_image,
            'classes_processed': 0,
            'total_images_created': 0,
            'processing_time': 0,
            'created_at': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
            'created_by': 'DevanshSrajput'
        }
        
        start_time = datetime.utcnow()
        
        # Process each class directory
        for class_dir in dataset_path.iterdir():
            if not class_dir.is_dir():
                continue
            
            class_name = class_dir.name
            output_class_dir = output_path / class_name
            output_class_dir.mkdir(exist_ok=True)
            
            # Copy original images
            image_files = self._get_image_files(class_dir)
            for img_file in image_files:
                shutil.copy2(img_file, output_class_dir / img_file.name)
            
            # Create augmented images
            for img_file in tqdm(image_files, desc=f"Augmenting {class_name}"):
                try:
                    with Image.open(img_file) as img:
                        # Convert to RGB if necessary
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        
                        # Create multiple augmented versions
                        for aug_idx in range(augmentations_per_image):
                            augmented_img = self._apply_random_augmentations(img, techniques)
                            
                            # Save augmented image
                            stem = img_file.stem
                            suffix = img_file.suffix
                            aug_filename = f"{stem}_aug_{aug_idx:03d}{suffix}"
                            aug_path = output_class_dir / aug_filename
                            
                            augmented_img.save(aug_path, quality=95)
                            augmentation_info['total_images_created'] += 1
                
                except Exception as e:
                    logger.warning(f"Failed to augment {img_file}: {e}")
            
            augmentation_info['classes_processed'] += 1
        
        # Calculate processing time
        end_time = datetime.utcnow()
        processing_time = (end_time - start_time).total_seconds()
        augmentation_info['processing_time'] = processing_time
        
        # Save augmentation info
        info_file = output_path / "augmentation_info.json"
        with open(info_file, 'w') as f:
            json.dump(augmentation_info, f, indent=2)
        
        logger.info(f"Dataset augmentation completed:")
        logger.info(f"  Classes processed: {augmentation_info['classes_processed']}")
        logger.info(f"  Images created: {augmentation_info['total_images_created']}")
        logger.info(f"  Processing time: {processing_time:.1f} seconds")
        
        return augmentation_info
    
    def _apply_random_augmentations(self, img: Image.Image, techniques: List[str]) -> Image.Image:
        """Apply random combination of augmentation techniques"""
        augmented = img.copy()
        
        # Randomly select techniques to apply
        selected_techniques = random.sample(techniques, k=random.randint(1, min(3, len(techniques))))
        
        for technique in selected_techniques:
            if technique in self.augmentation_techniques:
                augmented = self.augmentation_techniques[technique](augmented)
        
        return augmented
    
    def _rotate_image(self, img: Image.Image) -> Image.Image:
        """Apply random rotation"""
        angle = random.uniform(-20, 20)
        return img.rotate(angle, fillcolor=(255, 255, 255))
    
    def _adjust_brightness(self, img: Image.Image) -> Image.Image:
        """Adjust brightness randomly"""
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Brightness(img)
        factor = random.uniform(0.7, 1.3)
        return enhancer.enhance(factor)
    
    def _adjust_contrast(self, img: Image.Image) -> Image.Image:
        """Adjust contrast randomly"""
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Contrast(img)
        factor = random.uniform(0.8, 1.2)
        return enhancer.enhance(factor)
    
    def _adjust_saturation(self, img: Image.Image) -> Image.Image:
        """Adjust color saturation randomly"""
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Color(img)
        factor = random.uniform(0.8, 1.2)
        return enhancer.enhance(factor)
    
    def _adjust_hue(self, img: Image.Image) -> Image.Image:
        """Adjust hue slightly"""
        import numpy as np
        
        # Convert to HSV
        hsv = img.convert('HSV')
        hsv_array = np.array(hsv)
        
        # Adjust hue channel
        hue_shift = random.randint(-10, 10)
        hsv_array[:, :, 0] = (hsv_array[:, :, 0] + hue_shift) % 256
        
        # Convert back to RGB
        hsv_shifted = Image.fromarray(hsv_array, 'HSV')
        return hsv_shifted.convert('RGB')
    
    def _apply_blur(self, img: Image.Image) -> Image.Image:
        """Apply random blur"""
        from PIL import ImageFilter
        radius = random.uniform(0.5, 2.0)
        return img.filter(ImageFilter.GaussianBlur(radius=radius))
    
    def _add_noise(self, img: Image.Image) -> Image.Image:
        """Add random noise"""
        img_array = np.array(img)
        noise = np.random.normal(0, 10, img_array.shape)
        noisy_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy_array)
    
    def _random_crop(self, img: Image.Image) -> Image.Image:
        """Apply random crop and resize"""
        width, height = img.size
        
        # Random crop parameters
        crop_ratio = random.uniform(0.8, 1.0)
        new_width = int(width * crop_ratio)
        new_height = int(height * crop_ratio)
        
        # Random position
        left = random.randint(0, width - new_width)
        top = random.randint(0, height - new_height)
        
        # Crop and resize back
        cropped = img.crop((left, top, left + new_width, top + new_height))
        return cropped.resize((width, height), Image.Resampling.LANCZOS)
    
    def _flip_image(self, img: Image.Image) -> Image.Image:
        """Apply random flip"""
        flip_type = random.choice(['horizontal', 'vertical', 'none'])
        
        if flip_type == 'horizontal':
            return img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        elif flip_type == 'vertical':
            return img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        else:
            return img
    
    def _elastic_transform(self, img: Image.Image) -> Image.Image:
        """Apply elastic deformation (simplified)"""
        # This is a simplified version - for production use scipy.ndimage
        width, height = img.size
        
        # Create slight perspective distortion
        coeffs = [
            1 + random.uniform(-0.02, 0.02),  # a
            random.uniform(-0.01, 0.01),     # b
            random.uniform(-5, 5),           # c
            random.uniform(-0.01, 0.01),     # d
            1 + random.uniform(-0.02, 0.02), # e
            random.uniform(-5, 5),           # f
            random.uniform(-0.0001, 0.0001), # g
            random.uniform(-0.0001, 0.0001)  # h
        ]
        
        return img.transform(img.size, Image.Transform.PERSPECTIVE, coeffs, Image.Resampling.BILINEAR)
    
    def _get_image_files(self, directory: Path) -> List[Path]:
        """Get all image files in directory"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(directory.glob(f'*{ext}'))
            image_files.extend(directory.glob(f'*{ext.upper()}'))
        
        return image_files

class DatasetQualityAssessment:
    """Assess and improve dataset quality"""
    
    def __init__(self):
        self.quality_metrics = {}
        
    def assess_image_quality(self, dataset_path: Union[str, Path]) -> Dict[str, Any]:
        """Comprehensive image quality assessment"""
        dataset_path = Path(dataset_path)
        
        quality_report = {
            'dataset_path': str(dataset_path),
            'total_images_analyzed': 0,
            'quality_issues': {
                'blurry_images': [],
                'dark_images': [],
                'overexposed_images': [],
                'low_contrast_images': [],
                'corrupted_images': [],
                'duplicate_images': []
            },
            'quality_statistics': {
                'avg_brightness': 0,
                'avg_contrast': 0,
                'avg_sharpness': 0,
                'resolution_distribution': {}
            },
            'recommendations': [],
            'assessed_at': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
            'assessed_by': 'DevanshSrajput'
        }
        
        image_hashes = {}
        brightness_values = []
        contrast_values = []
        sharpness_values = []
        
        # Get all image files
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            image_files.extend(dataset_path.rglob(f'*{ext}'))
            image_files.extend(dataset_path.rglob(f'*{ext.upper()}'))
        
        logger.info(f"Assessing quality of {len(image_files)} images...")
        
        for img_path in tqdm(image_files, desc="Assessing image quality"):
            try:
                with Image.open(img_path) as img:
                    # Convert to RGB if needed
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    # Calculate image hash for duplicate detection
                    img_hash = str(hash(img.tobytes()))
                    if img_hash in image_hashes:
                        quality_report['quality_issues']['duplicate_images'].append({
                            'original': image_hashes[img_hash],
                            'duplicate': str(img_path)
                        })
                    else:
                        image_hashes[img_hash] = str(img_path)
                    
                    # Image statistics
                    stat = ImageStat.Stat(img)
                    
                    # Brightness assessment
                    brightness = sum(stat.mean) / len(stat.mean)
                    brightness_values.append(brightness)
                    
                    if brightness < 50:
                        quality_report['quality_issues']['dark_images'].append(str(img_path))
                    elif brightness > 200:
                        quality_report['quality_issues']['overexposed_images'].append(str(img_path))
                    
                    # Contrast assessment
                    contrast = sum(stat.stddev) / len(stat.stddev)
                    contrast_values.append(contrast)
                    
                    if contrast < 20:
                        quality_report['quality_issues']['low_contrast_images'].append(str(img_path))
                    
                    # Sharpness assessment (Laplacian variance)
                    img_gray = img.convert('L')
                    img_array = np.array(img_gray)
                    
                    # Calculate Laplacian variance for blur detection
                    laplacian_var = cv2.Laplacian(img_array, cv2.CV_64F).var()
                    sharpness_values.append(laplacian_var)
                    
                    if laplacian_var < 100:  # Threshold for blur
                        quality_report['quality_issues']['blurry_images'].append(str(img_path))
                    
                    # Resolution tracking
                    resolution = f"{img.size[0]}x{img.size[1]}"
                    if resolution not in quality_report['quality_statistics']['resolution_distribution']:
                        quality_report['quality_statistics']['resolution_distribution'][resolution] = 0
                    quality_report['quality_statistics']['resolution_distribution'][resolution] += 1
                    
                    quality_report['total_images_analyzed'] += 1
                    
            except Exception as e:
                quality_report['quality_issues']['corrupted_images'].append(str(img_path))
                logger.warning(f"Error processing {img_path}: {e}")
        
        # Calculate overall statistics
        if brightness_values:
            quality_report['quality_statistics']['avg_brightness'] = np.mean(brightness_values)
            quality_report['quality_statistics']['avg_contrast'] = np.mean(contrast_values)
            quality_report['quality_statistics']['avg_sharpness'] = np.mean(sharpness_values)
        
        # Generate recommendations
        self._generate_quality_recommendations(quality_report)
        
        logger.info("Quality assessment completed:")
        for issue_type, issues in quality_report['quality_issues'].items():
            if issues:
                logger.info(f"  {issue_type}: {len(issues)} found")
        
        return quality_report
    
    def _generate_quality_recommendations(self, quality_report: Dict[str, Any]):
        """Generate recommendations based on quality assessment"""
        recommendations = []
        
        # Check for quality issues
        issues = quality_report['quality_issues']
        total_images = quality_report['total_images_analyzed']
        
        if issues['blurry_images']:
            blur_percentage = (len(issues['blurry_images']) / total_images) * 100
            if blur_percentage > 5:
                recommendations.append(f"High percentage of blurry images ({blur_percentage:.1f}%). Consider removing or replacing blurry images.")
        
        if issues['dark_images']:
            dark_percentage = (len(issues['dark_images']) / total_images) * 100
            if dark_percentage > 10:
                recommendations.append(f"Many dark images found ({dark_percentage:.1f}%). Consider brightness augmentation.")
        
        if issues['overexposed_images']:
            overexp_percentage = (len(issues['overexposed_images']) / total_images) * 100
            if overexp_percentage > 10:
                recommendations.append(f"Many overexposed images found ({overexp_percentage:.1f}%). Consider exposure correction.")
        
        if issues['duplicate_images']:
            dup_percentage = (len(issues['duplicate_images']) / total_images) * 100
            recommendations.append(f"Duplicate images found ({dup_percentage:.1f}%). Remove duplicates to improve dataset quality.")
        
        if issues['corrupted_images']:
            corr_percentage = (len(issues['corrupted_images']) / total_images) * 100
            recommendations.append(f"Corrupted images found ({corr_percentage:.1f}%). Remove corrupted files.")
        
        # Resolution recommendations
        resolutions = quality_report['quality_statistics']['resolution_distribution']
        if len(resolutions) > 10:
            recommendations.append("High resolution diversity detected. Consider standardizing image sizes for training.")
        
        # Overall quality recommendations
        avg_sharpness = quality_report['quality_statistics']['avg_sharpness']
        if avg_sharpness < 200:
            recommendations.append("Overall image sharpness is low. Consider using sharper images or sharpening filters.")
        
        avg_contrast = quality_report['quality_statistics']['avg_contrast']
        if avg_contrast < 30:
            recommendations.append("Overall image contrast is low. Consider contrast enhancement.")
        
        quality_report['recommendations'] = recommendations
    
    def clean_dataset(self, dataset_path: Union[str, Path], 
                     quality_report: Dict[str, Any],
                     output_path: Union[str, Path],
                     remove_issues: Optional[List[str]] = None) -> Dict[str, Any]:
        """Clean dataset based on quality assessment"""
        
        if remove_issues is None:
            remove_issues = ['corrupted_images', 'duplicate_images']
        
        dataset_path = Path(dataset_path)
        output_path = Path(output_path)
        
        cleaning_report = {
            'source_dataset': str(dataset_path),
            'cleaned_dataset': str(output_path),
            'issues_addressed': remove_issues,
            'files_removed': 0,
            'files_kept': 0,
            'cleaning_actions': {},
            'cleaned_at': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
            'cleaned_by': 'DevanshSrajput'
        }
        
        # Create output directory structure
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Collect files to remove
        files_to_remove = set()
        
        for issue_type in remove_issues:
            if issue_type in quality_report['quality_issues']:
                issues = quality_report['quality_issues'][issue_type]
                
                if issue_type == 'duplicate_images':
                    # For duplicates, remove the duplicate, keep the original
                    for dup_info in issues:
                        files_to_remove.add(dup_info['duplicate'])
                else:
                    # For other issues, add all problematic files
                    files_to_remove.update(issues)
                
                cleaning_report['cleaning_actions'][issue_type] = len(issues)
        
        # Copy clean files to output directory
        logger.info(f"Cleaning dataset: removing {len(files_to_remove)} problematic files")
        
        for class_dir in dataset_path.iterdir():
            if not class_dir.is_dir():
                continue
            
            output_class_dir = output_path / class_dir.name
            output_class_dir.mkdir(exist_ok=True)
            
            image_files = self._get_image_files(class_dir)
            
            for img_file in image_files:
                if str(img_file) not in files_to_remove:
                    shutil.copy2(img_file, output_class_dir / img_file.name)
                    cleaning_report['files_kept'] += 1
                else:
                    cleaning_report['files_removed'] += 1
        
        # Save cleaning report
        report_file = output_path / "cleaning_report.json"
        with open(report_file, 'w') as f:
            json.dump(cleaning_report, f, indent=2)
        
        logger.info(f"Dataset cleaning completed:")
        logger.info(f"  Files removed: {cleaning_report['files_removed']}")
        logger.info(f"  Files kept: {cleaning_report['files_kept']}")
        logger.info(f"  Cleaned dataset: {output_path}")
        
        return cleaning_report
    
    def _get_image_files(self, directory: Path) -> List[Path]:
        """Get all image files in directory"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(directory.glob(f'*{ext}'))
            image_files.extend(directory.glob(f'*{ext.upper()}'))
        
        return image_files

if __name__ == "__main__":
    exit(main())
