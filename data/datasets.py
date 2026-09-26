"""
Dataset for DINOv2 training with efficient sharding and corruption handling.
"""

import os
import torch
from torch.utils.data import IterableDataset
from PIL import Image, PngImagePlugin, ImageFile
import zipfile
import io
import random
import numpy as np
import json
import glob
import time
import pickle
import fcntl
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Iterator, Set
import torch.distributed as dist

from .transforms import TMEDinoTransforms


class MemoryEfficientShardedPathologyDataset(IterableDataset):
    """
    Memory-efficient dataset with proper sharding for distributed training.
    Handles corrupted files and supports resuming from checkpoints.
    
    Args:
        base_dir: Root directory containing zip files
        index_file: Path to dataset index pickle file
        worker_id: Worker ID (set dynamically)
        num_workers: Number of workers (set dynamically)
        rank: Distributed training rank
        world_size: Total number of processes
        seed: Random seed
        global_size: Size of global crops
        local_size: Size of local crops
        local_crop_scale: Scale range for local crops
        global_crop_scale: Scale range for global crops
        n_local_crops: Number of local crops per image
        mean: Normalization mean
        std: Normalization std
        corruptions_dir: Directory containing corruption logs
    """
    def __init__(
        self,
        base_dir: str,
        index_file: str = "dataset_index.pkl",
        worker_id: int = 0,
        num_workers: int = 1,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 42,
        global_size: int = 224,
        local_size: int = 96,
        local_crop_scale: tuple = (0.05, 0.32),
        global_crop_scale: tuple = (0.32, 1.0),
        n_local_crops: int = 2,
        mean: tuple = (0.6816, 0.5640, 0.7232),
        std: tuple = (0.1617, 0.1714, 0.1389),
        corruptions_dir: str = "corruption_results",
        use_pathology_recipe: bool = False,
        ect_probability: float = 0.4,
        zip_interleave: int = 16,
        emit_scout: bool = False,
        scout_pool_mode: bool = False,
    ):
        super().__init__()
        self.base_dir = base_dir
        self.index_file = index_file
        self.worker_id = worker_id
        self.num_workers = num_workers
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.zip_interleave = zip_interleave
        self.corruptions_dir = corruptions_dir

        # Set parameters for transforms
        self.global_size = global_size
        self.local_size = local_size
        self.n_local_crops = n_local_crops
        self.local_crop_scale = local_crop_scale
        self.global_crop_scale = global_crop_scale
        self.mean = mean
        self.std = std
        self.use_pathology_recipe = use_pathology_recipe
        self.ect_probability = ect_probability

        # Initialize transforms
        self.transforms = TMEDinoTransforms(
            local_size=local_size,
            global_size=global_size,
            local_crop_scale=local_crop_scale,
            global_crop_scale=global_crop_scale,
            n_local_crops=n_local_crops,
            mean=mean,
            std=std,
            use_pathology_recipe=use_pathology_recipe,
            ect_probability=ect_probability,
            emit_scout=emit_scout,
            scout_pool_mode=scout_pool_mode,
        )

        # Setup corruption logging
        self.corruption_log_file = "runtime_corrupted_files.json"
        self.corruption_lock_file = f"{self.corruption_log_file}.lock"
        if not os.path.exists(self.corruption_lock_file):
            with open(self.corruption_lock_file, 'w') as f:
                pass
        
        # Load pre-known corrupted files
        self.corrupted_zip_files = self._load_known_corrupted_files()
        print(f"Loaded {len(self.corrupted_zip_files)} known corrupted zip files to exclude")
        
        # Load metadata
        self.index_metadata = self._load_index_metadata()
        self.total_images = self.index_metadata['total_images']
        
        # Filter corrupted files
        self._filter_corrupted_zip_files()
        
        # Sharding is computed lazily in __iter__ from get_worker_info(); no
        # precomputed shard state is stored. _epoch varies the per-worker shuffle.
        self._epoch = 0
        
        # Resume position (global; divided by live worker count in __iter__).
        self._resume_global = 0
        
        # Rate limiting
        self.error_count = 0
        self.last_error_time = time.time()
        self.max_errors_per_minute = 10
    
    def set_worker_info(self, worker_id: int, num_workers: int):
        """Vestigial: __iter__ now reads worker info from get_worker_info(). Kept as a
        no-op-compatible setter so existing callers (e.g. worker_init_fn) don't break."""
        self.worker_id = worker_id
        self.num_workers = num_workers
        print(f"Worker info set: worker_id={worker_id}, num_workers={num_workers}")

    def set_resume_position(self, global_samples_processed: int):
        """Store the global resume position. __iter__ divides it by the live worker
        count to get this worker's per-shard skip."""
        self._resume_global = global_samples_processed
        print(f"Resume: global samples processed = {self._resume_global}")

    def _load_known_corrupted_files(self) -> Set[str]:
        """Load pre-scanned corrupt file information from JSON files."""
        corrupted_zip_files = set()
        
        if not os.path.exists(self.corruptions_dir):
            print(f"Warning: Corruptions directory {self.corruptions_dir} does not exist")
            return corrupted_zip_files
        
        json_files = glob.glob(os.path.join(self.corruptions_dir, "*.json"))
        if not json_files:
            print(f"Warning: No corruption JSON files found in {self.corruptions_dir}")
            return corrupted_zip_files
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    corruptions = json.load(f)
                
                for item in corruptions:
                    if 'zip_path' in item and item.get('error_type') == 'BadZipFile':
                        corrupted_zip_files.add(item['zip_path'])
                        
                print(f"Processed {json_file}: found {len(corruptions)} corruptions")
            except Exception as e:
                print(f"Error loading corruption file {json_file}: {e}")
        
        return corrupted_zip_files
    
    def _filter_corrupted_zip_files(self):
        """Filter out known corrupted zip files from the dataset."""
        if not self.corrupted_zip_files:
            self.zip_files = self.index_metadata['zip_files']
            self.images_per_zip = self.index_metadata['images_per_zip']
            return
        
        filtered_zip_files = []
        filtered_images_per_zip = []
        filtered_count = 0
        filtered_zips = 0
        
        for i, zip_path in enumerate(self.index_metadata['zip_files']):
            if zip_path in self.corrupted_zip_files:
                filtered_count += self.index_metadata['images_per_zip'][i]
                filtered_zips += 1
                continue
            
            filtered_zip_files.append(zip_path)
            filtered_images_per_zip.append(self.index_metadata['images_per_zip'][i])
        
        self.zip_files = filtered_zip_files
        self.images_per_zip = filtered_images_per_zip
        
        if filtered_zips > 0:
            print(f"Filtered out {filtered_count} images from {filtered_zips} corrupted zip files")
    
    def _load_index_metadata(self):
        """Load only metadata about the index."""
        index_metadata_path = self.index_file.replace('.pkl', '_metadata.pkl')
        
        if not os.path.exists(index_metadata_path):
            print(f"Creating index metadata from {self.index_file}")
            self._create_index_metadata(self.index_file, index_metadata_path)
        
        with open(index_metadata_path, 'rb') as f:
            return pickle.load(f)
    
    @staticmethod
    def _create_index_metadata(index_path, metadata_path):
        """Create lightweight metadata file from full index."""
        with open(index_path, 'rb') as f:
            all_index = pickle.load(f)
        
        metadata = {
            'total_images': 0,
            'zip_files': [],
            'images_per_zip': []
        }
        
        for zip_path, image_names in all_index:
            num_images = len(image_names)
            if num_images > 0:
                metadata['zip_files'].append(zip_path)
                metadata['images_per_zip'].append(num_images)
                metadata['total_images'] += num_images
        
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
    
    def _log_corrupt_file(self, zip_path, image_name, exception):
        """Log a corrupt file to JSON log with proper locking."""
        current_time = time.time()
        if current_time - self.last_error_time >= 60:
            self.error_count = 0
            self.last_error_time = current_time
            
        if self.error_count >= self.max_errors_per_minute:
            return
            
        self.error_count += 1
        
        corrupt_entry = {
            'zip_path': zip_path,
            'image_name': image_name,
            'error_type': type(exception).__name__,
            'error_msg': str(exception),
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            with open(self.corruption_lock_file, 'r+') as lockf:
                try:
                    fcntl.flock(lockf, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    try:
                        try:
                            with open(self.corruption_log_file, 'r') as f:
                                existing_data = json.load(f)
                        except (FileNotFoundError, json.JSONDecodeError):
                            existing_data = []
                        
                        existing_data.append(corrupt_entry)
                        
                        with open(self.corruption_log_file, 'w') as f:
                            json.dump(existing_data, f, indent=2)
                    finally:
                        fcntl.flock(lockf, fcntl.LOCK_UN)
                except BlockingIOError:
                    pass
        except Exception as e:
            print(f"Error logging corrupt file: {e}")
    
    def __iter__(self):
        """Iterate this worker's shard. Shards by ZIP FILE (strided -> disjoint),
        keeps up to zip_interleave zips open at once and draws randomly among them
        to mix the stream. Reads live worker info from get_worker_info(); the
        self.worker_id / self.num_workers attributes are vestigial and ignored."""
        wi = torch.utils.data.get_worker_info()
        worker_id = wi.id if wi else 0
        num_workers = wi.num_workers if wi else 1

        gid = self.rank * num_workers + worker_id
        total = self.world_size * num_workers

        live = [z for z in self.zip_files if z not in self.corrupted_zip_files]
        my_zips = live[gid::total]  # strided -> disjoint by construction; O(1), no RNG, no hash()
        print(f"Worker {gid}/{total} -> {len(my_zips)} zips")

        epoch = self._epoch
        self._epoch += 1
        rng = random.Random((self.seed * 1000003) ^ (gid * 9973) ^ (epoch * 31))
        rng.shuffle(my_zips)

        skip_per_worker = self._resume_global // max(total, 1)
        # Wrap the resume skip within one pass of this worker's shard, so it can never skip the
        # whole shard and yield nothing (which, in the multi-dataset wrapper, raises StopIteration
        # out of the generator -> a PEP-479 RuntimeError that kills the worker on resume).
        # total_images is the full-dataset count; // total approximates this worker's per-pass share.
        per_worker_pass = getattr(self, 'total_images', 0) // max(total, 1)
        if per_worker_pass > 0:
            skip_per_worker %= per_worker_pass
        emitted = 0

        K = self.zip_interleave
        pending = iter(my_zips)
        streams = []  # list of [ZipFile, iterator-over-shuffled-names, zip_path]

        def _open_next():
            """Pull zips from `pending` until one opens with a usable name list."""
            while True:
                zpath = next(pending, None)
                if zpath is None:
                    return None
                if zpath in self.corrupted_zip_files:
                    continue
                try:
                    zf = zipfile.ZipFile(zpath, 'r')
                    names = sorted(n for n in zf.namelist()
                                   if n.endswith('.webp') and not n.startswith('__MACOSX'))
                except Exception as e:
                    self._log_corrupt_file(zpath, "", e)
                    self.corrupted_zip_files.add(zpath)
                    continue
                if not names:
                    zf.close()
                    continue
                rng.shuffle(names)
                return [zf, iter(names), zpath]

        while len(streams) < K:
            s = _open_next()
            if s is None:
                break
            streams.append(s)

        while streams:
            si = rng.randrange(len(streams))
            zf, names_it, zpath = streams[si]
            name = next(names_it, None)
            if name is None:
                zf.close()
                streams.pop(si)
                s = _open_next()
                if s is not None:
                    streams.append(s)
                continue
            emitted += 1
            if emitted <= skip_per_worker:
                # Resume fast-forward: advance the stream position WITHOUT reading or decoding
                # the image. The previous order (decode then discard) re-decoded the entire
                # pre-resume prefix -- ~iter*batch_per_gpu/num_workers images per worker (e.g.
                # ~2.9M at iter 112k) -- which read as a multi-hour, log-silent hang on every
                # checkpoint resume/requeue. Corrupt entries are not detected while skipping, so
                # the resume offset can differ from an uninterrupted run by the corrupt fraction
                # (~0.1%); immaterial for SSL on this stream.
                continue
            try:
                img = Image.open(io.BytesIO(zf.read(name))).convert('RGB')
            except Exception as e:
                now = time.time()
                if now - self.last_error_time >= 60:
                    self.error_count = 0
                    self.last_error_time = now
                if self.error_count < self.max_errors_per_minute:
                    print(f"Skipping corrupted image {name} from {zpath}: {e}")
                    self.error_count += 1
                self._log_corrupt_file(zpath, name, e)
                if isinstance(e, (zipfile.BadZipFile, zipfile.LargeZipFile)):
                    self.corrupted_zip_files.add(zpath)
                    zf.close()
                    streams.pop(si)
                    s = _open_next()
                    if s is not None:
                        streams.append(s)
                continue
            yield self.transforms(img)

    def __len__(self):
        """Approximate per-rank sample count. Not used by the DataLoader (this is an
        IterableDataset); kept sane for callers that query it."""
        return self.index_metadata['total_images'] // max(self.world_size, 1)


class DINOv2PathologyDataset(torch.utils.data.IterableDataset):
    """
    Optimized dataset for DINOv2 training that returns pre-augmented views.
    Wrapper around MemoryEfficientShardedPathologyDataset.
    
    Args:
        base_dir: Root directory containing zip files
        index_file: Path to dataset index
        n_standard_local_crops: Number of standard local crops
        global_views: Number of global views
        local_crop_size: Size of local crops
        worker_id: Worker ID
        num_workers: Number of workers
        rank: Distributed training rank
        world_size: Total processes
        seed: Random seed
        global_size: Size of global crops
        mean: Normalization mean
        std: Normalization std
    """
    def __init__(
        self,
        base_dir: str,
        index_file: str,
        n_standard_local_crops: int,
        global_views: int,
        local_crop_size: int = 96,
        worker_id: int = 0,
        num_workers: int = 1,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 42,
        global_size: int = 224,
        mean: tuple = (0.6816, 0.5640, 0.7232),
        std: tuple = (0.1617, 0.1714, 0.1389),
        use_pathology_recipe: bool = False,
        ect_probability: float = 0.4,
    ):
        self.n_standard_local_crops = n_standard_local_crops
        self.global_views = global_views
        self.local_crop_size = local_crop_size

        actual_global_views = max(2, global_views)

        self.base_dataset = MemoryEfficientShardedPathologyDataset(
            base_dir=base_dir,
            index_file=index_file,
            worker_id=worker_id,
            num_workers=num_workers,
            rank=rank,
            world_size=world_size,
            seed=seed,
            global_size=global_size,
            local_size=local_crop_size,
            n_local_crops=n_standard_local_crops,
            mean=mean,
            std=std,
            use_pathology_recipe=use_pathology_recipe,
            ect_probability=ect_probability,
        )
    
    def __iter__(self):
        """Return all pre-augmented views flexibly."""
        for crops in self.base_dataset:
            output = []
            
            for i in range(self.global_views):
                if i < len(crops):
                    output.append(crops[i])
            
            for i in range(self.n_standard_local_crops):
                idx = 2 + i
                if idx < len(crops) - 1:
                    output.append(crops[idx])
            
            output.append(crops[-1])
            
            yield tuple(output)

    def __len__(self):
        return len(self.base_dataset)

    def set_resume_position(self, global_samples_processed: int):
        """Set resume position for checkpoint recovery."""
        if hasattr(self, 'base_dataset'):
            self.base_dataset.set_resume_position(global_samples_processed)
            print(f"DINOv2PathologyDataset: Set resume position to {global_samples_processed} samples")



# ============================================================================
# Multi-Dataset Wrapper with Proportional Sampling
# ============================================================================

class ProportionalMultiDatasetWrapper(IterableDataset):
    """
    Combines multiple dataset sources with proportional sampling per batch.
    Each batch maintains the same proportion as overall dataset distribution.
    
    Example: If datasets are 30% TCGA, 10% CPTAC, 60% IMPACT, then each
    batch of size 32 will have ~10 TCGA, ~3 CPTAC, ~19 IMPACT samples.
    """
    def __init__(
        self,
        dataset_configs: List[Dict],
        batch_size_per_gpu: int,
        n_standard_local_crops: int,
        global_views: int,
        local_crop_size: int,
        worker_id: int = 0,
        num_workers: int = 1,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 42,
        global_size: int = 224,
        mean: tuple = (0.6816, 0.5640, 0.7232),
        std: tuple = (0.1617, 0.1714, 0.1389),
        use_pathology_recipe: bool = False,
        ect_probability: float = 0.4,
        emit_scout: bool = False,
        scout_pool_mode: bool = False,
    ):
        super().__init__()

        self.batch_size_per_gpu = batch_size_per_gpu
        self.worker_id = worker_id
        self.num_workers = num_workers
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        
        # Initialize individual datasets
        self.datasets = []
        self.dataset_names = []
        self.dataset_sizes = []
        
        print("\n" + "="*80)
        print("Initializing Multi-Dataset with Proportional Sampling")
        print("="*80)
        
        for config in dataset_configs:
            name = config['name']
            base_dir = config['base_dir']
            index_file = config['index_file']
            print(f"\nLoading {name} dataset from {base_dir}...")

            dataset = MemoryEfficientShardedPathologyDataset(
                base_dir=base_dir,
                index_file=os.path.join(base_dir, index_file),
                worker_id=worker_id,
                num_workers=num_workers,
                rank=rank,
                world_size=world_size,
                seed=seed,
                global_size=global_size,
                local_size=local_crop_size,  # Map parameter name
                n_local_crops=n_standard_local_crops,  # Map parameter name
                mean=mean,
                std=std,
                use_pathology_recipe=use_pathology_recipe,
                ect_probability=ect_probability,
                emit_scout=emit_scout,
                scout_pool_mode=scout_pool_mode,
            )

            self.datasets.append(dataset)
            self.dataset_names.append(name)
            self.dataset_sizes.append(dataset.index_metadata['total_images'])
            
            print(f"  {name}: {dataset.index_metadata['total_images']:,} images")
        
        # Calculate proportions
        total_images = sum(self.dataset_sizes)
        self.proportions = [size / total_images for size in self.dataset_sizes]
        
        print("\n" + "-"*80)
        print("Dataset Proportions:")
        for name, size, prop in zip(self.dataset_names, self.dataset_sizes, self.proportions):
            print(f"  {name}: {size:,} images ({prop*100:.2f}%)")
        print(f"Total: {total_images:,} images")
        
        # Calculate samples per dataset per batch
        self.samples_per_dataset = self._calculate_batch_distribution()
        
        print("\n" + "-"*80)
        print(f"Per-batch distribution (batch_size={batch_size_per_gpu}):")
        for name, count in zip(self.dataset_names, self.samples_per_dataset):
            print(f"  {name}: {count} samples per batch ({count/batch_size_per_gpu*100:.1f}%)")
        print("="*80 + "\n")
        
        # Create iterators
        self.iterators = None
    
    def _calculate_batch_distribution(self):
        """
        Calculate how many samples from each dataset per batch.
        Ensures proportions are maintained and sum equals batch_size.
        """
        # Calculate ideal samples (may be fractional)
        ideal_samples = [prop * self.batch_size_per_gpu for prop in self.proportions]
        
        # Round to integers (floor first)
        samples = [int(s) for s in ideal_samples]
        
        # Distribute remaining samples to maintain sum = batch_size
        remainder = self.batch_size_per_gpu - sum(samples)
        
        # Give remaining samples to datasets with largest fractional parts
        fractional_parts = [(ideal - actual, idx) 
                           for idx, (ideal, actual) in enumerate(zip(ideal_samples, samples))]
        fractional_parts.sort(reverse=True)
        
        for i in range(remainder):
            idx = fractional_parts[i][1]
            samples[idx] += 1
        
        assert sum(samples) == self.batch_size_per_gpu, \
            f"Batch distribution error: {sum(samples)} != {self.batch_size_per_gpu}"
        
        return samples
    
    def set_worker_info(self, worker_id, num_workers):
        """Propagate worker info to all datasets"""
        self.worker_id = worker_id
        self.num_workers = num_workers
        for dataset in self.datasets:
            dataset.set_worker_info(worker_id, num_workers)
    
    def set_resume_position(self, global_samples_processed):
        """Propagate resume position to each sub-dataset as ITS proportional share of the
        global count. A small dataset contributed only its fraction of the samples, so passing
        the full global count to every dataset would tell the small ones to skip more images
        than they have -- emptying them on resume (StopIteration out of the generator below)."""
        bs = max(self.batch_size_per_gpu, 1)
        for i, dataset in enumerate(self.datasets):
            share = int(global_samples_processed * self.samples_per_dataset[i] / bs)
            dataset.set_resume_position(share)
    
    def __iter__(self):
        """
        Yield samples in proportion-maintaining pattern.
        Pattern repeats every batch_size samples.
        """
        # Worker info from get_worker_info() so nothing depends on worker_init_fn
        # having propagated self.worker_id / self.num_workers.
        wi = torch.utils.data.get_worker_info()
        worker_id = wi.id if wi else 0
        num_workers = wi.num_workers if wi else 1

        # Create fresh iterators
        self.iterators = [iter(ds) for ds in self.datasets]
        
        # Create sampling pattern for one batch
        # Example: [0, 0, 0, ...(10x), 1, 1, 1 (3x), 2, 2, ...(19x)]
        pattern = []
        for dataset_idx, count in enumerate(self.samples_per_dataset):
            pattern.extend([dataset_idx] * count)
        
        # Shuffle pattern to avoid systematic bias within batch
        rng = random.Random(self.seed + self.rank * num_workers + worker_id)
        
        # Yield samples according to pattern
        while True:
            # Shuffle pattern for this batch
            batch_pattern = pattern.copy()
            rng.shuffle(batch_pattern)
            
            for dataset_idx in batch_pattern:
                try:
                    sample = next(self.iterators[dataset_idx])
                    yield sample
                except StopIteration:
                    # One dataset exhausted - recreate its iterator and try once more.
                    print(f"Dataset {self.dataset_names[dataset_idx]} exhausted, restarting...")
                    self.iterators[dataset_idx] = iter(self.datasets[dataset_idx])
                    try:
                        sample = next(self.iterators[dataset_idx])
                    except StopIteration:
                        # Still empty right after a restart (e.g. an over-skipped resume). Skip
                        # this slot instead of letting StopIteration escape the generator, which
                        # Python 3.7+ (PEP 479) converts to RuntimeError and kills the worker.
                        continue
                    yield sample
    
    def __len__(self):
        """Return combined length"""
        return sum(len(ds) for ds in self.datasets)