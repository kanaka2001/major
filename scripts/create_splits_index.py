#!/usr/bin/env python3
"""
VTU Major Project: Intelligent Medicinal Plant Recognition
Script: scripts/create_splits_index.py

Phase 6, Step 1: Few-Shot Safe Stratified Splitting
- Stratifies by (species, organ_part) with SEED = 42
- Enforces strict few-shot partitioning rules
- Generates zero-copy manifest: dataset/splits_manifest.csv
- Generates master metadata: dataset/class_mapping.json
- Computes Effective Number of Samples weights and Temperature-Smoothed probabilities

Author: Lead Deep Learning Architect
"""

import os
import sys
import csv
import json
import random
from collections import defaultdict, Counter

WORKSPACE_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
DATASET_DIR = os.path.join(WORKSPACE_ROOT, "dataset")
MANIFEST_CSV_PATH = os.path.join(DATASET_DIR, "splits_manifest.csv")
CLASS_MAPPING_JSON_PATH = os.path.join(DATASET_DIR, "class_mapping.json")

SEED = 42
PART_ORDER = ["leaf", "whole_plant", "general", "seed", "bark"]
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

def main():
    print("=" * 80)
    print("VTU Major Project: Few-Shot Safe Stratified Dataset Splitting (Phase 6)")
    print("=" * 80)

    # 1. Discover all species and plant parts
    all_species = sorted([
        d for d in os.listdir(DATASET_DIR)
        if os.path.isdir(os.path.join(DATASET_DIR, d)) and d != "_quarantine"
    ])
    species_to_idx = {name: idx for idx, name in enumerate(all_species)}
    idx_to_species = {str(idx): name for idx, name in enumerate(all_species)}

    part_to_idx = {name: idx for idx, name in enumerate(PART_ORDER)}
    idx_to_part = {str(idx): name for idx, name in enumerate(PART_ORDER)}

    print(f"Total Standardized Species: {len(all_species):,}")
    print(f"Target Anatomical Parts:     {PART_ORDER}")

    # 2. Collect all images grouped by (species, part)
    groups = defaultdict(list)
    total_images_scanned = 0

    for species in all_species:
        sp_dir = os.path.join(DATASET_DIR, species)
        for part in os.listdir(sp_dir):
            if part in part_to_idx:
                part_dir = os.path.join(sp_dir, part)
                if os.path.isdir(part_dir):
                    files = sorted([
                        f for f in os.listdir(part_dir)
                        if os.path.splitext(f)[1].lower() in VALID_EXTENSIONS
                    ])
                    for f in files:
                        rel_path = os.path.relpath(os.path.join(part_dir, f), WORKSPACE_ROOT)
                        groups[(species, part)].append(rel_path)
                        total_images_scanned += 1

    print(f"Total Clean Images Discovered: {total_images_scanned:,}")
    print(f"Total (Species, Part) Strata:  {len(groups):,}")

    # 3. Stratified Splitting with SEED = 42
    rng = random.Random(SEED)

    records = [] # (rel_path, species, species_idx, part, part_idx, split)
    
    split_counts = Counter()
    part_split_counts = defaultdict(Counter)
    species_split_counts = defaultdict(Counter)

    few_shot_classes_info = []

    for (species, part), file_list in sorted(groups.items()):
        n = len(file_list)
        # Deterministic shuffle
        shuffled = list(file_list)
        rng.shuffle(shuffled)

        sp_idx = species_to_idx[species]
        p_idx = part_to_idx[part]

        if n >= 7:
            # 70% Train, 15% Val, 15% Test (min 1 val, min 1 test)
            n_val = max(1, round(0.15 * n))
            n_test = max(1, round(0.15 * n))
            n_train = n - n_val - n_test
            if n_train < 1:
                n_train = 1
                n_val = (n - 1) // 2
                n_test = n - 1 - n_val

            train_files = shuffled[:n_train]
            val_files = shuffled[n_train:n_train + n_val]
            test_files = shuffled[n_train + n_val:]
        elif 3 <= n <= 6:
            val_files = [shuffled[0]]
            test_files = [shuffled[1]]
            train_files = shuffled[2:] # n - 2 images (1 to 4)
            few_shot_classes_info.append({
                "species": species,
                "part": part,
                "total": n,
                "train": len(train_files),
                "val": 1,
                "test": 1,
                "rule": "3 <= n <= 6 (1 val, 1 test, n-2 train)"
            })
        elif n == 2:
            train_files = [shuffled[0]]
            val_files = [shuffled[1]]
            test_files = []
            few_shot_classes_info.append({
                "species": species,
                "part": part,
                "total": n,
                "train": 1,
                "val": 1,
                "test": 0,
                "rule": "n == 2 (1 train, 1 val)"
            })
        else: # n == 1
            train_files = [shuffled[0]]
            val_files = []
            test_files = []
            few_shot_classes_info.append({
                "species": species,
                "part": part,
                "total": n,
                "train": 1,
                "val": 0,
                "test": 0,
                "rule": "n == 1 (1 train, full species coverage)"
            })

        for f in train_files:
            records.append((f, species, sp_idx, part, p_idx, "train"))
            split_counts["train"] += 1
            part_split_counts[part]["train"] += 1
            species_split_counts[species]["train"] += 1

        for f in val_files:
            records.append((f, species, sp_idx, part, p_idx, "val"))
            split_counts["val"] += 1
            part_split_counts[part]["val"] += 1
            species_split_counts[species]["val"] += 1

        for f in test_files:
            records.append((f, species, sp_idx, part, p_idx, "test"))
            split_counts["test"] += 1
            part_split_counts[part]["test"] += 1
            species_split_counts[species]["test"] += 1

    # Verify all 1,203 species are present in train
    species_in_train = set(sp for sp, c in species_split_counts.items() if c["train"] > 0)
    print(f"\nSpecies representation in Train: {len(species_in_train)} / {len(all_species)} (100.0%)")
    assert len(species_in_train) == len(all_species), "Error: Not all species represented in training split!"

    print("\nGlobal Split Distribution:")
    for s in ["train", "val", "test"]:
        pct = split_counts[s] / total_images_scanned * 100
        print(f"  -> {s.upper():5s}: {split_counts[s]:,} images ({pct:.2f}%)")

    # 4. Compute Effective Number of Samples weights & Temperature Smoothed Sampler
    # W_c = (1 - beta) / (1 - beta^n_c)
    BETA = 0.9999
    TEMPERATURE = 0.5 # P(c) \propto n_c^0.5

    class_counts_train = {}
    class_counts_val = {}
    class_counts_test = {}
    class_counts_total = {}
    class_weights_beta = {}
    temp_smoothed_probs = {}

    total_train_images = split_counts["train"]
    unnorm_temp_probs = {}

    for species in all_species:
        n_train = species_split_counts[species]["train"]
        n_val = species_split_counts[species]["val"]
        n_test = species_split_counts[species]["test"]
        n_tot = n_train + n_val + n_test

        class_counts_train[species] = n_train
        class_counts_val[species] = n_val
        class_counts_test[species] = n_test
        class_counts_total[species] = n_tot

        # Effective number weight
        if n_train > 0:
            eff_num = (1.0 - (BETA ** n_train)) / (1.0 - BETA)
            weight = 1.0 / max(eff_num, 1e-8)
        else:
            weight = 1.0
        class_weights_beta[species] = weight

        # Temperature smoothed probability
        unnorm_temp_probs[species] = (n_train ** TEMPERATURE) if n_train > 0 else 0.0

    # Normalize weights so mean is 1.0
    mean_w = sum(class_weights_beta.values()) / len(class_weights_beta)
    for sp in class_weights_beta:
        class_weights_beta[sp] /= mean_w

    # Normalize temperature smoothed probabilities
    sum_temp = sum(unnorm_temp_probs.values())
    for sp in unnorm_temp_probs:
        temp_smoothed_probs[sp] = unnorm_temp_probs[sp] / sum_temp

    # 5. Write splits_manifest.csv
    print(f"\nWriting split manifest to '{os.path.relpath(MANIFEST_CSV_PATH, WORKSPACE_ROOT)}'...")
    with open(MANIFEST_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["rel_path", "species", "species_idx", "part", "part_idx", "split"])
        for r in records:
            writer.writerow(r)
    print(f"  -> Successfully wrote {len(records):,} manifest rows.")

    # 6. Write class_mapping.json
    print(f"Writing class metadata to '{os.path.relpath(CLASS_MAPPING_JSON_PATH, WORKSPACE_ROOT)}'...")
    mapping_payload = {
        "num_species": len(all_species),
        "num_parts": len(PART_ORDER),
        "species_to_idx": species_to_idx,
        "idx_to_species": idx_to_species,
        "part_to_idx": part_to_idx,
        "idx_to_part": idx_to_part,
        "splits_summary": {
            "total_images": total_images_scanned,
            "train_images": split_counts["train"],
            "val_images": split_counts["val"],
            "test_images": split_counts["test"],
            "species_count": len(all_species)
        },
        "part_split_distribution": {
            part: {s: part_split_counts[part][s] for s in ["train", "val", "test"]}
            for part in PART_ORDER
        },
        "class_counts_train": class_counts_train,
        "class_counts_val": class_counts_val,
        "class_counts_test": class_counts_test,
        "class_counts_total": class_counts_total,
        "class_weights_beta_0_9999": class_weights_beta,
        "temperature_smoothed_probs": temp_smoothed_probs,
        "few_shot_classes_count": len(few_shot_classes_info),
        "few_shot_classes_sample": few_shot_classes_info[:20]
    }

    with open(CLASS_MAPPING_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(mapping_payload, f, indent=2)
    print(f"  -> Successfully generated {CLASS_MAPPING_JSON_PATH}")

    # Print summary table of parts across splits
    print("\nPart Breakdown Across Splits:")
    print(f"{'Part':<15} | {'Train':>10} | {'Val':>10} | {'Test':>10} | {'Total':>10}")
    print("-" * 65)
    for part in PART_ORDER:
        tr = part_split_counts[part]["train"]
        va = part_split_counts[part]["val"]
        te = part_split_counts[part]["test"]
        tot = tr + va + te
        print(f"{part:<15} | {tr:>10,} | {va:>10,} | {te:>10,} | {tot:>10,}")
    print("-" * 65)
    print(f"{'TOTAL':<15} | {split_counts['train']:>10,} | {split_counts['val']:>10,} | {split_counts['test']:>10,} | {total_images_scanned:>10,}")

    print("\n[Phase 6, Step 1] Complete!")

if __name__ == "__main__":
    main()

