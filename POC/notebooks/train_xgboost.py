#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
XGBoost Exercise Classifier Training & Evaluation Script
Created for Samarth ML pipeline.

This script implements:
1. Feature engineering on kinematic/scalar parameters extracted from joint coordinates.
2. Subject-aware 5-fold Stratified Group Cross-Validation.
3. XGBClassifier training with class-imbalance weighting.
4. Exporting the trained model (both pickle for pipeline compatibility and json for backend import).

Usage:
    python train_xgboost.py --csv-path data/final_cleaned_data.csv --save-dir models
"""

import os
import argparse
import pickle
import numpy as np
import pandas as pd
from collections import Counter

from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report

try:
    from xgboost import XGBClassifier
except ImportError:
    import subprocess
    import sys
    print("XGBoost is not installed. Attempting to install...")
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'xgboost', '-q'])
    from xgboost import XGBClassifier


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers hand-crafted kinematic and statistical features from raw angular measurements.
    Matches the logic used in the FULL_INTEGRATED_2 notebook.
    """
    X = df.copy()

    # ── A. Bilateral asymmetry ─────────────────────────────────────────────
    for joint in ['hip', 'knee', 'ankle']:
        l_rom_col = f'left_{joint}_ROM'
        r_rom_col = f'right_{joint}_ROM'
        
        # Ensure columns exist before calculating asymmetry
        if l_rom_col in X.columns and r_rom_col in X.columns:
            l_rom, r_rom = X[l_rom_col], X[r_rom_col]
            denom = ((l_rom + r_rom) / 2).replace(0, np.nan)
            X[f'fe_{joint}_ROM_asym'] = (l_rom - r_rom).abs() / denom

        l_vel_col = f'left_{joint}_mean_velocity'
        r_vel_col = f'right_{joint}_mean_velocity'
        if l_vel_col in X.columns and r_vel_col in X.columns:
            l_vel, r_vel = X[l_vel_col], X[r_vel_col]
            denom_v = ((l_vel + r_vel) / 2).replace(0, np.nan)
            X[f'fe_{joint}_vel_asym'] = (l_vel - r_vel).abs() / denom_v

        l_peak_col = f'left_{joint}_peak_angle'
        r_peak_col = f'right_{joint}_peak_angle'
        if l_peak_col in X.columns and r_peak_col in X.columns:
            X[f'fe_{joint}_peak_asym'] = (X[l_peak_col] - X[r_peak_col]).abs()

        l_smooth_col = f'left_{joint}_smoothness'
        r_smooth_col = f'right_{joint}_smoothness'
        if l_smooth_col in X.columns and r_smooth_col in X.columns:
            X[f'fe_{joint}_smooth_diff'] = X[l_smooth_col] - X[r_smooth_col]

    # ── B. Hip-knee kinematic chain ────────────────────────────────────────
    if 'left_hip_ROM' in X.columns and 'left_knee_ROM' in X.columns:
        X['fe_left_hip_knee_ROM_ratio'] = X['left_hip_ROM'] / (X['left_knee_ROM'] + 1e-6)
    if 'right_hip_ROM' in X.columns and 'right_knee_ROM' in X.columns:
        X['fe_right_hip_knee_ROM_ratio'] = X['right_hip_ROM'] / (X['right_knee_ROM'] + 1e-6)
    if 'left_hip_knee_coupling' in X.columns and 'right_hip_knee_coupling' in X.columns:
        X['fe_hk_coupling_asym'] = X['left_hip_knee_coupling'] - X['right_hip_knee_coupling']

    # ── C. Knee flexion deficit ────────────────────────────────────────────
    if 'left_knee_start_angle' in X.columns and 'left_knee_peak_angle' in X.columns:
        X['fe_left_knee_flex_deficit'] = X['left_knee_start_angle'] - X['left_knee_peak_angle']
    if 'right_knee_start_angle' in X.columns and 'right_knee_peak_angle' in X.columns:
        X['fe_right_knee_flex_deficit'] = X['right_knee_start_angle'] - X['right_knee_peak_angle']
    if 'fe_left_knee_flex_deficit' in X.columns and 'fe_right_knee_flex_deficit' in X.columns:
        X['fe_knee_flex_deficit_asym'] = X['fe_left_knee_flex_deficit'] - X['fe_right_knee_flex_deficit']

    # ── D. Ankle dorsiflexion ──────────────────────────────────────────────
    if 'left_ankle_dorsiflexion' in X.columns and 'right_ankle_dorsiflexion' in X.columns:
        X['fe_ankle_dors_asym'] = (X['left_ankle_dorsiflexion'] - X['right_ankle_dorsiflexion']).abs()
    if 'left_ankle_ROM' in X.columns and 'right_ankle_ROM' in X.columns:
        X['fe_ankle_ROM_combined'] = X['left_ankle_ROM'] + X['right_ankle_ROM']

    # ── E. Trunk lean ──────────────────────────────────────────────────────
    if 'left_trunk_lean_proxy' in X.columns and 'right_trunk_lean_proxy' in X.columns:
        X['fe_trunk_lean_asym'] = (X['left_trunk_lean_proxy'] - X['right_trunk_lean_proxy']).abs()
        X['fe_trunk_lean_total'] = X['left_trunk_lean_proxy'] + X['right_trunk_lean_proxy']

    # ── F. Movement quality composite ─────────────────────────────────────
    for side in ['left', 'right']:
        for joint in ['hip', 'knee', 'ankle']:
            smooth_col = f'{side}_{joint}_smoothness'
            ec_col = f'{side}_{joint}_EC_ratio'
            if smooth_col in X.columns and ec_col in X.columns:
                X[f'fe_{side}_{joint}_quality'] = X[smooth_col] / (X[ec_col] + 1e-6)

    # ── G. Eccentric control ratio ─────────────────────────────────────────
    for side in ['left', 'right']:
        for joint in ['hip', 'knee']:
            vel_desc = f'{side}_{joint}_mean_vel_desc'
            vel_asc = f'{side}_{joint}_mean_vel_asc'
            if vel_desc in X.columns and vel_asc in X.columns:
                X[f'fe_{side}_{joint}_ecc_ratio'] = X[vel_desc] / (X[vel_desc] + X[vel_asc] + 1e-6)

    # ── H. Hesitation index ────────────────────────────────────────────────
    l_hes_cols = ['left_hip_hesitations', 'left_knee_hesitations', 'left_ankle_hesitations']
    r_hes_cols = ['right_hip_hesitations', 'right_knee_hesitations', 'right_ankle_hesitations']
    
    if all(c in X.columns for c in l_hes_cols):
        X['fe_left_total_hesitations'] = X[l_hes_cols].sum(axis=1)
    if all(c in X.columns for c in r_hes_cols):
        X['fe_right_total_hesitations'] = X[r_hes_cols].sum(axis=1)
    if 'fe_left_total_hesitations' in X.columns and 'fe_right_total_hesitations' in X.columns:
        X['fe_hesitation_asym'] = X['fe_left_total_hesitations'] - X['fe_right_total_hesitations']

    # ── I. Temporal coordination lag ──────────────────────────────────────
    lag_cols = ['hip_temporal_lag_fr', 'knee_temporal_lag_fr', 'ankle_temporal_lag_fr']
    if all(c in X.columns for c in lag_cols):
        X['fe_total_temporal_lag'] = X[lag_cols].abs().sum(axis=1)

    return X


class ExerciseClassifier:
    """
    Handles training, cross-validation, prediction, and export of the XGBoost classifier.
    """
    IDENTIFIERS = ['rep_id', 'subject_id']
    TARGET = 'exercise'
    IMU_COLS_PREFIX = 'imu_'
    SLR_COLS = ['left_knee_stability_slr', 'left_knee_min_during_slr',
                'right_knee_stability_slr', 'right_knee_min_during_slr']

    def __init__(self):
        self.model = None
        self.le_target = LabelEncoder()
        self.feature_cols = None
        self.trained = False

    def _get_drop_cols(self, df):
        imu_cols = [c for c in df.columns if c.startswith(self.IMU_COLS_PREFIX)]
        all_drops = list(set(self.IDENTIFIERS + [self.TARGET] + imu_cols + self.SLR_COLS))
        return [c for c in all_drops if c in df.columns]

    def train_and_eval(self, csv_path: str, cv_splits: int = 5):
        print(f"Loading dataset from: {csv_path}")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Dataset path not found: {csv_path}")

        df = pd.read_csv(csv_path, low_memory=False)
        
        # Verify required columns exist
        if self.TARGET not in df.columns:
            raise ValueError(f"Target column '{self.TARGET}' not found in dataset columns: {list(df.columns)}")
        if 'subject_id' not in df.columns:
            print("Warning: 'subject_id' not found in dataset. Creating dummy subject_id for CV.")
            df['subject_id'] = np.arange(len(df))

        groups = df['subject_id'].values
        y = self.le_target.fit_transform(df[self.TARGET].values)
        
        drop_cols = self._get_drop_cols(df)
        X_base = df.drop(columns=drop_cols, errors='ignore').copy()

        # Engineer features
        print("Engineering features...")
        X_eng = engineer_features(X_base)

        # Handle null values by filling with column medians
        null_cols = X_eng.columns[X_eng.isnull().any()].tolist()
        for col in null_cols:
            X_eng[col] = X_eng[col].fillna(X_eng[col].median())

        self.feature_cols = X_eng.columns.tolist()
        X = X_eng.copy()

        print(f"Dataset shape: {X.shape} (features: {len(self.feature_cols)})")
        print(f"Target distribution: {Counter(df[self.TARGET]) if 'Counter' in globals() else dict(pd.Series(y).value_counts())}")

        # Compute sample weights to handle class imbalance
        sample_weights = compute_sample_weight(class_weight='balanced', y=y)

        # Subject-aware stratified group cross-validation
        sgkf = StratifiedGroupKFold(n_splits=cv_splits, shuffle=True, random_state=42)
        fold_results = []
        all_y_true, all_y_pred = [], []

        print(f"Running {cv_splits}-fold subject-aware Stratified Group CV...")
        for fold, (tr_idx, te_idx) in enumerate(sgkf.split(X.values, y, groups)):
            m = XGBClassifier(
                n_estimators=300, max_depth=5, learning_rate=0.05,
                subsample=0.80, colsample_bytree=0.70, min_child_weight=5,
                gamma=1.0, reg_alpha=0.10, reg_lambda=1.0,
                objective='multi:softprob', num_class=len(self.le_target.classes_),
                eval_metric='mlogloss', random_state=42, n_jobs=-1,
            )
            m.fit(
                X.values[tr_idx], y[tr_idx],
                sample_weight=sample_weights[tr_idx],
                eval_set=[(X.values[te_idx], y[te_idx])],
                verbose=False
            )
            pred = m.predict(X.values[te_idx])
            acc = accuracy_score(y[te_idx], pred)
            f1 = f1_score(y[te_idx], pred, average='macro', zero_division=0)
            
            fold_results.append({'fold': fold + 1, 'acc': acc, 'f1': f1})
            all_y_true.extend(y[te_idx])
            all_y_pred.extend(pred)
            print(f"  Fold {fold + 1}: Accuracy = {acc:.3f} | Macro-F1 = {f1:.3f}")

        mean_acc = np.mean([r['acc'] for r in fold_results])
        mean_f1 = np.mean([r['f1'] for r in fold_results])
        print(f"\nOverall CV Results: Mean Accuracy = {mean_acc:.3f} | Mean Macro-F1 = {mean_f1:.3f}")
        print("\nClassification Report:")
        print(classification_report(all_y_true, all_y_pred, target_names=self.le_target.classes_, zero_division=0))

        # Train the final model on all data
        print("Training final model on all data...")
        self.model = XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.80, colsample_bytree=0.70, min_child_weight=5,
            gamma=1.0, reg_alpha=0.10, reg_lambda=1.0,
            objective='multi:softprob', num_class=len(self.le_target.classes_),
            eval_metric='mlogloss', random_state=42, n_jobs=-1,
        )
        self.model.fit(X.values, y, sample_weight=sample_weights, verbose=False)
        self.trained = True
        print("Final model training complete.")

        return mean_acc, mean_f1

    def save(self, pkl_save_path: str, json_save_path: str):
        """
        Saves the model in two formats:
        1. Pickle (.pkl): Containing model instance, label encoder, and feature column list.
        2. JSON (.json): Native XGBoost save format (easy for loading in production backends).
        """
        if not self.trained:
            raise ValueError("Model must be trained before saving.")

        # Ensure directories exist
        os.makedirs(os.path.dirname(os.path.abspath(pkl_save_path)), exist_ok=True)
        os.makedirs(os.path.dirname(os.path.abspath(json_save_path)), exist_ok=True)

        # 1. Save pickle file
        with open(pkl_save_path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'le': self.le_target,
                'feature_cols': self.feature_cols
            }, f)
        print(f"Saved Pickle bundle to: {pkl_save_path}")

        # 2. Save native XGBoost JSON
        self.model.save_model(json_save_path)
        print(f"Saved Native XGBoost JSON model to: {json_save_path}")

        # Write labels list to a small JSON next to the model for easy loading in production backend
        labels_path = json_save_path.replace('.json', '_labels.json')
        import json
        with open(labels_path, 'w') as f:
            json.dump({
                'classes': list(self.le_target.classes_),
                'feature_cols': self.feature_cols
            }, f, indent=2)
        print(f"Saved label mapping and feature column list to: {labels_path}")


def main():
    parser = argparse.ArgumentParser(description="Train XGBoost Exercise Classifier")
    parser.add_argument('--csv-path', type=str, default='data/final_cleaned_data.csv',
                        help='Path to the cleaned dataset CSV (default: data/final_cleaned_data.csv)')
    parser.add_argument('--save-dir', type=str, default='models',
                        help='Directory to save output models (default: models)')
    parser.add_argument('--cv-splits', type=int, default=5,
                        help='Number of cross-validation splits (default: 5)')
    args = parser.parse_args()

    # Paths setup
    pkl_path = os.path.join(args.save_dir, 'xgb_exercise_classifier.pkl')
    json_path = os.path.join(args.save_dir, 'xgb_exercise_classifier.json')

    clf = ExerciseClassifier()
    try:
        clf.train_and_eval(csv_path=args.csv_path, cv_splits=args.cv_splits)
        clf.save(pkl_save_path=pkl_path, json_save_path=json_path)
        print("\nWorkflow completed successfully!")
    except Exception as e:
        print(f"\nError encountered during execution: {e}")
        print("Please check your input data path or missing columns.")


if __name__ == '__main__':
    main()
