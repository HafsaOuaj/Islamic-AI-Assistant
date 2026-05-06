#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tafsir Completion Pipeline

- Loads tafsir dataset
- Fills missing tafsir from backup JSONL files
- Saves enriched dataset

Author: Hafsa Ouajdi
"""

import os
import logging
from typing import List
import pandas as pd
 
# CONFIG
INPUT_DATASET = "hf://datasets/gurgutan/quran-tafseer-qurancom/quran-en-ibn-kathir-qurancom.jsonl.gz"
BACKUP_DIR = "data/bronze/quran"
OUTPUT_DIR = "data/silver"
OUTPUT_FILE = "tafsir_dataset.json"


# LOGGING SETUP
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s"
)


# FUNCTIONS
def load_main_dataset(path: str) -> pd.DataFrame:
    """_summary_

    Args:
        path (str): _description_

    Returns:
        pd.DataFrame: _description_
    """
    logging.info("Loading main dataset...")
    df = (
        pd.read_json(
            path,
            lines=True,
            dtype={
                'surah_n': 'Int64',
                'ayah_n': 'Int64',
                'ayah_text_ar': 'str',
                'ayah_text': 'str',
                'ayah_tafsir': 'str'
            }
        )
        .filter(['surah_n', 'ayah_n', 'ayah_text_ar', 'ayah_text', 'ayah_tafsir'])
    )
    logging.info(f"Dataset loaded: {len(df)} rows")
    return df


def split_dataset(df: pd.DataFrame):
    """_summary_

    Args:
        df (pd.DataFrame): _description_

    Returns:
        _type_: _description_
    """
    df_missing = df[df['ayah_tafsir'] == "MISSING"].copy()
    df_complete = df[df['ayah_tafsir'] != "MISSING"].copy()

    logging.info(f"Missing tafsir rows: {len(df_missing)}")
    logging.info(f"Complete tafsir rows: {len(df_complete)}")

    return df_missing, df_complete


def load_backup_surah(surah_id: int) -> pd.DataFrame:
    """_summary_

    Args:
        surah_id (int): _description_

    Returns:
        pd.DataFrame: _description_
    """
    file_path = os.path.join(BACKUP_DIR, f"surah_{surah_id:03d}.jsonl")

    if not os.path.exists(file_path):
        logging.warning(f"Missing backup file: {file_path}")
        return pd.DataFrame()

    return (
        pd.read_json(file_path, lines=True)
        .filter(['surah', 'ayah', 'text_plain'])
        .rename(columns={
            'surah': 'surah_n',
            'ayah': 'ayah_n',
            'text_plain': 'ayah_tafsir'
        })
        .astype({
            'surah_n': 'Int64',
            'ayah_n': 'Int64',
            'ayah_tafsir': 'str'
        })
    )


def load_all_backups(surah_ids: List[int]) -> pd.DataFrame:
    """_summary_

    Args:
        surah_ids (List[int]): _description_

    Raises:
        ValueError: _description_

    Returns:
        pd.DataFrame: _description_
    """
    logging.info("Loading backup data...")

    backup_list = []

    for surah_id in surah_ids:
        if pd.isna(surah_id):
            continue

        logging.info(f"Loading surah {surah_id}")
        df_backup = load_backup_surah(int(surah_id))

        if not df_backup.empty:
            backup_list.append(df_backup)

    if not backup_list:
        raise ValueError("No backup data loaded.")

    backup_df = pd.concat(backup_list, ignore_index=True)
    logging.info(f"Total backup rows: {len(backup_df)}")

    return backup_df


def fill_missing_tafsir(df_missing: pd.DataFrame, backup_df: pd.DataFrame) -> pd.DataFrame:
    """_summary_

    Args:
        df_missing (pd.DataFrame): _description_
        backup_df (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """
    logging.info("Filling missing tafsir...")

    df_filled = (
        df_missing
        .drop(columns=['ayah_tafsir'])
        .merge(backup_df, on=['surah_n', 'ayah_n'], how='left')
    )

    return df_filled


def combine_datasets(df_complete: pd.DataFrame, df_filled: pd.DataFrame) -> pd.DataFrame:
    """_summary_

    Args:
        df_complete (pd.DataFrame): _description_
        df_filled (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """
    df_final = pd.concat([df_complete, df_filled], ignore_index=True)
    return df_final


def save_dataset(df: pd.DataFrame, output_dir: str, filename: str):
    """_summary_

    Args:
        df (pd.DataFrame): _description_
        output_dir (str): _description_
        filename (str): _description_
    """
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, filename)

    logging.info(f"Saving dataset to {output_path}")

    df.to_json(output_path, orient="records", lines=True)

    logging.info("Save complete.")


def validate(df: pd.DataFrame):
    """_summary_

    Args:
        df (pd.DataFrame): _description_
    """
    missing_count = df['ayah_tafsir'].isna().sum()
    logging.info(f"Remaining missing tafsir: {missing_count}")



# MAIN PIPELINE
def main():
    """_summary_
    """
    df = load_main_dataset(INPUT_DATASET)

    df_missing, df_complete = split_dataset(df)

    surah_ids = df_missing["surah_n"].dropna().unique()

    backup_df = load_all_backups(surah_ids)

    df_missing_filled = fill_missing_tafsir(df_missing, backup_df)

    df_final = combine_datasets(df_complete, df_missing_filled)

    validate(df_final)

    save_dataset(df_final, OUTPUT_DIR, OUTPUT_FILE)


if __name__ == "__main__":
    main()