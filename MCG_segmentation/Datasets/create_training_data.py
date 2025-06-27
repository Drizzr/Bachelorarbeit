import os
import random
import shutil
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
LOGGER = logging.getLogger(__name__)

DATA_DIR = "MCG_segmentation/Datasets"

def create_directory(directory):
    """Creates directory and adds a .gitignore file."""
    os.makedirs(directory, exist_ok=True)
    gitignore_path = os.path.join(directory, '.gitignore')
    if not os.path.exists(gitignore_path):
        try:
            with open(gitignore_path, 'w') as f:
                f.write('*\n!.gitignore\n')
            LOGGER.info(f"Created .gitignore in {directory}")
        except IOError as e:
            LOGGER.warning(f"Could not create .gitignore in {directory}: {e}")

def split_files(processed_path, train_dir, val_dir, test_dir=None, train_ratio=0.8, random_seed=42, exclude_records=None):
    """Split CSV files into train and val directories, excluding specified records."""
    random.seed(random_seed)

    # Get all CSV files, excluding hidden files
    all_files = [f for f in os.listdir(processed_path) if f.endswith('.csv') and not f.startswith('.')]
    
    if not all_files:
        LOGGER.warning(f"No CSV files found in {processed_path}")
        return
    
    # Extract unique record names (assuming files are named like record_name_chX.csv)
    record_names = []
    for f in all_files:
        if '_' in f:
            record_name = f.split('_')[0]
            if record_name not in record_names:
                record_names.append(record_name)
    
    record_names = sorted(record_names)
    
    if not record_names:
        LOGGER.warning(f"No valid record names found in {processed_path}")
        return

    # Filter out excluded records
    if exclude_records is None:
        exclude_records = []
    
    valid_records = [record for record in record_names if record not in exclude_records]
    test_records = [record for record in record_names if record in exclude_records]
    
    LOGGER.info(f"Filtering records: {len(record_names)} total, {len(valid_records)} included, {len(test_records)} excluded")

    if not valid_records:
        LOGGER.error(f"No valid records found in {processed_path} after excluding specified records")
        return

    # Shuffle records for random split
    random.shuffle(valid_records)

    # Calculate split index
    train_count = int(len(valid_records) * train_ratio)
    train_records = valid_records[:train_count]
    val_records = valid_records[train_count:]

    # Create train and val directories
    create_directory(train_dir)
    create_directory(val_dir)
    if test_dir:
        create_directory(test_dir)

    # Move files
    files_moved = 0
    for record in record_names:
        record_files = [f for f in all_files if f.startswith(record + '_') or f == record + '.csv']
        if record in valid_records:
            target_dir = train_dir if record in train_records else val_dir
        else:
            if not test_dir:
                LOGGER.warning(f"Test directory not provided but excluded record '{record}' exists.")
                continue
            target_dir = test_dir

        for file in record_files:
            src_path = os.path.join(processed_path, file)
            dst_path = os.path.join(target_dir, file)
            try:
                shutil.copy2(src_path, dst_path)
                files_moved += 1
                LOGGER.debug(f"Copied {file} to {os.path.basename(target_dir)}")
            except Exception as e:
                LOGGER.error(f"Failed to copy {file} to {target_dir}: {e}")

    LOGGER.info(f"Split complete for {processed_path}: {len(train_records)} train, {len(val_records)} val, "
                f"{len(test_records)} test records ({files_moved} files total)")


def main():
    # Define dataset directories
    qtdb_processed = os.path.join(DATA_DIR, 'base/qtdb/processed')
    ludb_processed = os.path.join(DATA_DIR, 'base/ludb/processed')
    
    # Define train and val directories for each dataset
    train_dir = os.path.join(DATA_DIR, "train")
    val_dir = os.path.join(DATA_DIR, "val")
    test_dir = os.path.join(DATA_DIR, "test")  # Optional test directory

    # Define QTDB records to exclude (records that should go to test set)
    qtdb_exclude_records = [
        "sel102", "sel104", "sel221", "sel232", "sel310", "sel35", "sel36", "sel37",
    ]
    
    # Split QTDB files
    if os.path.exists(qtdb_processed):
        LOGGER.info(f"Processing QTDB directory: {qtdb_processed}")
        split_files(
            processed_path=qtdb_processed,
            train_dir=train_dir,
            val_dir=val_dir,
            test_dir=test_dir,
            train_ratio=0.8,
            random_seed=123,
            exclude_records=qtdb_exclude_records
        )
    else:
        LOGGER.error(f"QTDB processed directory not found: {qtdb_processed}")
    
    # Split LUDB files (no exclusions)
    if os.path.exists(ludb_processed):
        LOGGER.info(f"Processing LUDB directory: {ludb_processed}")
        split_files(
            processed_path=ludb_processed,
            train_dir=train_dir,
            val_dir=val_dir,
            train_ratio=0.8,
            random_seed=123,
            exclude_records=None  # No exclusions for LUDB
        )
    else:
        LOGGER.error(f"LUDB processed directory not found: {ludb_processed}")

if __name__ == '__main__':
    main()