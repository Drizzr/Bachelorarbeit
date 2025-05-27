import os
import random
import shutil
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
LOGGER = logging.getLogger(__name__)

DATA_DIR = "MCG_segmentation/Datasets/base"

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

def split_files(processed_path, train_dir, val_dir, train_ratio=0.8, random_seed=42, include_records="all"):
    """Split CSV files into train and val directories, optionally limiting to specified records."""
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

    # Limit to specified records if provided
    if include_records != "all":
        valid_records = [record for record in record_names if record in include_records]
        LOGGER.info(f"Filtering records: {len(record_names)} total, {len(valid_records)} included")
    else:
        valid_records = record_names

    if not valid_records:
        LOGGER.error(f"No valid records found in {processed_path}")
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

    # Move files to train or val directories
    files_moved = 0
    for record in valid_records:
        # Find all channel files for this record
        record_files = [f for f in all_files if f.startswith(record + '_') or f == record + '.csv']
        target_dir = train_dir if record in train_records else val_dir

        for file in record_files:
            src_path = os.path.join(processed_path, file)
            dst_path = os.path.join(target_dir, file)
            try:
                shutil.copy2(src_path, dst_path)
                files_moved += 1
                LOGGER.debug(f"Copied {file} to {os.path.basename(target_dir)}")
            except Exception as e:
                LOGGER.error(f"Failed to copy {file} to {target_dir}: {e}")

    LOGGER.info(f"Split complete for {processed_path}: {len(train_records)} records ({files_moved} files) - "
                f"Train: {len(train_records)}, Val: {len(val_records)}")


def main():
    # Define dataset directories
    qtdb_processed = os.path.join(DATA_DIR, 'qtdb/processed')
    ludb_processed = os.path.join(DATA_DIR, 'ludb/processed')
    
    # Define train and val directories for each dataset
    train_dir = os.path.join(DATA_DIR, "train")
    val_dir = os.path.join(DATA_DIR, "val")

    # Define QTDB records to include (fixed record names)
    qtdb_include_records = [
        'sel223', 'sel301', 'sel821', 'sel840', 'sel16539', 'sel16786', 'sel16795', 'sel17453',
        'sele0106', 'sele0114', 'sele0116', 'sele0124', 'sele0136', 'sele0166', 'sele0210', 
        'sele0211', 'sele0303', 'sele0409', 'sele0609', 'sele0612', 'sel30', 'sel31', 'sel32', 
        'sel33', 'sel34', 'sel38', 'sel39', 'sel40', 'sel41', 'sel42', 'sel43', 'sel44', 'sel45', 
        'sel46', 'sel47', 'sel48', 'sel49', 'sel51', 'sel52', 'sel1752', 'sel14172'
    ]
    
    # Split QTDB files
    if os.path.exists(qtdb_processed):
        LOGGER.info(f"Processing QTDB directory: {qtdb_processed}")
        split_files(
            processed_path=qtdb_processed,
            train_dir=train_dir,
            val_dir=val_dir,
            train_ratio=0.8,
            random_seed=123,
            include_records=qtdb_include_records
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
            include_records="all"
        )
    else:
        LOGGER.error(f"LUDB processed directory not found: {ludb_processed}")

if __name__ == '__main__':
    main()