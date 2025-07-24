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

def split_files_for_training(processed_path, train_dir, val_dir, test_dir=None, val_ratio=0.15, random_seed=42, exclude_records=None, is_qtdb=False):
    """Splits CSV files into train, val, and test directories based on specified logic."""
    random.seed(random_seed)

    all_files = [f for f in os.listdir(processed_path) if f.endswith('.csv') and not f.startswith('.')]
    
    if not all_files:
        LOGGER.warning(f"No CSV files found in {processed_path}")
        return
    
    record_names = sorted(list(set([f.split('_')[0] for f in all_files if '_' in f])))
    
    if not record_names:
        LOGGER.warning(f"No valid record names found in {processed_path}")
        return

    if exclude_records is None:
        exclude_records = []
    
    # Separate test records from the main pool
    test_records = [record for record in record_names if record in exclude_records]
    train_val_records = [record for record in record_names if record not in exclude_records]
    
    LOGGER.info(f"Total records: {len(record_names)}, Test records: {len(test_records)}, Train/Val records: {len(train_val_records)}")

    train_records, val_records = [], []

    if is_qtdb:
        # All non-excluded QTDB records go to training
        train_records = train_val_records
    else:
        # For LUDB, split the train/val records
        random.shuffle(train_val_records)
        val_count = int(len(train_val_records) * val_ratio)
        val_records = train_val_records[:val_count]
        train_records = train_val_records[val_count:]

    # Create directories
    create_directory(train_dir)
    create_directory(val_dir)
    if test_dir:
        create_directory(test_dir)

    # Function to move files for a list of records to a target directory
    def move_files(records, target_dir):
        files_moved = 0
        for record in records:
            record_files = [f for f in all_files if f.startswith(record + '_') or f == record + '.csv']
            for file in record_files:
                src_path = os.path.join(processed_path, file)
                dst_path = os.path.join(target_dir, file)
                try:
                    shutil.copy2(src_path, dst_path)
                    files_moved += 1
                    LOGGER.debug(f"Copied {file} to {os.path.basename(target_dir)}")
                except Exception as e:
                    LOGGER.error(f"Failed to copy {file} to {target_dir}: {e}")
        return files_moved

    # Move files to respective directories
    train_files_moved = move_files(train_records, train_dir)
    val_files_moved = move_files(val_records, val_dir)
    test_files_moved = 0
    if test_dir:
        test_files_moved = move_files(test_records, test_dir)

    LOGGER.info(f"Split complete for {os.path.basename(processed_path)}: "
                f"{len(train_records)} train records ({train_files_moved} files), "
                f"{len(val_records)} val records ({val_files_moved} files), "
                f"{len(test_records)} test records ({test_files_moved} files)")


def main():
    # Define dataset directories
    qtdb_processed = os.path.join(DATA_DIR, 'base/qtdb/processed')
    ludb_processed = os.path.join(DATA_DIR, 'base/ludb/processed')
    
    # Define output directories
    train_dir = os.path.join(DATA_DIR, "train")
    val_dir = os.path.join(DATA_DIR, "val")
    test_dir = os.path.join(DATA_DIR, "test")

    # Manually excluded QTDB records for the test set
    qtdb_exclude_records = [
        "sel102", "sel104", "sel221", "sel232", "sel310", "sel35", "sel36", "sel37",
    ]
    
    # Process QTDB dataset
    if os.path.exists(qtdb_processed):
        LOGGER.info(f"Processing QTDB directory: {qtdb_processed}")
        split_files_for_training(
            processed_path=qtdb_processed,
            train_dir=train_dir,
            val_dir=val_dir,
            test_dir=test_dir,
            exclude_records=qtdb_exclude_records,
            is_qtdb=True # Flag to handle QTDB logic
        )
    else:
        LOGGER.error(f"QTDB processed directory not found: {qtdb_processed}")
    
    # Process LUDB dataset
    if os.path.exists(ludb_processed):
        LOGGER.info(f"Processing LUDB directory: {ludb_processed}")
        split_files_for_training(
            processed_path=ludb_processed,
            train_dir=train_dir,
            val_dir=val_dir,
            test_dir=test_dir, # LUDB has no test files, but directory might be needed
            val_ratio=0.15,
            random_seed=123,
            is_qtdb=False # Flag to handle LUDB logic
        )
    else:
        LOGGER.error(f"LUDB processed directory not found: {ludb_processed}")

if __name__ == '__main__':
    main()