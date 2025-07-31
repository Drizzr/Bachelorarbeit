import os
import random
import shutil
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
LOGGER = logging.getLogger(__name__)

DATA_DIR = "MCG_segmentation/Datasets"

def create_directory(directory):
    """Creates a directory if it does not exist."""
    os.makedirs(directory, exist_ok=True)

def split_dataset(processed_dir, train_dir, val_dir, test_dir, unused_dir=None,
                  train_ratio=0.7, val_ratio=0.1, test_ratio=0.2,
                  random_seed=42, exclude_records=None):
    """
    Splits CSV files from a single processed directory into train, val, test sets,
    and optionally moves excluded records to an unused directory.
    """
    random.seed(random_seed)

    all_files = [f for f in os.listdir(processed_dir) if f.endswith('.csv') and not f.startswith('.')]
    if not all_files:
        LOGGER.error(f"No CSV files found in {processed_dir}")
        return

    # Extract record names (before first underscore or whole filename if no underscore)
    record_names = sorted(list(set([f.split('_')[0] for f in all_files])))

    if exclude_records is None:
        exclude_records = []

    # Separate excluded records
    unused_records = [r for r in record_names if r in exclude_records]
    usable_records = [r for r in record_names if r not in exclude_records]

    LOGGER.info(f"Total records: {len(record_names)}, usable: {len(usable_records)}, unused: {len(unused_records)}")

    # Shuffle usable records and split
    random.shuffle(usable_records)
    total = len(usable_records)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_records = usable_records[:train_end]
    val_records = usable_records[train_end:val_end]
    test_records = usable_records[val_end:]

    LOGGER.info(f"Train: {len(train_records)}, Val: {len(val_records)}, Test: {len(test_records)}")

    # Create directories
    for d in [train_dir, val_dir, test_dir]:
        create_directory(d)
    if unused_dir:
        create_directory(unused_dir)

    def move_files(records, target_dir):
        count = 0
        for record in records:
            record_files = [f for f in all_files if f.startswith(record + '_') or f == record + '.csv']
            for file in record_files:
                src_path = os.path.join(processed_dir, file)
                dst_path = os.path.join(target_dir, file)
                try:
                    shutil.copy2(src_path, dst_path)
                    count += 1
                except Exception as e:
                    LOGGER.error(f"Failed to copy {file} to {target_dir}: {e}")
        return count

    train_count = move_files(train_records, train_dir)
    val_count = move_files(val_records, val_dir)
    test_count = move_files(test_records, test_dir)
    unused_count = 0
    if unused_dir:
        unused_count = move_files(unused_records, unused_dir)

    LOGGER.info(f"Files moved — Train: {train_count}, Val: {val_count}, Test: {test_count}, Unused: {unused_count}")

def main():
    # Processed directories
    qtdb_processed = os.path.join(DATA_DIR, "base/qtdb/processed")
    ludb_processed = os.path.join(DATA_DIR, "base/ludb/processed")

    # Output directories with dataset subfolders
    train = os.path.join(DATA_DIR, "train/")
    val = os.path.join(DATA_DIR, "val/")
    test = os.path.join(DATA_DIR, "test/")
    unused = os.path.join(DATA_DIR, "unused/")

    # No unused for LUDB in your original logic, so omitted here

    # QTDB excluded records (moved to unused)
    qtdb_exclude_records = [
        "sel102", "sel104", "sel221", "sel232", "sel310", "sel35", "sel36", "sel37",
    ]

    # Split QTDB with unused handling
    if os.path.exists(qtdb_processed):
        LOGGER.info("Splitting QTDB dataset...")
        split_dataset(
            processed_dir=qtdb_processed,
            train_dir=train,
            val_dir=val,
            test_dir=test,
            unused_dir=unused,
            train_ratio=0.94,
            val_ratio=0.06,
            test_ratio=0.0,
            random_seed=42,
            exclude_records=qtdb_exclude_records
        )
    else:
        LOGGER.error(f"QTDB processed directory not found: {qtdb_processed}")

    # Split LUDB without unused (excluded records logic omitted)
    if os.path.exists(ludb_processed):
        LOGGER.info("Splitting LUDB dataset...")
        split_dataset(
            processed_dir=ludb_processed,
            train_dir=train,
            val_dir=val,
            test_dir=test,
            unused_dir=None,
            train_ratio=0.55,
            val_ratio=0.05,
            test_ratio=0.40,
            random_seed=123,
            exclude_records=None
        )
    else:
        LOGGER.error(f"LUDB processed directory not found: {ludb_processed}")

if __name__ == "__main__":
    main()
