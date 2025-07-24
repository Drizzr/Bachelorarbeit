import os
import random
import shutil
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
LOGGER = logging.getLogger(__name__)

DATA_DIR = "MCG_segmentation/Datasets"

def prepare_output_dirs(directories):
    """Leert und erstellt Verzeichnisse, um einen sauberen Start zu gewährleisten."""
    for directory in directories:
        if os.path.exists(directory):
            LOGGER.info(f"Lösche existierendes Verzeichnis: {directory}")
            shutil.rmtree(directory)
        os.makedirs(directory, exist_ok=True)
        # Erstellt .gitignore im neu erstellten Verzeichnis
        gitignore_path = os.path.join(directory, '.gitignore')
        try:
            with open(gitignore_path, 'w') as f:
                f.write('*\n!.gitignore\n')
            LOGGER.info(f"Created .gitignore in {directory}")
        except IOError as e:
            LOGGER.warning(f"Could not create .gitignore in {directory}: {e}")

def split_files_independently(processed_path, train_dir, val_dir, test_dir, val_ratio=0.15, random_seed=42, exclude_records=None, is_qtdb=False):
    """
    Teilt CSV-Dateien unabhängig voneinander in Trainings-, Validierungs- und Testverzeichnisse auf.
    Jede Datei wird als einzelner Datenpunkt behandelt.
    """
    random.seed(random_seed)

    all_files = [f for f in os.listdir(processed_path) if f.endswith('.csv') and not f.startswith('.')]
    
    if not all_files:
        LOGGER.warning(f"Keine CSV-Dateien in {processed_path} gefunden")
        return

    if exclude_records is None:
        exclude_records = []
    
    # --- KERNÄNDERUNG: Aufteilung auf Dateiebene ---
    test_files = []
    train_val_files = []

    # Teile die Dateien basierend auf der Ausschlussliste in Test- und den Rest auf
    for file in all_files:
        # Prüft, ob der Dateiname mit einem der auszuschließenden Präfixe beginnt
        if any(file.startswith(prefix) for prefix in exclude_records):
            test_files.append(file)
        else:
            train_val_files.append(file)
            
    LOGGER.info(f"Dateien gefunden: {len(all_files)} gesamt, {len(test_files)} für Test markiert, {len(train_val_files)} für Train/Val.")

    train_files = []
    val_files = []

    if is_qtdb:
        # Für QTDB gehen alle verbleibenden Dateien direkt ins Training
        train_files = train_val_files
        LOGGER.info(f"Zuweisung aller {len(train_files)} nicht ausgeschlossenen QTDB-Dateien zum Trainings-Set.")
    else:
        # Für LUDB werden die verbleibenden Dateien aufgeteilt
        random.shuffle(train_val_files)
        val_count = int(len(train_val_files) * val_ratio)
        val_files = train_val_files[:val_count]
        train_files = train_val_files[val_count:]
        LOGGER.info(f"Aufteilung der LUDB-Dateien: {len(train_files)} für Training, {len(val_files)} für Validierung.")

    # Funktion zum Kopieren einer Liste von Dateien in ein Zielverzeichnis
    def copy_file_list(files, target_dir):
        for file in files:
            src_path = os.path.join(processed_path, file)
            dst_path = os.path.join(target_dir, file)
            try:
                shutil.copy2(src_path, dst_path)
            except Exception as e:
                LOGGER.error(f"Fehler beim Kopieren von {file} nach {target_dir}: {e}")
        return len(files)

    # Kopiere die sortierten Dateien
    train_files_moved = copy_file_list(train_files, train_dir)
    val_files_moved = copy_file_list(val_files, val_dir)
    test_files_moved = copy_file_list(test_files, test_dir)

    LOGGER.info(f"Aufteilung für {os.path.basename(processed_path)} abgeschlossen: "
                f"{train_files_moved} Trainings-Dateien, "
                f"{val_files_moved} Validierungs-Dateien, "
                f"{test_files_moved} Test-Dateien.")


def main():
    # Definiere Verzeichnisse
    qtdb_processed = os.path.join(DATA_DIR, 'base/qtdb/processed')
    ludb_processed = os.path.join(DATA_DIR, 'base/ludb/processed')
    
    train_dir = os.path.join(DATA_DIR, "train")
    val_dir = os.path.join(DATA_DIR, "val")
    test_dir = os.path.join(DATA_DIR, "test")

    # Bereite die Ausgabeordner vor (leeren und erstellen)
    prepare_output_dirs([train_dir, val_dir, test_dir])

    # Datensätze, die aus QTDB für den Test-Satz ausgeschlossen werden sollen (Präfixe)
    qtdb_exclude_records = [
        "sel102", "sel104", "sel221", "sel232", "sel310", "sel35", "sel36", "sel37",
    ]
    
    # Verarbeite QTDB-Datensatz
    if os.path.exists(qtdb_processed):
        LOGGER.info(f"--- Verarbeite QTDB-Verzeichnis: {qtdb_processed} ---")
        split_files_independently(
            processed_path=qtdb_processed,
            train_dir=train_dir,
            val_dir=val_dir,
            test_dir=test_dir,
            exclude_records=qtdb_exclude_records,
            is_qtdb=True,
            random_seed=123
        )
    else:
        LOGGER.error(f"QTDB 'processed' Verzeichnis nicht gefunden: {qtdb_processed}")
    
    # Verarbeite LUDB-Datensatz
    if os.path.exists(ludb_processed):
        LOGGER.info(f"--- Verarbeite LUDB-Verzeichnis: {ludb_processed} ---")
        split_files_independently(
            processed_path=ludb_processed,
            train_dir=train_dir,
            val_dir=val_dir,
            test_dir=test_dir,
            val_ratio=0.15,
            random_seed=123,
            exclude_records=None, # Für LUDB werden keine Dateien ausgeschlossen
            is_qtdb=False
        )
    else:
        LOGGER.error(f"LUDB 'processed' Verzeichnis nicht gefunden: {ludb_processed}")

if __name__ == '__main__':
    main()