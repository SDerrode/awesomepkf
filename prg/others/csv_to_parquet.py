import pandas as pd
import chardet
import logging
import sys

# Configuration du logger
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def detect_encoding(file_path, n_bytes=50000):
    """Détecte l'encodage d'un fichier texte."""
    try:
        with open(file_path, 'rb') as f:
            raw_data = f.read(n_bytes)
        result = chardet.detect(raw_data)
        encoding = result['encoding'] if result['encoding'] else 'utf-8'
        logger.info(f"Encodage détecté : {encoding} (confiance={result.get('confidence',0):.2f})")
        return encoding
    except Exception as e:
        logger.warning(f"Impossible de détecter l'encodage, utilisation de 'utf-8'. Erreur: {e}")
        return 'utf-8'

def csv_to_parquet(csv_file_path, parquet_file_path, engine='pyarrow'):
    """Convertit un CSV en Parquet de manière robuste."""
    try:
        # Détecter l'encodage
        encoding = detect_encoding(csv_file_path)

        # Lire le CSV
        logger.info(f"Lecture du CSV : {csv_file_path}")
        df = pd.read_csv(csv_file_path, encoding=encoding)
        logger.info(f"CSV chargé avec {len(df)} lignes et {len(df.columns)} colonnes.")

        # Sauvegarder en Parquet
        logger.info(f"Écriture du Parquet : {parquet_file_path}")
        df.to_parquet(parquet_file_path, engine=engine, index=False)
        logger.info("Conversion terminée avec succès !")

    except FileNotFoundError:
        logger.error(f"Fichier CSV introuvable : {csv_file_path}")
    except Exception as e:
        logger.error(f"Erreur lors de la conversion : {e}")

# --- Exemple d'utilisation ---
if __name__ == "__main__":
    if len(sys.argv) < 3:
        logger.error("Usage : python3 csv_to_parquet.py <fichier.csv> <fichier.parquet>")
    else:
        csv_file = sys.argv[1]
        parquet_file = sys.argv[2]
        csv_to_parquet(csv_file, parquet_file)
