#!/usr/bin/env bash

# Récupère le dossier où se trouve le script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Répertoires à nettoyer (relatifs au dossier du script)
DIRS=("datafile" "historyTracker" "plot")

for dir in "${DIRS[@]}"; do
    TARGET_DIR="$SCRIPT_DIR/$dir"

    if [ -d "$TARGET_DIR" ]; then
        echo "Nettoyage du répertoire: $dir"

        find "$TARGET_DIR" -mindepth 1 ! -name ".gitkeep" -exec rm -rf {} +

    else
        echo "Répertoire inexistant: $TARGET_DIR"
    fi
done

echo "Nettoyage terminé."
