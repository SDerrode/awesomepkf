#!/usr/bin/env bash
# FIX : #!/usr/bin/env bash pour portabilité
set -euo pipefail   # FIX : toute erreur interrompt le script avant d'écraser le README

# ---------------------------------------------------------------------------
# Vérification des dépendances
# FIX : tree et git vérifiés — sinon README vidé silencieusement
# ---------------------------------------------------------------------------
for cmd in git tree awk; do
    if ! command -v "$cmd" &>/dev/null; then
        echo "❌ Commande requise introuvable : $cmd"
        exit 1
    fi
done

# ---------------------------------------------------------------------------
# Vérification que README.md existe dans le répertoire courant
# FIX : évite d'écraser / créer un README vide si lancé au mauvais endroit
# ---------------------------------------------------------------------------
if [ ! -f "README.md" ]; then
    echo "❌ README.md introuvable dans le répertoire courant ($(pwd))."
    echo "   Lancez ce script depuis la racine du projet."
    exit 1
fi

# ---------------------------------------------------------------------------
# Fichiers temporaires avec nettoyage garanti
# FIX : trap assure la suppression de README.tmp même en cas d'erreur
# ---------------------------------------------------------------------------
TMP_TREE=$(mktemp)
TMP_README=$(mktemp)
trap 'rm -f "$TMP_TREE" "$TMP_README"' EXIT

# ---------------------------------------------------------------------------
# Génération de l'arborescence
# ---------------------------------------------------------------------------
git ls-files | tree --fromfile -F -a -L 4 --dirsfirst \
    -I "logs|venv|*.csv|*.pkl|*.png|__pycache__|*.code-workspace|*.ipynb|.vscode|.gitkeep|.DS_Store" \
    > "$TMP_TREE"

# Vérification que tree a produit un résultat non vide
if [ ! -s "$TMP_TREE" ]; then
    echo "❌ tree n'a produit aucune sortie — README non modifié."
    exit 1
fi

# ---------------------------------------------------------------------------
# Mise à jour du README
# FIX : TMP_TREE passé via -v (plus d'interpolation shell dans awk → pas d'injection)
# FIX : résultat écrit dans TMP_README d'abord, puis mv atomique vers README.md
# ---------------------------------------------------------------------------
awk -v tmp="$TMP_TREE" '
/<!-- PROJECT_STRUCTURE_START -->/ {
    print
    print "```text"
    while ((getline line < tmp) > 0) print line
    close(tmp)
    print "```"
    skip = 1
    next
}
/<!-- PROJECT_STRUCTURE_END -->/ { skip = 0 }
!skip
' README.md > "$TMP_README"

mv "$TMP_README" README.md

echo "✅ README mis à jour."
