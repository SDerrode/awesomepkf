#!/bin/bash

# ==========================
# Configuration
# ==========================
LOG_DIR="logs"
LOG_FILE="$LOG_DIR/test_$(date '+%Y-%m-%d_%H-%M-%S').log"

mkdir -p "$LOG_DIR"

# Redirection stdout + stderr vers fichier + console
exec > >(tee -a "$LOG_FILE") 2>&1

# ==========================
# Mode d'affichage
# ==========================
QUIET=1   # 0 = verbeux, 1 = silencieux
# pour lancer en mode silencieux : QUIET=1 ./run_tests.sh

# ==========================
# Couleurs
# ==========================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# ==========================
# Utils
# ==========================
timestamp() {
    date "+%Y-%m-%d %H:%M:%S"
}

log() {
    # n'affiche que si QUIET=0
    if [ "$QUIET" -eq 0 ]; then
        echo -e "$(timestamp) $1"
    fi
}

logfinal() {
    echo -e "$(timestamp) $1"
}

# ==========================
# Compteurs
# ==========================
OK=0
FAIL=0
REPORT=()

# ==========================
# Runner
# ==========================
run_test() {
    local NAME="$1"
    local CMD="$2"

    log "${BLUE}▶ Lancement : $NAME${NC}"
    log "  Commande : $CMD"

    local START END DURATION STATUS

    START=$(date +%s)
    eval "$CMD"
    STATUS=$?
    END=$(date +%s)

    DURATION=$((END - START))

    if [ "$STATUS" -eq 0 ]; then
        log "${GREEN}✔ $NAME : OK${NC} (${DURATION}s)"
        REPORT+=("✔ $NAME (${DURATION}s)")
        OK=$((OK + 1))
    else
        log "${RED}✖ $NAME : FAIL${NC} (code=$STATUS, ${DURATION}s)"
        REPORT+=("✖ $NAME (code=$STATUS, ${DURATION}s)")
        FAIL=$((FAIL + 1))
    fi
}

# ==========================
# En-tête
# ==========================
log "${YELLOW}=================================${NC}"
log "${YELLOW}  Lancement des tests de la lib${NC}"
log "${YELLOW}=================================${NC}"

# ==========================
# Tests
# ==========================
run_test "Programme bash_augmentation_NL.sh" "./prg/tests/bash_augmentation_NL.sh"
run_test "Programme bash_augmentation_L.sh" "./prg/tests/bash_augmentation_L.sh"
# run_test "Programme 3" "python mylib/programme3.py input.txt"

# ==========================
# Rapport final
# ==========================
log ""
log "${YELLOW}=================================${NC}"
log "${YELLOW}           BILAN FINAL${NC}"
log "${YELLOW}=================================${NC}"

for line in "${REPORT[@]}"; do
    log "$line"
done

log ""
log "${GREEN}✔ Succès : $OK${NC}"
log "${RED}✖ Échecs : $FAIL${NC}"

if [ "$FAIL" -eq 0 ]; then
    logfinal "${GREEN}🎉 Tous les programmes se sont déroulés normalement${NC}"
    exit 0
else
    logfinal "${RED}⚠️ Certains programmes ont échoué${NC}"
    exit 1
fi
