import csv
import re
import requests
import pandas as pd
from pathlib import Path

# =========================
# PARAMÈTRES UTILISATEUR
# =========================
START_DATE = "2025-01-01"   # inclus
END_DATE   = "2025-02-01"   # exclu  -> janvier 2025 uniquement
DEPARTEMENTS = ["93", "95"]  # Seine-Saint-Denis, Val-d'Oise

OUT_DIR = Path("boamp_par_departement")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# BOAMP (OpenDataSoft) — API OUVERTE
# =========================
BOAMP_BASE = "https://boamp-datadila.opendatasoft.com"
DATASET = "boamp"
EXPORT_CSV_URL = f"{BOAMP_BASE}/api/explore/v2.1/catalog/datasets/{DATASET}/exports/csv"

# =========================
# COLONNES À CONSERVER (UNIQUEMENT)
# =========================
COLUMNS_TO_KEEP = [
    "id_web",
    "id",
    "objet",
    "famille_libelle",
    "dateparution",
    "datefindiffusion",
    "datelimitereponse",
    "nomacheteur",
    "descripteur_libelle",
    "type_marche",
    "type_marche_facette",
    "siren",  # 👈 ajout
]

# Alias fréquents (ODS peut nommer différemment)
ALIASES = {
    "id_web": ["id_web", "idweb", "idWeb"],
    "nomacheteur": ["nomacheteur", "nom_acheteur", "acheteur", "nomAcheteur"],
    "dateparution": ["dateparution", "date_parution"],
    "datefindiffusion": ["datefindiffusion", "date_fin_diffusion"],
    "datelimitereponse": ["datelimitereponse", "date_limite_reponse"],
    "descripteur_libelle": ["descripteur_libelle", "descripteur", "descripteurs_libelle"],
    "type_marche": ["type_marche", "typemarche"],
    "type_marche_facette": ["type_marche_facette", "type_marche_facet"],
    "famille_libelle": ["famille_libelle", "famille"],
    "objet": ["objet", "object", "intitule"],
    "id": ["id", "identifiant"],

    # 👇 au cas où le dataset expose déjà siren/siret sous un autre nom
    "siren": ["siren", "siren_acheteur", "id_siren", "sirenacheteur", "sirenAcheteur"],
}

# =========================
# SIREN/SIRET HELPERS
# =========================
SIREN_RE = re.compile(r"\b(\d{9})\b")
SIRET_RE = re.compile(r"\b(\d{14})\b")

def normalize_digits(x: str | None) -> str | None:
    if x is None:
        return None
    s = re.sub(r"\D", "", str(x))
    return s or None

def derive_siren_from_df(df: pd.DataFrame) -> pd.Series:
    """
    Crée une Series 'siren' en essayant, dans l'ordre :
    1) colonnes dont le nom contient 'siren'
    2) colonnes dont le nom contient 'siret' -> siren = 9 premiers chiffres
    3) scan regex dans quelques colonnes texte (si présentes)
    """
    # 1) Colonnes candidates "siren"
    siren_cols = [c for c in df.columns if "siren" in c.lower()]
    for c in siren_cols:
        s = df[c].astype(str).map(normalize_digits)
        s = s.where(s.str.len() == 9)
        if s.notna().any():
            return s

    # 2) Colonnes candidates "siret" -> 9 premiers chiffres
    siret_cols = [c for c in df.columns if "siret" in c.lower()]
    for c in siret_cols:
        s14 = df[c].astype(str).map(normalize_digits)
        s14 = s14.where(s14.str.len() == 14)
        if s14.notna().any():
            return s14.str[:9]

    # 3) Fallback regex dans des champs texte “probables”
    text_candidate_cols = []
    for c in df.columns:
        cl = c.lower()
        if any(k in cl for k in ["acheteur", "contact", "adresse", "texte", "description", "nom", "organisme"]):
            text_candidate_cols.append(c)

    # limite pour éviter de scanner trop large
    text_candidate_cols = text_candidate_cols[:25]

    def extract_siren_from_row(row) -> str | None:
        for c in text_candidate_cols:
            val = row.get(c)
            if val is None:
                continue
            txt = str(val)

            m14 = SIRET_RE.search(txt)
            if m14:
                return m14.group(1)[:9]

            m9 = SIREN_RE.search(txt)
            if m9:
                return m9.group(1)

        return None

    if text_candidate_cols:
        return df.apply(extract_siren_from_row, axis=1)

    return pd.Series([None] * len(df), index=df.index, dtype="object")

# =========================
# BOAMP DOWNLOAD + READ
# =========================
def download_boamp_csv(dep: str, out_path: Path):
    """
    Télécharge un CSV BOAMP filtré par département et période.
    """
    where = f"dateparution >= date'{START_DATE}' AND dateparution < date'{END_DATE}'"
    params = {
        "where": where,
        "refine[0]": f"code_departement:{dep}",
    }

    print(f"[BOAMP] Téléchargement département {dep}...")
    r = requests.get(EXPORT_CSV_URL, params=params, timeout=300)
    r.raise_for_status()
    out_path.write_bytes(r.content)
    print(f"✅ CSV téléchargé : {out_path} ({out_path.stat().st_size} bytes)")

def read_csv_robust(path: Path) -> pd.DataFrame:
    """
    Lecture robuste du CSV BOAMP.
    - engine="python" pour éviter les erreurs de tokenization du moteur C
    - sep=None pour auto-détecter le séparateur
    - on_bad_lines="skip" pour éviter crash si quelques lignes sont cassées
    """
    try:
        return pd.read_csv(
            path,
            engine="python",
            sep=None,                 # auto detect
            dtype=str,
            quoting=csv.QUOTE_MINIMAL,
            on_bad_lines="skip",
        )
    except Exception:
        return pd.read_csv(
            path,
            engine="python",
            sep=";",
            dtype=str,
            quoting=csv.QUOTE_MINIMAL,
            on_bad_lines="skip",
        )

def resolve_column(df: pd.DataFrame, wanted: str) -> str | None:
    if wanted in df.columns:
        return wanted

    # alias directs
    for alt in ALIASES.get(wanted, []):
        if alt in df.columns:
            return alt

    # case-insensitive
    lower_map = {c.lower(): c for c in df.columns}
    if wanted.lower() in lower_map:
        return lower_map[wanted.lower()]
    for alt in ALIASES.get(wanted, []):
        if alt.lower() in lower_map:
            return lower_map[alt.lower()]

    return None

def keep_only_columns(csv_path: Path):
    """
    - Lit le CSV complet
    - Ajoute/complète une colonne 'siren'
    - Ne garde que COLUMNS_TO_KEEP
    - Réécrit un CSV propre
    """
    df_full = read_csv_robust(csv_path)

    # Ajout/derivation du SIREN depuis le CSV complet
    df_full["siren"] = derive_siren_from_df(df_full)

    keep_real = []
    missing = []
    rename_map = {}

    for col in COLUMNS_TO_KEEP:
        real = resolve_column(df_full, col)
        if real is None:
            missing.append(col)
        else:
            keep_real.append(real)
            if real != col:
                rename_map[real] = col

    if missing:
        print(f"⚠️ Colonnes absentes dans {csv_path.name} (ignorées) : {missing}")

    df = df_full[keep_real].rename(columns=rename_map)

    # Nettoyage final optionnel : siren doit être 9 chiffres
    df["siren"] = df["siren"].astype(str).map(normalize_digits)
    df.loc[df["siren"].str.len() != 9, "siren"] = None

    df.to_csv(csv_path, index=False, encoding="utf-8")
    nb_siren = df["siren"].notna().sum()
    print(f"🧹 CSV filtré : {csv_path} | colonnes={len(df.columns)} | lignes={len(df)} | siren trouvés={nb_siren}")

def main():
    print("=== EXTRACTION BOAMP (janvier 2025, API ouverte) ===")
    print(f"Période: {START_DATE} -> {END_DATE} (END exclue)")
    print(f"Départements: {DEPARTEMENTS}\n")

    labels = {"93": "seine_saint_denis", "95": "val_d_oise"}

    for dep in DEPARTEMENTS:
        out_path = OUT_DIR / f"boamp_{dep}_{labels[dep]}.csv"
        download_boamp_csv(dep, out_path)
        keep_only_columns(out_path)

    print("\n✅ Terminé.")

if __name__ == "__main__":
    main()