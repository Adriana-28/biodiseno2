from __future__ import annotations

import argparse
import logging
import pickle
import sys
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, HuberRegressor, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVR

warnings.filterwarnings("ignore")
logging.getLogger("utils").setLevel(logging.ERROR)

# ── Resolver paths para importar utils y palm_utils ─────────────────────────
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for candidate in (_HERE, _ROOT / "Servidor", _ROOT):
    if (candidate / "utils.py").exists():
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
        break
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from utils import (
    PERCENTILE_LEVELS as NAIL_PERCENTILE_LEVELS,
    calculate_features,
    cut_image,
    detect_nail_and_skin_bboxes,
    detect_white_reference,
    get_feature_names,
    normalize_features,
)
from palm_utils import (
    PERCENTILE_LEVELS as PALM_PERCENTILE_LEVELS,
    RATIO_PERCENTILES as PALM_RATIO_PERCENTILES,
    detect_white_reference_from_roi,
    extract_best_palm_frame,
    get_palm_roi_from_frame,
    calculate_palm_features,
    process_video_for_training,
)

# ── Constantes ───────────────────────────────────────────────────────────────
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

NAIL_RATIO_PERCENTILES = [25, 50, 75]

# Dimensiones por bloque
N_NAIL_BASE    = len(get_feature_names())                                    # 21
N_NAIL_RATIOS  = len(NAIL_RATIO_PERCENTILES) * 2                             # 6
N_NAIL_TOTAL   = N_NAIL_BASE + N_NAIL_RATIOS                                 # 27

N_PALM_TOTAL   = len(PALM_PERCENTILE_LEVELS) * 3 + len(PALM_RATIO_PERCENTILES) * 2  # 69

N_FEAT_COMBINED  = N_NAIL_TOTAL + N_PALM_TOTAL + 1   # +Sexo
N_FEAT_UNAS_ONLY = N_NAIL_TOTAL + 1
N_FEAT_PALM_ONLY = N_PALM_TOTAL + 1

WHITE_REF_NAIL_DEFAULT = {"R": 1.0, "G": 1.0, "B": 1.0}
WHITE_REF_PALM_DEFAULT = [220.0, 200.0, 185.0]


# ═══════════════════════════════════════════════════════════════════════════
#  Helpers: leer archivos
# ═══════════════════════════════════════════════════════════════════════════

def _find_file(directory: Path, patient_id: str,
               extensions: set[str]) -> Path | None:
    pid = str(patient_id).strip()
    if not pid.upper().startswith("ID"):
        pid_with = f"ID{pid}"
    else:
        pid_with = pid

    for base in (pid_with, pid):
        for ext in extensions:
            cand = directory / f"{base}{ext}"
            if cand.exists():
                return cand
    # Búsqueda case-insensitive
    for f in directory.iterdir():
        stem_lo = f.stem.lower()
        if f.suffix.lower() in extensions:
            if stem_lo in (pid.lower(), pid_with.lower()):
                return f
    return None


def _load_rgb_image(path: Path) -> np.ndarray | None:
    try:
        import skimage.io as skio
        im = skio.imread(path)
        if im.ndim == 3:
            return im[:, :, :3].astype(np.uint8)
    except Exception:
        pass
    im = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if im is None:
        return None
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def _hb_to_gdl(v) -> float | None:
    if pd.isna(v):
        return None
    try:
        s = str(v).strip().replace(",", ".")
        if not s or s.upper() == "?":
            return None
        f = float(s)
        return f / 10.0 if f > 30 else f
    except ValueError:
        return None


# ═══════════════════════════════════════════════════════════════════════════
#  Carga de metadata CSV
# ═══════════════════════════════════════════════════════════════════════════

def load_metadata(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    # Normalizar nombres → canónicos en minúscula
    rename_map = {
        "Hemoglobina": "hb", "hemoglobina": "hb", "HB": "hb", "Hb": "hb",
        "HEMOGLOBINA": "hb",
        "Sexo": "sexo", "SEXO": "sexo", "Sex": "sexo", "SEX": "sexo",
        "Género": "sexo",
        "ID": "id_paciente", "Id": "id_paciente",
        "id_paciente": "id_paciente",
        "Patient_ID": "id_paciente",
        "UNAS": "unas", "Unas": "unas", "unas": "unas",
    }
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns},
              inplace=True)

    required = {"hb", "sexo", "id_paciente"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV faltan columnas: {missing}. Disponibles: {list(df.columns)}"
        )

    df["id_paciente"] = df["id_paciente"].astype(str).str.strip()
    df["hb"]          = df["hb"].apply(_hb_to_gdl)
    df["sexo_feat"]   = df["sexo"].astype(str).str.strip().str.upper().map(
        {"M": 1.0, "F": 0.0}
    )

    n_before = len(df)
    df = df.dropna(subset=["hb", "sexo_feat"])
    dropped = n_before - len(df)
    if dropped:
        print(f"[WARN] Eliminadas {dropped} filas con Hb/Sexo inválidos.")

    return df.reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════
#  Extracción de features de uña
# ═══════════════════════════════════════════════════════════════════════════

def _nail_features(rgb: np.ndarray) -> np.ndarray | None:
    """Extrae 27 features de uña: 21 percentiles RGB + 6 ratios de color."""
    try:
        boxes = detect_nail_and_skin_bboxes(rgb)
        nt, nl, nb, nr = boxes["nail"]
        nail_crop = rgb[nt:nb, nl:nr]
        if nail_crop.size == 0 or nail_crop.shape[0] < 6 or nail_crop.shape[1] < 6:
            h, w = rgb.shape[:2]
            nail_crop = rgb[:int(h * 0.28), int(w * 0.10):int(w * 0.90)]

        wr   = detect_white_reference(rgb) or WHITE_REF_NAIL_DEFAULT
        raw  = calculate_features(nail_crop, "NAIL")
        norm = normalize_features(raw, wr)
        base = np.array([float(norm[n]) for n in get_feature_names()],
                        dtype=np.float32)

        central = cut_image(nail_crop)
        r = central[:, :, 0].ravel().astype(np.float32)
        g = central[:, :, 1].ravel().astype(np.float32)
        b = central[:, :, 2].ravel().astype(np.float32)

        ratios = []
        for p in NAIL_RATIO_PERCENTILES:
            rp = float(np.percentile(r, p))
            gp = float(np.percentile(g, p))
            bp = float(np.percentile(b, p))
            ratios.append(rp / (gp + 1e-6))
            ratios.append(rp / (bp + 1e-6))

        return np.concatenate([base, ratios], dtype=np.float32)
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════
#  Estimación de white_ref global de palma
# ═══════════════════════════════════════════════════════════════════════════

def estimate_global_palm_white_ref(
    palmas_dir: Path,
    df: pd.DataFrame,
    sample_n: int = 30,
) -> list[float]:
    refs = []
    sample = df.sample(min(sample_n, len(df)), random_state=42)
    print(f"[INFO] Estimando ref. blanca palma ({len(sample)} videos)...")

    for _, row in sample.iterrows():
        vid = _find_file(palmas_dir, str(row["id_paciente"]), VIDEO_EXTENSIONS)
        if vid is None:
            continue
        try:
            result = extract_best_palm_frame(vid)
            if result is None:
                continue
            frame_rgb, bbox, _, _method = result
            roi = get_palm_roi_from_frame(frame_rgb, bbox)
            ref = detect_white_reference_from_roi(roi)
            if ref is not None:
                refs.append(ref)
        except Exception as e:
            print(f"  [WARN] {vid.name}: {e}")

    if refs:
        wr = list(np.median(refs, axis=0).round(1))
        print(f"[INFO] Ref. blanca palma global: {wr}")
        return wr
    print(f"[WARN] Usando ref. blanca por defecto: {WHITE_REF_PALM_DEFAULT}")
    return WHITE_REF_PALM_DEFAULT


# ═══════════════════════════════════════════════════════════════════════════
#  Extracción batch
# ═══════════════════════════════════════════════════════════════════════════

def extract_all_features(
    df: pd.DataFrame,
    unas_dir: Path | None,
    palmas_dir: Path | None,
    white_ref_palm: list[float],
    mode: str = "combined",
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Extrae features de todos los pacientes.

    mode: "combined" | "unas_only" | "palma_only"

    Retorna (X, y, ids_ok).
    Pacientes con alguna parte faltante son omitidos.
    """
    X_rows, y_rows, ids_ok = [], [], []
    total = len(df)
    t0 = time.time()

    print(f"\n[INFO] Extrayendo features ({mode}) — {total} pacientes...")

    for i, (_, row) in enumerate(df.iterrows(), 1):
        pid  = str(row["id_paciente"])
        hb   = float(row["hb"])
        sexo = float(row["sexo_feat"])

        nail_fv = None
        palm_fv = None

        # ── Uña ─────────────────────────────────────────────────
        if mode in ("combined", "unas_only") and unas_dir is not None:
            img_path = _find_file(unas_dir, pid, IMAGE_EXTENSIONS)
            if img_path is None:
                status = "sin imagen uña"
                _print_row(i, total, pid, hb, status, t0)
                continue
            rgb = _load_rgb_image(img_path)
            if rgb is None:
                _print_row(i, total, pid, hb, "imagen corrupta", t0)
                continue
            nail_fv = _nail_features(rgb)
            if nail_fv is None:
                _print_row(i, total, pid, hb, "feature uña falló", t0)
                continue

        # ── Palma ────────────────────────────────────────────────
        if mode in ("combined", "palma_only") and palmas_dir is not None:
            vid_path = _find_file(palmas_dir, pid, VIDEO_EXTENSIONS)
            if vid_path is None:
                status = "sin video palma"
                _print_row(i, total, pid, hb, status, t0)
                continue
            try:
                palm_fv = process_video_for_training(vid_path, white_ref_palm)
            except Exception as e:
                _print_row(i, total, pid, hb, f"error palma: {e}", t0)
                continue
            if palm_fv is None:
                _print_row(i, total, pid, hb, "no detectó palma", t0)
                continue

        # ── Ensamblar feature vector ──────────────────────────────
        parts = []
        if mode == "combined":
            parts = [nail_fv, palm_fv, [sexo]]
        elif mode == "unas_only":
            parts = [nail_fv, [sexo]]
        else:  # palma_only
            parts = [palm_fv, [sexo]]

        fv_full = np.concatenate(parts, dtype=np.float32)
        X_rows.append(fv_full)
        y_rows.append(hb)
        ids_ok.append(pid)
        _print_row(i, total, pid, hb, "✓", t0)

    if not X_rows:
        raise RuntimeError("No se extrajo features de ningún paciente.")

    X = np.array(X_rows, dtype=np.float32)
    y = np.array(y_rows, dtype=np.float32)
    print(f"\n[INFO] Dataset: {X.shape[0]} muestras × {X.shape[1]} features")
    return X, y, ids_ok


def _print_row(i, total, pid, hb, status, t0):
    elapsed = time.time() - t0
    avg     = elapsed / i if i else 0
    eta     = avg * (total - i) / 60
    mark    = "✓" if status == "✓" else "✗"
    msg     = f"  [{i:3d}/{total}] {pid}: {mark}  Hb={hb:.1f}"
    if status != "✓":
        msg += f"  — {status}"
    msg += f"  ETA {eta:.1f} min"
    print(msg)


# ═══════════════════════════════════════════════════════════════════════════
#  Construcción de nombres de features
# ═══════════════════════════════════════════════════════════════════════════

def _build_feature_names(mode: str) -> list[str]:
    names = []
    if mode in ("combined", "unas_only"):
        for n in get_feature_names():
            names.append(f"nail_{n}")
        for p in NAIL_RATIO_PERCENTILES:
            names += [f"nail_RoverG_p{p}", f"nail_RoverB_p{p}"]
    if mode in ("combined", "palma_only"):
        for ch in ("R", "G", "B"):
            for p in PALM_PERCENTILE_LEVELS:
                names.append(f"palm_{ch}_p{p}")
        for p in PALM_RATIO_PERCENTILES:
            names += [f"palm_RoverG_p{p}", f"palm_RoverB_p{p}"]
    names.append("Sexo")
    return names


# ═══════════════════════════════════════════════════════════════════════════
#  Modelos disponibles
# ═══════════════════════════════════════════════════════════════════════════

def _get_model(name: str):
    models = {
        "ridge":   Ridge(alpha=1.0),
        "huber":   HuberRegressor(epsilon=1.35, max_iter=300),
        "elastic": ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=500),
        "svr":     SVR(kernel="rbf", C=10, epsilon=0.3, gamma="scale"),
        "rf":      RandomForestRegressor(n_estimators=300, max_depth=7,
                                          min_samples_leaf=3, random_state=42),
        "gbr":     GradientBoostingRegressor(n_estimators=300, max_depth=4,
                                              learning_rate=0.05, subsample=0.8,
                                              random_state=42),
    }
    if name not in models:
        raise ValueError(f"Modelo desconocido: '{name}'. Opciones: {list(models)}")
    return models[name]


# ═══════════════════════════════════════════════════════════════════════════
#  Entrenamiento con CV
# ═══════════════════════════════════════════════════════════════════════════

def train_and_evaluate(
    X: np.ndarray,
    y: np.ndarray,
    model_name: str,
    n_folds: int = 5,
) -> tuple[Pipeline, dict]:
    reg = _get_model(model_name)
    pipeline = Pipeline([
        ("scaler",    RobustScaler()),
        ("regressor", reg),
    ])

    print(f"\n[INFO] Entrenando: {type(reg).__name__} | {n_folds}-fold CV")
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    y_pred_cv = cross_val_predict(pipeline, X, y, cv=kf)

    mae  = mean_absolute_error(y, y_pred_cv)
    rmse = mean_squared_error(y, y_pred_cv) ** 0.5
    r2   = r2_score(y, y_pred_cv)

    print(f"  CV MAE  = {mae:.3f} g/dL")
    print(f"  CV RMSE = {rmse:.3f} g/dL")
    print(f"  CV R²   = {r2:.3f}")

    pipeline.fit(X, y)
    print(f"[OK] Modelo final entrenado con {len(y)} muestras.")

    return pipeline, {"MAE": mae, "RMSE": rmse, "R2": r2, "n_samples": int(len(y))}


# ═══════════════════════════════════════════════════════════════════════════
#  Guardar bundle
# ═══════════════════════════════════════════════════════════════════════════

def save_bundle(
    pipeline: Pipeline,
    white_ref_nail: dict,
    white_ref_palm: list[float],
    metrics: dict,
    feature_names: list[str],
    output_path: Path,
    model_name: str,
    mode: str,
    n_features: int,
) -> None:
    bundle = {
        "model":              pipeline,
        "white_ref":          white_ref_nail,   # compatible con /predict de uñas
        "white_ref_palm":     white_ref_palm,
        "metrics":            metrics,
        "feature_names":      feature_names,
        "n_features":         n_features,
        "model_type":         model_name,
        "hb_unit":            "g/dL",
        "source":             mode,
        "nail_percentile_levels":  NAIL_PERCENTILE_LEVELS,
        "nail_ratio_percentiles":  NAIL_RATIO_PERCENTILES,
        "palm_percentile_levels":  PALM_PERCENTILE_LEVELS,
        "palm_ratio_percentiles":  PALM_RATIO_PERCENTILES,
        "n_nail_features":    N_NAIL_TOTAL if mode != "palma_only" else 0,
        "n_palm_features":    N_PALM_TOTAL if mode != "unas_only"  else 0,
    }
    with open(output_path, "wb") as f:
        pickle.dump(bundle, f)
    print(f"\n[OK] Bundle guardado: {output_path}  ({n_features} features)")


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Entrena modelo Hb combinado uña+palma."
    )
    parser.add_argument("--data_dir", default=".")
    parser.add_argument("--unas_subdir",    default="unas",
                        help="Subcarpeta con imágenes de uñas.")
    parser.add_argument("--palmas_subdir",  default="Palmas",
                        help="Subcarpeta con videos de palmas.")
    parser.add_argument("--csv",            default="metadata1.csv")
    parser.add_argument("--model",          default="ridge",
                        choices=["ridge", "huber", "elastic", "svr", "rf", "gbr"])
    parser.add_argument("--folds",          default=5, type=int)
    parser.add_argument("--output",         default="hb_model_combined.pkl")
    parser.add_argument("--mode",           default="combined",
                        choices=["combined", "unas_only", "palma_only"],
                        help="Qué fuentes usar para features.")
    parser.add_argument("--skip_white_ref", action="store_true")
    args = parser.parse_args()

    data_dir    = Path(args.data_dir)
    unas_dir    = data_dir / args.unas_subdir
    palmas_dir  = data_dir / args.palmas_subdir
    csv_path    = data_dir / args.csv
    output      = Path(args.output)
    mode        = args.mode

    # Validar paths
    for p, label in [(data_dir, "data_dir"), (csv_path, "csv")]:
        if not p.exists():
            print(f"[ERROR] No encontrado: {p}  ({label})")
            sys.exit(1)
    if mode in ("combined", "unas_only") and not unas_dir.exists():
        print(f"[ERROR] No encontrada carpeta uñas: {unas_dir}")
        sys.exit(1)
    if mode in ("combined", "palma_only") and not palmas_dir.exists():
        print(f"[ERROR] No encontrada carpeta palmas: {palmas_dir}")
        sys.exit(1)

    # Dimensión esperada según modo
    n_features_map = {
        "combined":   N_FEAT_COMBINED,
        "unas_only":  N_FEAT_UNAS_ONLY,
        "palma_only": N_FEAT_PALM_ONLY,
    }
    n_features = n_features_map[mode]

    print(f"[INFO] Modo:     {mode}")
    print(f"[INFO] CSV:      {csv_path}")
    print(f"[INFO] Uñas:     {unas_dir}")
    print(f"[INFO] Palmas:   {palmas_dir}")
    print(f"[INFO] Modelo:   {args.model}")
    print(f"[INFO] Features: {n_features}")
    print(f"[INFO] Salida:   {output}")

    # 1. Cargar metadata
    df = load_metadata(csv_path)
    print(f"[INFO] Pacientes en CSV: {len(df)}")

    # 2. Estimar ref. blanca de palma
    white_ref_palm = WHITE_REF_PALM_DEFAULT
    if mode in ("combined", "palma_only"):
        if args.skip_white_ref:
            print(f"[INFO] Ref. blanca palma por defecto: {white_ref_palm}")
        else:
            white_ref_palm = estimate_global_palm_white_ref(palmas_dir, df)

    # 3. Extraer features
    X, y, ids_ok = extract_all_features(
        df, unas_dir, palmas_dir, white_ref_palm, mode
    )

    # Verificar dimensiones
    if X.shape[1] != n_features:
        print(f"[WARN] Dimensión real {X.shape[1]} ≠ esperada {n_features}. "
              f"Usando {X.shape[1]}.")
        n_features = X.shape[1]

    # 4. Entrenar
    pipeline, metrics = train_and_evaluate(X, y, args.model, args.folds)

    # 5. Guardar bundle
    feature_names = _build_feature_names(mode)
    save_bundle(
        pipeline, WHITE_REF_NAIL_DEFAULT, white_ref_palm,
        metrics, feature_names, output, args.model, mode, n_features
    )

    print("\n=== RESUMEN ===")
    print(f"  Pacientes procesados : {len(ids_ok)} / {len(df)}")
    print(f"  MAE                  : {metrics['MAE']:.3f} g/dL")
    print(f"  RMSE                 : {metrics['RMSE']:.3f} g/dL")
    print(f"  R²                   : {metrics['R2']:.3f}")
    print(f"  Bundle               : {output}")


if __name__ == "__main__":
    main()
