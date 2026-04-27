import argparse
import csv
import os
from collections import Counter


def parse_args():
    parser = argparse.ArgumentParser(
        description="Check which subjects are missing the four required CSV files for each fold."
    )
    parser.add_argument(
        "--demographics-csv",
        default="/data05/learn2reg/zixi/1DCNN/csv_HCP_1000/new_500.csv",
        type=str,
        help="Path to the demographics CSV with SUB_ID, DX_GROUP and fold columns.",
    )
    parser.add_argument(
        "--data-path",
        required=True,
        type=str,
        help="Root data directory used by training, e.g. /data05/learn2reg/zixi/1DCNN/csv_HCP_1000",
    )
    parser.add_argument(
        "--subject-col",
        default="SUB_ID",
        type=str,
        help="Subject ID column name in the demographics CSV.",
    )
    parser.add_argument(
        "--fold-col",
        default="fold",
        type=str,
        help="Fold column name in the demographics CSV.",
    )
    return parser.parse_args()


def build_candidate_paths(root_dir, sid):
    primary = {
        "anatomical": os.path.join(root_dir, sid, "AnatomicalTracts", "diffusion_measurements_anatomical_tracts.csv"),
        "commissural": os.path.join(root_dir, sid, "FiberClustering", "SeparatedClusters", "diffusion_measurements_commissural.csv"),
        "left_hemisphere": os.path.join(root_dir, sid, "FiberClustering", "SeparatedClusters", "diffusion_measurements_left_hemisphere.csv"),
        "right_hemisphere": os.path.join(root_dir, sid, "FiberClustering", "SeparatedClusters", "diffusion_measurements_right_hemisphere.csv"),
    }
    fallback = {
        "anatomical": os.path.join(root_dir, "UKF_2T_AtlasSpace", "anatomical_tracts", sid + ".csv"),
        "commissural": os.path.join(root_dir, "UKF_2T_AtlasSpace", "tracts_commissural", sid + ".csv"),
        "left_hemisphere": os.path.join(root_dir, "UKF_2T_AtlasSpace", "tracts_left_hemisphere", sid + ".csv"),
        "right_hemisphere": os.path.join(root_dir, "UKF_2T_AtlasSpace", "tracts_right_hemisphere", sid + ".csv"),
    }
    return primary, fallback


def resolve_missing_kinds(root_dir, sid):
    primary, fallback = build_candidate_paths(root_dir, sid)
    missing_kinds = []
    resolved_paths = {}

    for kind in ["anatomical", "commissural", "left_hemisphere", "right_hemisphere"]:
        if os.path.exists(primary[kind]):
            resolved_paths[kind] = primary[kind]
        elif os.path.exists(fallback[kind]):
            resolved_paths[kind] = fallback[kind]
        else:
            missing_kinds.append(kind)
            resolved_paths[kind] = fallback[kind]

    return missing_kinds, resolved_paths


def load_rows(csv_path):
    with open(csv_path, "r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def main():
    args = parse_args()
    rows = load_rows(args.demographics_csv)

    if not rows:
        raise ValueError(f"No rows found in demographics CSV: {args.demographics_csv}")

    required_cols = [args.subject_col, args.fold_col]
    for col in required_cols:
        if col not in rows[0]:
            raise KeyError(f"Column `{col}` not found in demographics CSV.")

    fold_values = sorted({int(float(str(r[args.fold_col]).strip())) for r in rows})

    print(f"demographics_csv: {args.demographics_csv}")
    print(f"data_path: {args.data_path}")
    print(f"total_rows: {len(rows)}")
    print(f"folds: {fold_values}")
    print()

    for fold in fold_values:
        fold_rows = [r for r in rows if int(float(str(r[args.fold_col]).strip())) == fold]
        missing_subjects = []

        for row in fold_rows:
            sid = str(row[args.subject_col]).strip()
            missing_kinds, resolved_paths = resolve_missing_kinds(args.data_path, sid)
            if missing_kinds:
                missing_subjects.append(
                    {
                        "sid": sid,
                        "missing_kinds": missing_kinds,
                        "resolved_paths": resolved_paths,
                    }
                )

        print(f"===== fold {fold} =====")
        print(f"csv_subjects: {len(fold_rows)}")
        print(f"missing_subjects: {len(missing_subjects)}")

        if missing_subjects:
            kind_counter = Counter()
            for item in missing_subjects:
                kind_counter.update(item["missing_kinds"])
            print(f"missing_kind_counts: {dict(kind_counter)}")
            print("missing_ids:")
            for item in missing_subjects:
                print(f"  {item['sid']} -> missing {', '.join(item['missing_kinds'])}")
                for kind in item["missing_kinds"]:
                    print(f"    {kind}: expected {item['resolved_paths'][kind]}")
        else:
            print("missing_ids: none")

        print()


if __name__ == "__main__":
    main()
