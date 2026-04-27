import argparse
import csv
import os
from collections import Counter, defaultdict


REQUIRED_KINDS = [
    "anatomical",
    "commissural",
    "left_hemisphere",
    "right_hemisphere",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a new 500-subject CSV, keep the smallest SUB_IDs after filtering, then assign 5 folds in order."
    )
    parser.add_argument(
        "--reference-csv",
        default="/data05/learn2reg/zixi/1DCNN/csv_HCP_1000/new_500.csv",
        type=str,
        help="Previous CSV used as the subject selection list.",
    )
    parser.add_argument(
        "--demographics-csv",
        default="/data05/learn2reg/HCP_1000.csv",
        type=str,
        help="More complete CSV used to look up SUB_ID and DX_GROUP.",
    )
    parser.add_argument(
        "--data-path",
        required=True,
        type=str,
        help="Root data directory used by training.",
    )
    parser.add_argument(
        "--output-csv",
        default=os.path.join(os.path.dirname(__file__), "DTI", "new_500_sorted_fold.csv"),
        type=str,
        help="Where to write the rebuilt CSV.",
    )
    parser.add_argument(
        "--subject-col",
        default="SUB_ID",
        type=str,
        help="Subject ID column name.",
    )
    parser.add_argument(
        "--label-col",
        default="DX_GROUP",
        type=str,
        help="Label column name.",
    )
    parser.add_argument(
        "--target-size",
        default=500,
        type=int,
        help="How many kept subjects to write into the new CSV.",
    )
    parser.add_argument(
        "--num-folds",
        default=5,
        type=int,
        help="Number of folds.",
    )
    return parser.parse_args()


def load_rows(csv_path):
    with open(csv_path, "r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


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


def subject_is_complete(root_dir, sid):
    primary, fallback = build_candidate_paths(root_dir, sid)
    for kind in REQUIRED_KINDS:
        if os.path.exists(primary[kind]) or os.path.exists(fallback[kind]):
            continue
        return False
    return True


def deduplicate_rows(rows, subject_col):
    dedup = {}
    duplicates = defaultdict(int)
    for row in rows:
        sid = str(row[subject_col]).strip()
        if sid in dedup:
            duplicates[sid] += 1
            continue
        dedup[sid] = row
    return dedup, duplicates


def sort_subject_rows(rows, subject_col):
    def sort_key(row):
        sid = str(row[subject_col]).strip()
        return (0, int(sid)) if sid.isdigit() else (1, sid)

    return sorted(rows, key=sort_key)


def assign_folds_in_order(rows, num_folds):
    if len(rows) % num_folds != 0:
        raise ValueError(f"Selected row count {len(rows)} is not divisible by num_folds={num_folds}.")

    fold_size = len(rows) // num_folds
    folded_rows = []
    for idx, row in enumerate(rows):
        fold = idx // fold_size
        new_row = dict(row)
        new_row["fold"] = fold
        folded_rows.append(new_row)
    return folded_rows


def write_rows(rows, output_csv, subject_col, label_col):
    out_dir = os.path.dirname(output_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(output_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=[subject_col, label_col, "fold"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    subject_col: row[subject_col],
                    label_col: row[label_col],
                    "fold": row["fold"],
                }
            )


def summarize(rows, label_col):
    label_counts = Counter(str(row[label_col]).strip() for row in rows)
    fold_counts = Counter(int(row["fold"]) for row in rows)
    fold_label_counts = defaultdict(Counter)
    for row in rows:
        fold_label_counts[int(row["fold"])][str(row[label_col]).strip()] += 1
    return label_counts, fold_counts, fold_label_counts


def main():
    args = parse_args()

    reference_rows = load_rows(args.reference_csv)
    demographics_rows = load_rows(args.demographics_csv)

    if not reference_rows:
        raise ValueError(f"No rows found in reference CSV: {args.reference_csv}")
    if not demographics_rows:
        raise ValueError(f"No rows found in demographics CSV: {args.demographics_csv}")

    for col in [args.subject_col]:
        if col not in reference_rows[0]:
            raise KeyError(f"Column `{col}` not found in reference CSV.")
        if col not in demographics_rows[0]:
            raise KeyError(f"Column `{col}` not found in demographics CSV.")

    if args.label_col not in demographics_rows[0]:
        raise KeyError(f"Column `{args.label_col}` not found in demographics CSV.")

    reference_map, reference_duplicates = deduplicate_rows(reference_rows, args.subject_col)
    demographics_map, demographics_duplicates = deduplicate_rows(demographics_rows, args.subject_col)

    missing_in_demographics = []
    missing_on_disk = []
    selected_rows = []

    for sid, ref_row in reference_map.items():
        if sid not in demographics_map:
            missing_in_demographics.append(sid)
            continue
        if not subject_is_complete(args.data_path, sid):
            missing_on_disk.append(sid)
            continue

        demo_row = demographics_map[sid]
        selected_rows.append(
            {
                args.subject_col: sid,
                args.label_col: demo_row[args.label_col],
            }
        )

    selected_rows = sort_subject_rows(selected_rows, args.subject_col)

    if len(selected_rows) < args.target_size:
        raise ValueError(
            f"After matching reference CSV with demographics CSV and data-path, got {len(selected_rows)} subjects, "
            f"which is fewer than target-size {args.target_size}."
        )

    dropped_rows = selected_rows[args.target_size:]
    selected_rows = selected_rows[:args.target_size]

    folded_rows = assign_folds_in_order(selected_rows, args.num_folds)
    write_rows(folded_rows, args.output_csv, args.subject_col, args.label_col)

    label_counts, fold_counts, fold_label_counts = summarize(folded_rows, args.label_col)

    print(f"reference_csv: {args.reference_csv}")
    print(f"demographics_csv: {args.demographics_csv}")
    print(f"data_path: {args.data_path}")
    print(f"output_csv: {args.output_csv}")
    print(f"reference_rows: {len(reference_rows)}")
    print(f"reference_unique_subjects: {len(reference_map)}")
    print(f"reference_duplicate_subject_ids_skipped: {len(reference_duplicates)}")
    print(f"demographics_rows: {len(demographics_rows)}")
    print(f"demographics_unique_subjects: {len(demographics_map)}")
    print(f"demographics_duplicate_subject_ids_skipped: {len(demographics_duplicates)}")
    print(f"matched_subjects_before_cut: {len(selected_rows) + len(dropped_rows)}")
    print(f"dropped_after_sorting_to_target_size: {len(dropped_rows)}")
    print(f"selected_subjects: {len(folded_rows)}")
    print(f"missing_in_demographics: {len(missing_in_demographics)}")
    print(f"missing_on_disk: {len(missing_on_disk)}")
    print(f"label_counts: {dict(label_counts)}")
    print(f"fold_counts: {dict(sorted(fold_counts.items()))}")
    print("fold_label_counts:")
    for fold in sorted(fold_label_counts):
        print(f"  fold {fold}: {dict(fold_label_counts[fold])}")

    if missing_in_demographics:
        print("missing_in_demographics_ids:")
        for sid in missing_in_demographics:
            print(f"  {sid}")

    if missing_on_disk:
        print("missing_on_disk_ids:")
        for sid in missing_on_disk:
            print(f"  {sid}")

    if dropped_rows:
        print("dropped_after_sorting_ids:")
        for row in dropped_rows:
            print(f"  {row[args.subject_col]}")


if __name__ == "__main__":
    main()
