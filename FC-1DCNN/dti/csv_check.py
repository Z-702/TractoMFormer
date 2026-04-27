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
        description="Compare demographics CSV subjects against the current data-path contents."
    )
    parser.add_argument(
        "--demographics-csv",
        default="/data05/learn2reg/zixi/1DCNN/csv_HCP_1000/new_500.csv",
        type=str,
        help="Path to the demographics CSV.",
    )
    parser.add_argument(
        "--data-path",
        required=True,
        type=str,
        help="Root data directory used by training.",
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


def load_csv_rows(csv_path):
    with open(csv_path, "r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No rows found in demographics CSV: {csv_path}")
    return rows


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


def check_subject(root_dir, sid):
    primary, fallback = build_candidate_paths(root_dir, sid)
    status = {}
    for kind in REQUIRED_KINDS:
        if os.path.exists(primary[kind]):
            status[kind] = ("primary", primary[kind])
        elif os.path.exists(fallback[kind]):
            status[kind] = ("fallback", fallback[kind])
        else:
            status[kind] = ("missing", fallback[kind])
    return status


def collect_disk_subject_ids(root_dir):
    subject_dirs = set()
    if os.path.isdir(root_dir):
        for name in os.listdir(root_dir):
            full = os.path.join(root_dir, name)
            if os.path.isdir(full) and name != "UKF_2T_AtlasSpace":
                subject_dirs.add(name)

    ukf_root = os.path.join(root_dir, "UKF_2T_AtlasSpace")
    ukf_ids = set()
    if os.path.isdir(ukf_root):
        for subdir in [
            "anatomical_tracts",
            "tracts_commissural",
            "tracts_left_hemisphere",
            "tracts_right_hemisphere",
        ]:
            folder = os.path.join(ukf_root, subdir)
            if not os.path.isdir(folder):
                continue
            for name in os.listdir(folder):
                if name.lower().endswith(".csv"):
                    ukf_ids.add(os.path.splitext(name)[0])
    return subject_dirs, ukf_ids


def main():
    args = parse_args()
    rows = load_csv_rows(args.demographics_csv)
    if args.subject_col not in rows[0]:
        raise KeyError(f"Column `{args.subject_col}` not found in demographics CSV.")

    csv_subjects = []
    for row in rows:
        sid = str(row[args.subject_col]).strip()
        fold = row.get(args.fold_col, "")
        csv_subjects.append((sid, fold))

    subject_dirs, ukf_ids = collect_disk_subject_ids(args.data_path)

    missing_any = []
    complete_subjects = []
    missing_kind_counter = Counter()
    fold_missing_counter = Counter()
    source_counter = Counter()

    for sid, fold in csv_subjects:
        status = check_subject(args.data_path, sid)
        missing_kinds = [kind for kind, (src, _) in status.items() if src == "missing"]

        if missing_kinds:
            missing_any.append((sid, fold, missing_kinds, status))
            missing_kind_counter.update(missing_kinds)
            fold_missing_counter.update([str(fold)])
        else:
            complete_subjects.append((sid, fold, status))
            for kind in REQUIRED_KINDS:
                source_counter.update([status[kind][0]])

    csv_ids = {sid for sid, _ in csv_subjects}
    disk_only_subject_dirs = sorted(subject_dirs - csv_ids)
    disk_only_ukf_ids = sorted(ukf_ids - csv_ids)
    csv_missing_from_both = sorted(
        sid for sid in csv_ids if sid not in subject_dirs and sid not in ukf_ids
    )

    print(f"demographics_csv: {args.demographics_csv}")
    print(f"data_path: {args.data_path}")
    print(f"csv_rows: {len(csv_subjects)}")
    print(f"csv_unique_subjects: {len(csv_ids)}")
    print(f"subject_dirs_on_disk: {len(subject_dirs)}")
    print(f"ukf_subject_ids_on_disk: {len(ukf_ids)}")
    print()

    print("=== Summary ===")
    print(f"complete_subjects: {len(complete_subjects)}")
    print(f"subjects_with_missing_files: {len(missing_any)}")
    print(f"missing_kind_counts: {dict(missing_kind_counter)}")
    print(f"missing_by_fold: {dict(sorted(fold_missing_counter.items(), key=lambda kv: int(kv[0])))}")
    print(f"resolved_source_counts_for_complete_subjects: {dict(source_counter)}")
    print(f"csv_subjects_missing_from_both_primary_and_ukf: {len(csv_missing_from_both)}")
    print(f"disk_only_subject_dirs_not_in_csv: {len(disk_only_subject_dirs)}")
    print(f"disk_only_ukf_ids_not_in_csv: {len(disk_only_ukf_ids)}")
    print()

    print("=== CSV Subjects Missing From Both Primary And UKF ===")
    if csv_missing_from_both:
        for sid in csv_missing_from_both:
            print(sid)
    else:
        print("none")
    print()

    print("=== Subjects With Missing Required Files ===")
    if missing_any:
        for sid, fold, missing_kinds, status in missing_any:
            print(f"{sid} (fold={fold}) -> missing {', '.join(missing_kinds)}")
            for kind in missing_kinds:
                print(f"  {kind}: expected {status[kind][1]}")
    else:
        print("none")
    print()

    print("=== Disk Subject Dirs Not In CSV ===")
    if disk_only_subject_dirs:
        for sid in disk_only_subject_dirs:
            print(sid)
    else:
        print("none")
    print()

    print("=== UKF Subject IDs Not In CSV ===")
    if disk_only_ukf_ids:
        for sid in disk_only_ukf_ids:
            print(sid)
    else:
        print("none")


if __name__ == "__main__":
    main()
