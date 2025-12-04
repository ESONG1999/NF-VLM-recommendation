import argparse
import pandas as pd
import pickle
import os
import numpy as np


# python split_and_clean.py \
#     --items data/amazon2023/items.csv \
#     --item_map data/amazon2023/item2idx_clip.pkl \
#     --interactions data/amazon2023/interactions.csv \
#     --out_dir data/amazon2023


# ===========================
#  Utility: load item2idx.pkl
# ===========================
def load_item_map(item_map_path):
    with open(item_map_path, "rb") as f:
        return pickle.load(f)


# =====================================
#  Step 1: Clean items.csv (remove corrupted)
# =====================================
def clean_items(items_csv, item2idx, out_items, out_corrupted):
    df = pd.read_csv(items_csv)
    all_items = set(df["item_id"].tolist())
    valid_items = set(item2idx.keys())

    corrupted = list(all_items - valid_items)
    pd.DataFrame({"corrupted_item": corrupted}).to_csv(out_corrupted, index=False)

    df_clean = df[df["item_id"].isin(valid_items)]
    df_clean.to_csv(out_items, index=False)

    print(f"[Items] {items_csv}: {len(df)} → {len(df_clean)} (removed {len(df)-len(df_clean)})")
    print(f"[Corrupted Items Logged] {out_corrupted}")

    return valid_items


# =====================================
# Step 2: Split interactions into train/val/test
# =====================================
def split_user_interactions(df_user):
    """
    Split 1 user's interactions:
     - If >=3: last → test, second last → val, rest → train
     - If 2:    1 → train, 1 → test
     - If 1:    1 → train
    """
    if len(df_user) >= 3:
        return (
            df_user.iloc[:-2],
            df_user.iloc[-2:-1],
            df_user.iloc[-1:]
        )
    elif len(df_user) == 2:
        return (
            df_user.iloc[:1],
            df_user.iloc[:0],   # no val
            df_user.iloc[1:]
        )
    else:
        return (
            df_user.iloc[:1],
            df_user.iloc[:0],
            df_user.iloc[:0]
        )


def split_interactions(inter_path, out_dir):
    df = pd.read_csv(inter_path)

    # sort by timestamp or random shuffle
    if "timestamp" in df.columns:
        print("[INFO] Sorting by timestamp...")
        df = df.sort_values(["user_id", "timestamp"])
    else:
        print("[INFO] No timestamp found → random shuffle")
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    train_list, val_list, test_list = [], [], []

    for uid, group in df.groupby("user_id"):
        train_u, val_u, test_u = split_user_interactions(group)
        train_list.append(train_u)
        val_list.append(val_u)
        test_list.append(test_u)

    train_df = pd.concat(train_list, ignore_index=True)
    val_df = pd.concat(val_list, ignore_index=True)
    test_df = pd.concat(test_list, ignore_index=True)

    train_path = os.path.join(out_dir, "interactions_train.csv")
    val_path   = os.path.join(out_dir, "interactions_val.csv")
    test_path  = os.path.join(out_dir, "interactions_test.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"[Split Done] train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    return train_path, val_path, test_path


# =====================================
# Step 3: Clean interactions using valid item list
# =====================================
def clean_interactions(inter_path, valid_items, out_path):
    df = pd.read_csv(inter_path)
    before = len(df)

    # 1.过滤 item
    df = df[df["item_id"].isin(valid_items)]

    # 2.去掉没有任何行为的用户
    df = df.groupby("user_id").filter(lambda x: len(x) > 0)

    df.to_csv(out_path, index=False)
    print(f"[Clean] {inter_path}: {before} → {len(df)} (removed {before-len(df)})")

    return out_path


# =====================================
#            MASTER FUNCTION
# =====================================
def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    # ---- Load item2idx ----
    print("\n[1] Loading item embedding map...")
    item2idx = load_item_map(args.item_map)

    # ---- Step 1: Clean items.csv ----
    print("\n[2] Cleaning items.csv...")
    valid_items = clean_items(
        args.items,
        item2idx,
        os.path.join(args.out_dir, "items_clean.csv"),
        os.path.join(args.out_dir, "corrupted_items.csv")
    )

    # ---- Step 2: Split interactions ----
    print("\n[3] Splitting interactions...")
    train_path, val_path, test_path = split_interactions(args.interactions, args.out_dir)

    # ---- Step 3: Clean interactions ----
    print("\n[4] Cleaning interaction splits...")
    clean_interactions(train_path, valid_items, os.path.join(args.out_dir, "train_clean.csv"))
    clean_interactions(val_path,   valid_items, os.path.join(args.out_dir, "val_clean.csv"))
    clean_interactions(test_path,  valid_items, os.path.join(args.out_dir, "test_clean.csv"))

    print("\n[ALL DONE] Your dataset is ready for model training!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--items", required=True, help="Path to items.csv")
    parser.add_argument("--item_map", required=True, help="Path to item2idx.pkl")
    parser.add_argument("--interactions", required=True, help="Path to interactions.csv")
    parser.add_argument("--out_dir", required=True, help="Output directory")

    args = parser.parse_args()
    main(args)
