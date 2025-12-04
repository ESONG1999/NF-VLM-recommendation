import os
import csv
import random
import pickle
import requests
from tqdm import tqdm
from datasets import load_dataset
from concurrent.futures import ThreadPoolExecutor, as_completed

def download_one_image(args):
    asin, urls, img_path = args

    for url in urls:
        try:
            img_data = requests.get(url, timeout=5).content
            with open(img_path, "wb") as f:
                f.write(img_data)
            return asin, img_path, url, True
        except Exception:
            continue

    return asin, img_path, None, False


def parallel_download_images_and_write_csv(
    items_to_download,
    meta_map,
    img_dir,
    items_csv_path,
    max_workers=32,
):
    """
    并行下载，把成功的图片实时写入 items.csv
    """

    # 打开 CSV 写入句柄
    csv_f = open(items_csv_path, "w", newline="", encoding="utf-8")
    writer = csv.writer(csv_f)
    writer.writerow(["item_id", "image_path", "img_url"])

    tasks = []
    for asin in items_to_download:
        img_path = os.path.join(img_dir, f"{asin}.jpg")
        urls = meta_map[asin]
        tasks.append((asin, urls, img_path))

    print(f"[Amazon] Parallel image download ({max_workers} threads) ...")

    success_count = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(download_one_image, t): t[0] for t in tasks
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading"):
            asin, img_path, url, ok = future.result()

            if ok:
                writer.writerow([asin, img_path, url])
                success_count += 1

    csv_f.close()
    print(f"[Amazon] DONE: {success_count} / {len(items_to_download)} images downloaded.")

    return success_count


def load_or_build_meta_map(category, out_dir):
    """
    如果 meta_map.pkl 存在则直接加载；
    否则构建 meta_map 并保存到本地。
    """
    meta_pkl = os.path.join(out_dir, "meta_map.pkl")

    # ----------- 1. 直接加载缓存 -----------
    if os.path.exists(meta_pkl):
        print(f"[Amazon] Loading cached meta_map from {meta_pkl} ...")
        with open(meta_pkl, "rb") as f:
            return pickle.load(f)

    # ----------- 2. 重新构建 -----------
    print("[Amazon] Loading metadata from HuggingFace ...")
    ds_meta = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023",
        f"raw_meta_{category}",
        split="full",
        trust_remote_code=True,
    )

    meta_map = {}
    for m in tqdm(ds_meta, desc="Building meta_map"):
        asin = m.get("parent_asin")
        if asin is None:
            continue

        imgs = m.get("images", {})
        urls = []
        for k in ["hi_res", "large", "thumb", "variant"]:
            lst = imgs.get(k)
            if lst:
                urls.extend([u for u in lst if u])

        if urls:
            meta_map[asin] = urls

    # ----------- 3. 保存缓存 -----------
    with open(meta_pkl, "wb") as f:
        pickle.dump(meta_map, f)

    print(f"[Amazon] meta_map saved to {meta_pkl}")
    return meta_map



def prepare_amazon(
    category="Clothing_Shoes_and_Jewelry",
    out_dir="data/amazon2023",
    max_items=None,
    max_users=None,
    neg_per_pos=4,
):
    """
    最终版：
    - 使用 pickle 缓存 meta_map
    - 写 interactions (positive + negative)
    - 根据 interactions 决定要下载哪些 item
    - 写 items.csv + 下载图片
    """
    os.makedirs(out_dir, exist_ok=True)
    img_dir = os.path.join(out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    # --------------------------------------------------------
    # 1) 加载或构建 meta_map（item->urls），极大加速流程
    # --------------------------------------------------------
    meta_map = load_or_build_meta_map(category, out_dir)
    all_items = list(meta_map.keys())

    # --------------------------------------------------------
    # 2) 加载 review 构建 positive interactions
    # --------------------------------------------------------
    print("[Amazon] Loading reviews ...")
    ds_rev = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023",
        f"raw_review_{category}",
        split="full",
        trust_remote_code=True,
    )

    interactions_path = os.path.join(out_dir, "interactions.csv")
    user_pos = {}

    print("[Amazon] Writing positives ...")
    with open(interactions_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["user_id", "item_id", "label"])

        seen_users = set()

        for r in tqdm(ds_rev):
            u = r["user_id"]
            asin = r["parent_asin"]
            if asin not in meta_map:
                continue

            w.writerow([u, asin, 1])

            if u not in user_pos:
                user_pos[u] = set()
            user_pos[u].add(asin)

            seen_users.add(u)
            if max_users and len(seen_users) >= max_users:
                break

    # --------------------------------------------------------
    # 3) 负采样 negative interactions
    # --------------------------------------------------------
    print(f"[Amazon] Negative sampling: {neg_per_pos} per positive ...")
    user_neg_items = {}

    with open(interactions_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)

        for u, pos_set in tqdm(user_pos.items(), desc="Sampling negatives"):
            n_pos = len(pos_set)
            n_negs = n_pos * neg_per_pos

            pos_set = set(pos_set)
            negs = set()

            while len(negs) < n_negs:
                neg = random.choice(all_items)
                if neg not in pos_set:
                    negs.add(neg)

            user_neg_items[u] = negs

            for neg in negs:
                w.writerow([u, neg, 0])

    # --------------------------------------------------------
    # 4) 决定哪些 item 需要下载图片
    # --------------------------------------------------------
    items_to_download = set()

    for u in user_pos:
        items_to_download |= user_pos[u]

    for u in user_neg_items:
        items_to_download |= user_neg_items[u]

    # 控制规模
    if max_items:
        items_to_download = set(list(items_to_download)[:max_items])

    print(f"[Amazon] Total items to download = {len(items_to_download)}")

    # --------------------------------------------------------
    # 5) 并行下载图片 + 实时写入 items.csv
    # --------------------------------------------------------
    items_path = os.path.join(out_dir, "items.csv")
    
    parallel_download_images_and_write_csv(
        items_to_download=items_to_download,
        meta_map=meta_map,
        img_dir=img_dir,
        items_csv_path=items_path,
        max_workers=32,   # 可调
    )



if __name__ == "__main__":
    prepare_amazon(
        category="Clothing_Shoes_and_Jewelry",
        out_dir="data/amazon2023",
        max_items=None,   # 可调
        max_users=10000,   # 可调
        neg_per_pos=4,
    )
