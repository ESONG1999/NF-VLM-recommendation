# embeddings.py
import os
import csv
import pickle
from typing import Dict, List

import torch
from torch import nn
from PIL import Image
from tqdm import tqdm

# Qwen-VL
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers import AutoModelForImageTextToText

# CLIP (open_clip)
import open_clip


class BaseVisionEmbedder(nn.Module):
    def encode_images(self, paths: List[str]) -> torch.Tensor:
        raise NotImplementedError

def find_vision_tower(model):
    # Traverse the whole model tree and find any module with "vision" in name or class
    for name, module in model.named_modules():
        cls_name = module.__class__.__name__.lower()
        if ("vision" in name.lower() or "vision" in cls_name) and "projector" not in name.lower():
            # Reject too-small modules, keep only ViT blocks
            try:
                test = module.forward  # must have forward
                # Found the vision tower
                print(f"[QwenVL] Vision tower detected at: {name} ({module.__class__.__name__})")
                return module
            except:
                pass

    raise ValueError("❌ Could not locate vision tower inside Qwen2.5-VL model.")


class Qwen25VLPureVisionEmbedder:
    def __init__(self, 
                 model_name="Qwen/Qwen2.5-VL-7B-Instruct",
                 device="cuda",
                 dtype=torch.float16):
        self.device = device

        # Load processor for preprocessing
        self.processor = AutoProcessor.from_pretrained(
            model_name, trust_remote_code=True
        )

        print("Loading model (full model, will remove LLM afterwards)...")
        full_model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
            device_map="auto"
        )
        full_model.eval()

        # ⭐ Dynamically search for the vision tower
        self.vision = find_vision_tower(full_model.model)
        self.vision.to(device).eval()

        # Find projector
        self.projector = full_model.model.mm_projector
        self.projector.to(device).eval()

        # Free LLM part
        del full_model.model.language_model
        torch.cuda.empty_cache()

        print("[PureVision] Ready. Only vision tower + projector loaded.")

    @torch.no_grad()
    def encode_images(self, paths):
        images = [Image.open(p).convert("RGB") for p in paths]

        inputs = self.processor(
            images=images,
            return_tensors="pt"
        ).to(self.device)

        pixel_values = inputs.pixel_values

        # Forward vision tower
        vis = self.vision(pixel_values=pixel_values)

        # Extract CLS
        hidden = vis.last_hidden_state
        feats = hidden[:, 0, :]

        # Project to embedding space
        feats = self.projector(feats[:, None, :])[:, 0, :]

        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu()




# -------- CLIP Embedder (open_clip) --------
class CLIPEmbedder(BaseVisionEmbedder):
    def __init__(self, model_name="ViT-H-14", pretrained="laion2b_s32b_b79k", device="cuda"):
        super().__init__()
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model.to(device)
        self.model.eval()

    @torch.no_grad()
    def encode_images(self, paths):
        images = []
        valid_idx = []
    
        for i, p in enumerate(paths):
            try:
                img = Image.open(p).convert("RGB")
                img = self.preprocess(img)
                images.append(img)
                valid_idx.append(i)
            except Exception as e:
                print(f"[WARN] Skipping corrupted image: {p}")
                continue
    
        if len(images) == 0:
            # return zero embeddings if *all* images bad
            return torch.zeros((len(paths), self.model.visual.output_dim))
    
        images = torch.stack(images).to(self.device)
    
        feats = self.model.encode_image(images)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        feats = feats.cpu()
    
        # Create output with original batch size (fill missing with zeros)
        out_dim = feats.shape[-1]
        final = torch.zeros((len(paths), out_dim))
    
        for idx, fidx in enumerate(valid_idx):
            final[fidx] = feats[idx]
    
        return final



# -------- SigLIP Embedder (占位，可按需实现) --------
class SigLIPEembedder(BaseVisionEmbedder):
    def __init__(self, model_name="google/siglip-base-patch16-224", device="cuda"):
        super().__init__()
        raise NotImplementedError("TODO: use transformers AutoModel + AutoImageProcessor")


def build_embedder(kind: str, device="cuda") -> BaseVisionEmbedder:
    kind = kind.lower()
    if kind == "qwen-vl":
        return Qwen25VLPureVisionEmbedder(
            model_name="Qwen/Qwen2.5-VL-7B-Instruct",
            device=device
        )
    elif kind == "clip":
        return CLIPEmbedder(device=device)
    elif kind == "siglip":
        return SigLIPEembedder(device=device)
    else:
        raise ValueError(f"Unknown embedder kind: {kind}")


def build_item_embeddings(
    items_csv: str,
    embedder_kind: str,
    output_emb_path: str,
    output_map_path: str,
    batch_size: int = 64,
    device: str = "cuda",
):
    """
    从 items.csv 读取 item_id, image_path，生成：
    - item_embeddings.pt: [N, D] Tensor
    - item2idx.pkl: {item_id: idx}
    """
    embedder = build_embedder(embedder_kind, device=device)

    rows = []
    with open(items_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    item2idx: Dict[str, int] = {}
    emb_list = []

    print(f"[Embed] Total items in {items_csv}: {len(rows)}")

    for i in tqdm(range(0, len(rows), batch_size), desc=f"Embedding ({embedder_kind})"):
        batch_rows = rows[i : i + batch_size]
        paths = []
        ids = []
        for r in batch_rows:
            iid = r["item_id"]
            img_path = r["image_path"]
            if not os.path.exists(img_path):
                continue
            paths.append(img_path)
            ids.append(iid)

        if not paths:
            continue

        feats = embedder.encode_images(paths)  # [B, D]
        for iid, feat in zip(ids, feats):
            if iid not in item2idx:
                idx = len(item2idx)
                item2idx[iid] = idx
                emb_list.append(feat.unsqueeze(0))

    emb_matrix = torch.cat(emb_list, dim=0)  # [N, D]
    torch.save(emb_matrix, output_emb_path)
    with open(output_map_path, "wb") as f:
        pickle.dump(item2idx, f)

    print(f"[Embed] Saved embeddings to {output_emb_path}, item2idx to {output_map_path}")
    print(f"[Embed] #items={len(item2idx)}, emb_dim={emb_matrix.shape[1]}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--items_csv", type=str, required=True)
    parser.add_argument("--embedder", type=str, default="qwen-vl",
                        choices=["qwen-vl", "clip"])
    parser.add_argument("--output_emb", type=str, required=True)
    parser.add_argument("--output_map", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    build_item_embeddings(
        items_csv=args.items_csv,
        embedder_kind=args.embedder,
        output_emb_path=args.output_emb,
        output_map_path=args.output_map,
        batch_size=args.batch_size,
        device=args.device,
    )
