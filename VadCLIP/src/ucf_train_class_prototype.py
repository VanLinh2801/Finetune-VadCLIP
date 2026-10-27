import csv
import json
import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
import torch.utils.data as data

import ucf_option_class_prototype
from model import CLIPVAD
from ucf_test_description import test
from utils import tools
from utils.tools import get_batch_label, get_prompt_text


class UCFPrototypeDataset(data.Dataset):
    def __init__(
        self,
        clip_dim,
        file_path,
        test_mode,
        feature_root,
        normal=False,
        return_sample_id=False,
    ):
        with open(file_path, "r", encoding="utf-8") as f:
            self.rows = list(csv.DictReader(f))
        self.clip_dim = clip_dim
        self.test_mode = test_mode
        self.feature_root = Path(feature_root)
        self.return_sample_id = return_sample_id

        if normal and not test_mode:
            self.rows = [row for row in self.rows if row["label"] == "Normal"]
        elif not test_mode:
            self.rows = [row for row in self.rows if row["label"] != "Normal"]

        self.validate_feature_paths()

    def validate_feature_paths(self):
        missing = []
        for row in self.rows:
            feature_path = self.feature_root / row["path"]
            if not feature_path.exists():
                missing.append((row.get("video_id"), row.get("label"), row.get("path")))
        if missing:
            preview = "\n".join(f"  {video_id},{label},{path}" for video_id, label, path in missing[:50])
            extra = "" if len(missing) <= 50 else f"\n  ... and {len(missing) - 50} more"
            raise FileNotFoundError(
                "Missing feature files referenced by list.\n"
                f"feature_root: {self.feature_root}\n"
                f"missing_files: {len(missing)}\n"
                "First missing entries:\n"
                f"{preview}{extra}"
            )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        feature_path = self.feature_root / row["path"]
        clip_feature = np.load(feature_path)

        if not self.test_mode:
            clip_feature, clip_length = tools.process_feat(clip_feature, self.clip_dim)
        else:
            clip_feature, clip_length = tools.process_split(clip_feature, self.clip_dim)

        item = (torch.tensor(clip_feature), row["label"], clip_length, row["video_id"])
        if self.return_sample_id:
            return (*item, Path(row["path"]).stem)
        return item


def length_to_int(length, max_length):
    if hasattr(length, "item"):
        length = length.item()
    length = int(length)
    return max(1, min(length, max_length))


def topk_count(valid_length):
    return max(1, min(valid_length // 16 + 1, valid_length))


def CLASM(logits, labels, lengths, device):
    instance_logits = torch.zeros(0).to(device)
    labels = labels / torch.sum(labels, dim=1, keepdim=True)
    labels = labels.to(device)

    for i in range(logits.shape[0]):
        valid_length = length_to_int(lengths[i], logits.shape[1])
        tmp, _ = torch.topk(logits[i, :valid_length], k=topk_count(valid_length), largest=True, dim=0)
        instance_logits = torch.cat([instance_logits, torch.mean(tmp, 0, keepdim=True)], dim=0)

    milloss = -torch.mean(torch.sum(labels * F.log_softmax(instance_logits, dim=1), dim=1), dim=0)
    return milloss


def CLAS2(logits, labels, lengths, device):
    instance_logits = torch.zeros(0).to(device)
    labels = 1 - labels[:, 0].reshape(labels.shape[0])
    labels = labels.to(device)
    logits = torch.sigmoid(logits).reshape(logits.shape[0], logits.shape[1])

    for i in range(logits.shape[0]):
        valid_length = length_to_int(lengths[i], logits.shape[1])
        tmp, _ = torch.topk(logits[i, :valid_length], k=topk_count(valid_length), largest=True)
        tmp = torch.mean(tmp).view(1)
        instance_logits = torch.cat([instance_logits, tmp], dim=0)

    return F.binary_cross_entropy(instance_logits, labels)


def text_regularization_loss(text_features, device):
    loss = torch.zeros(1).to(device)
    text_feature_normal = text_features[0] / text_features[0].norm(dim=-1, keepdim=True)
    for j in range(1, text_features.shape[0]):
        text_feature_abr = text_features[j] / text_features[j].norm(dim=-1, keepdim=True)
        loss += torch.abs(text_feature_normal @ text_feature_abr)
    return loss / 13 * 1e-1


def get_autocast_context(device, use_amp):
    if device == "cuda" and use_amp:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def encode_texts_with_clip(model, texts, batch_size=128):
    from clip import clip

    embeddings = []
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            tokens = clip.tokenize(batch, truncate=True).to(model.device)
            word_embeddings = model.clipmodel.encode_token(tokens)
            text_features = model.clipmodel.encode_text(word_embeddings, tokens)
            embeddings.append(text_features.detach().float())
    if was_training:
        model.train()
    return torch.cat(embeddings, dim=0)


def load_class_prototype_texts(prototype_json, label_map, prototype_schema, compact_prototype_types):
    with open(prototype_json, "r", encoding="utf-8") as f:
        payload = json.load(f)

    classes = payload.get("classes", {})
    prototype_texts = {}
    for raw_label in label_map:
        class_payload = classes.get(raw_label)
        if not class_payload:
            raise KeyError(f"Missing class prototypes for class: {raw_label}")

        if prototype_schema == "long":
            final_prototypes = class_payload.get("final_prototypes", [])
            texts = [item.get("text", "").strip() for item in final_prototypes if item.get("text", "").strip()]
            if len(texts) != 5:
                raise ValueError(f"Expected 5 final prototypes for {raw_label}, got {len(texts)}")
        else:
            compact_prototypes = class_payload.get("compact_prototypes", [])
            by_type = {
                item.get("type"): item.get("text", "").strip()
                for item in compact_prototypes
                if item.get("type") and item.get("text", "").strip()
            }
            missing_types = [prototype_type for prototype_type in compact_prototype_types if prototype_type not in by_type]
            if missing_types:
                raise ValueError(
                    f"Missing compact prototype types for {raw_label}: {missing_types}. "
                    f"Available types: {sorted(by_type)}"
                )
            texts = [by_type[prototype_type] for prototype_type in compact_prototype_types]

        prototype_texts[raw_label] = texts
    return prototype_texts


def build_prototype_centroids(model, prototype_json, label_map, device, prototype_schema, compact_prototype_types):
    prototype_texts = load_class_prototype_texts(
        prototype_json, label_map, prototype_schema, compact_prototype_types
    )
    centroids = []
    for raw_label in label_map:
        embeddings = encode_texts_with_clip(model, prototype_texts[raw_label])
        embeddings = F.normalize(embeddings.float(), dim=-1)
        centroid = F.normalize(embeddings.mean(dim=0, keepdim=True), dim=-1).squeeze(0)
        centroids.append(centroid)

    prototype_centroids = torch.stack(centroids, dim=0).to(device)
    print("Prototype centroids:", prototype_centroids.shape)
    print("Prototype schema:", prototype_schema)
    if prototype_schema == "compact":
        print("Compact prototype types:", compact_prototype_types)
    return prototype_centroids


def class_prototype_loss(text_features, prototype_centroids):
    text_features = F.normalize(text_features.float(), dim=-1)
    prototype_centroids = F.normalize(prototype_centroids.float(), dim=-1)
    cosine = torch.sum(text_features * prototype_centroids, dim=-1)
    return torch.mean(1 - cosine)


def prototype_logits_from_visual(visual_features, prototype_centroids, temperature):
    visual_features = F.normalize(visual_features.float(), dim=-1)
    prototype_centroids = F.normalize(prototype_centroids.float(), dim=-1)
    return visual_features @ prototype_centroids.t() / temperature


def train(model, normal_loader, anomaly_loader, testloader, args, label_map, device):
    model.to(device)
    Path(args.checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_model_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.save_cur_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.epoch_checkpoint_dir).mkdir(parents=True, exist_ok=True)

    gt = np.load(args.gt_path)
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)

    if args.use_pretrained_model:
        model.load_state_dict(torch.load(args.pretrained_model_path, map_location=device))
        print("Loaded pretrained model:", args.pretrained_model_path)
    else:
        print("Skipped VadCLIP checkpoint load. Training from scratch for VadCLIP-specific layers.")

    for clip_param in model.clipmodel.parameters():
        clip_param.requires_grad = False

    prompt_text = get_prompt_text(label_map)
    prototype_centroids = build_prototype_centroids(
        model,
        args.prototype_json,
        label_map,
        device,
        args.prototype_schema,
        args.compact_prototype_types,
    )

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = MultiStepLR(optimizer, args.scheduler_milestones, args.scheduler_rate)
    use_amp = bool(args.use_amp and device == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_score = 0

    print(
        "Class prototype config:",
        "| train_from_pretrained:", args.use_pretrained_model,
        "| lambda_proto_mil:", args.lambda_proto_mil,
        "| lambda_proto_text:", args.lambda_proto,
        "| prototype_temperature:", args.prototype_temperature,
        "| prototype_mode:", args.prototype_mode,
        "| prototype_schema:", args.prototype_schema,
        "| compact_prototype_types:", args.compact_prototype_types,
        "| prototype_json:", args.prototype_json,
    )

    for e in range(args.max_epoch):
        model.train()
        normal_iter = iter(normal_loader)
        anomaly_iter = iter(anomaly_loader)
        loss_total1 = 0
        loss_total2 = 0
        loss_total_proto_mil = 0
        loss_total_proto_text = 0

        for i in range(min(len(normal_loader), len(anomaly_loader))):
            normal_features, normal_label, normal_lengths, normal_video_ids = next(normal_iter)
            anomaly_features, anomaly_label, anomaly_lengths, anomaly_video_ids = next(anomaly_iter)

            visual_features = torch.cat([normal_features, anomaly_features], dim=0).to(device, non_blocking=True)
            text_labels_raw = list(normal_label) + list(anomaly_label)
            feat_lengths_tensor = torch.cat([normal_lengths, anomaly_lengths], dim=0)
            feat_lengths = [length_to_int(length, args.visual_length) for length in feat_lengths_tensor]
            text_labels = get_batch_label(text_labels_raw, prompt_text, label_map).to(device, non_blocking=True)

            with get_autocast_context(device, use_amp):
                text_features, logits1, logits2, encoded_visual_features = model(
                    visual_features, None, prompt_text, feat_lengths, return_visual=True
                )
                prototype_logits = prototype_logits_from_visual(
                    encoded_visual_features, prototype_centroids, args.prototype_temperature
                )

            loss1 = CLAS2(logits1.float(), text_labels.float(), feat_lengths, device)
            loss2 = CLASM(logits2.float(), text_labels.float(), feat_lengths, device)
            loss_proto_mil = CLASM(prototype_logits.float(), text_labels.float(), feat_lengths, device)
            loss3 = text_regularization_loss(text_features.float(), device)
            loss_proto_text = class_prototype_loss(text_features.float(), prototype_centroids)
            loss = (
                loss1
                + loss2
                + args.lambda_proto_mil * loss_proto_mil
                + loss3
                + args.lambda_proto * loss_proto_text
            )

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_total1 += loss1.item()
            loss_total2 += loss2.item()
            loss_total_proto_mil += loss_proto_mil.item()
            loss_total_proto_text += loss_proto_text.item()

            step = i * normal_loader.batch_size * 2
            if args.eval_steps > 0 and step % args.eval_steps == 0 and step != 0:
                print(
                    "epoch:", e + 1,
                    "| step:", step,
                    "| loss1:", loss_total1 / (i + 1),
                    "| loss2:", loss_total2 / (i + 1),
                    "| loss_proto_mil:", loss_total_proto_mil / (i + 1),
                    "| loss_proto_text:", loss_total_proto_text / (i + 1),
                )
                auc, ap = test(model, testloader, args.visual_length, prompt_text, gt, gtsegments, gtlabels, device)
                if auc > best_score:
                    best_score = auc
                    torch.save(model.state_dict(), args.checkpoint_path)

        scheduler.step()
        num_steps = max(1, min(len(normal_loader), len(anomaly_loader)))
        print(
            "epoch:", e + 1,
            "| loss1:", loss_total1 / num_steps,
            "| loss2:", loss_total2 / num_steps,
            "| loss_proto_mil:", loss_total_proto_mil / num_steps,
            "| loss_proto_text:", loss_total_proto_text / num_steps,
        )
        torch.save(model.state_dict(), args.save_cur_path)
        epoch_checkpoint_path = Path(args.epoch_checkpoint_dir) / f"model_epoch_{e + 1:03d}_class_prototype.pth"
        torch.save(model.state_dict(), epoch_checkpoint_path)
        print("Saved epoch checkpoint:", epoch_checkpoint_path)

    if best_score > 0:
        model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
    torch.save(model.state_dict(), args.output_model_path)
    print("Saved fine-tuned model:", args.output_model_path)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = ucf_option_class_prototype.parser.parse_args()
    setup_seed(args.seed)

    label_map = {
        "Normal": "normal",
        "Abuse": "abuse",
        "Arrest": "arrest",
        "Arson": "arson",
        "Assault": "assault",
        "Burglary": "burglary",
        "Explosion": "explosion",
        "Fighting": "fighting",
        "RoadAccidents": "roadAccidents",
        "Robbery": "robbery",
        "Shooting": "shooting",
        "Shoplifting": "shoplifting",
        "Stealing": "stealing",
        "Vandalism": "vandalism",
    }

    pin_memory = bool(args.pin_memory and device == "cuda")
    dataloader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": pin_memory,
    }
    if args.num_workers > 0:
        dataloader_kwargs["persistent_workers"] = True

    normal_dataset = UCFPrototypeDataset(args.visual_length, args.train_list, False, args.feature_root, True)
    normal_loader = DataLoader(
        normal_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, **dataloader_kwargs
    )
    anomaly_dataset = UCFPrototypeDataset(args.visual_length, args.train_list, False, args.feature_root, False)
    anomaly_loader = DataLoader(
        anomaly_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, **dataloader_kwargs
    )

    test_dataset = UCFPrototypeDataset(args.visual_length, args.test_list, True, args.feature_root, False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, **dataloader_kwargs)

    model = CLIPVAD(
        args.classes_num,
        args.embed_dim,
        args.visual_length,
        args.visual_width,
        args.visual_head,
        args.visual_layers,
        args.attn_window,
        args.prompt_prefix,
        args.prompt_postfix,
        device,
    )

    train(model, normal_loader, anomaly_loader, test_loader, args, label_map, device)
