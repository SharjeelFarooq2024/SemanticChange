"""CPU, last-block definition/usage fine-tuning pilot on training/validation only."""
from __future__ import annotations

import gc
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))
from Assignment_3.src.advanced_hybrid import word_class_weights
from Assignment_3.src.evaluation import export_jsonl, score_predictions
from Assignment_3.src.experiments import make_split
from Assignment_3.src.improved_hybrid import HybridFeatureBuilder, combine_blocks, query_document_features
from Assignment_3.src.runtime import Measurements, dump_json, load_data, logger, resolve_model, sha256

CONFIG = ROOT / "Assignment_3/configs/neural_pilot.json"


def main():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    output = ROOT / config["output_dir"]
    output.mkdir(parents=True, exist_ok=True)
    log = logger(output)
    measurements = Measurements(output)
    dump_json(output / "config.json", config)
    seed = config["outer_seed"]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(2)
    data, definitions = load_data()
    outer_fit, outer_test = make_split(data, "grouped_text", seed)
    outer = data.iloc[outer_fit]
    split = StratifiedGroupKFold(n_splits=config["validation_folds"], shuffle=True, random_state=config["validation_seed"])
    fit_index, val_index = next(split.split(outer, outer.label_value, outer.text_group))
    fit, val = outer.iloc[fit_index].copy(), outer.iloc[val_index].copy()
    if set(fit.text_group)&set(val.text_group) or (set(fit.sentence_id)|set(val.sentence_id))&set(data.iloc[outer_test].sentence_id):
        raise AssertionError("Pilot partition leakage")
    pd.concat([fit.assign(role="fit"), val.assign(role="validation")])[
        ["sentence_id","word","text_group","role"]].to_csv(output / "splits.csv", index=False)
    feature_config = json.loads((ROOT / "Assignment_3/configs/hybrid_improvement.json").read_text(encoding="utf-8"))
    baseline_setting = {"context_weight":1., "retrieval_weight":.25, "embedding_weight":.5}
    builder = HybridFeatureBuilder(feature_config)
    with measurements.measure("matched_classical_control"):
        xfit = builder.fit_transform(fit)
        xval = builder.transform(val)
        control = LinearSVC(C=1., max_iter=20000, random_state=seed)
        control.fit(combine_blocks(xfit, baseline_setting), fit.label_value,
                    sample_weight=word_class_weights(fit.word, fit.label_value))
        baseline_frame = val.assign(prediction=control.predict(combine_blocks(xval, baseline_setting)))
        control_score = score_predictions(baseline_frame)["official_f1"]
    log.info("Pilot matched control inner-validation official F1 %.4f", control_score)
    del builder, control, xfit, xval
    gc.collect()
    reference_config = json.loads((ROOT / config["reference_run"] / "config.json").read_text(encoding="utf-8"))
    profile = reference_config["models"]["e5_small"]
    path = resolve_model(profile)
    with measurements.measure("pilot_encoder_load"):
        tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
        encoder = AutoModel.from_pretrained(str(path), local_files_only=True)
        encoder.eval().requires_grad_(False)
        layers = encoder.encoder.layer[-config["trainable_encoder_layers"]:]
        for layer in layers:
            layer.requires_grad_(True)
    head = torch.nn.Sequential(torch.nn.Dropout(.1), torch.nn.Linear(encoder.config.hidden_size+10,2))
    trainable_names = {name for name,p in encoder.named_parameters() if p.requires_grad}
    trainable_before = {name:p.detach().clone() for name,p in encoder.named_parameters() if p.requires_grad}
    n_trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)+sum(p.numel() for p in head.parameters())
    log.info("Pilot trainable parameters: %d; fit=%d validation=%d; outer test excluded", n_trainable, len(fit), len(val))
    queries = definitions.set_index("word").definition.to_dict()
    def tokenize(frame):
        return tokenizer([f"query: {word}: {queries[word]}" for word in frame.word],
            ["passage: "+text for text in frame.text], truncation="only_second", max_length=config["max_length"])
    fit_tokens, val_tokens = tokenize(fit), tokenize(val)
    numeric_fit,_ = query_document_features(fit, config["reference_run"])
    numeric_val,_ = query_document_features(val, config["reference_run"])
    scaler = StandardScaler().fit(numeric_fit)
    numeric_fit = torch.tensor(scaler.transform(numeric_fit)*.25, dtype=torch.float32)
    numeric_val = torch.tensor(scaler.transform(numeric_val)*.25, dtype=torch.float32)
    labels = torch.tensor(fit.label_value.to_numpy(), dtype=torch.long)
    weights = torch.tensor(word_class_weights(fit.word, fit.label_value), dtype=torch.float32)
    optimizer = torch.optim.AdamW([
        {"params":[p for p in encoder.parameters() if p.requires_grad], "lr":config["encoder_learning_rate"]},
        {"params":head.parameters(), "lr":config["head_learning_rate"]}], weight_decay=config["weight_decay"])
    def batch(tokens, indices):
        return tokenizer.pad([{key:tokens[key][int(i)] for key in tokens} for i in indices], padding=True, return_tensors="pt")
    def logits(inputs, numeric):
        hidden = encoder(**inputs).last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden*mask).sum(1)/mask.sum(1).clamp(min=1)
        return head(torch.cat([pooled,numeric],dim=1))
    def predict_validation():
        encoder.eval(); head.eval()
        results = []
        with torch.inference_mode():
            for start in range(0,len(val),config["batch_size"]):
                indices = np.arange(start,min(start+config["batch_size"],len(val)))
                results.extend(logits(batch(val_tokens,indices), numeric_val[indices]).argmax(1).numpy().tolist())
        return np.asarray(results)
    results, best = [], None
    start_time = time.perf_counter()
    for epoch in range(1,config["epochs"]+1):
        encoder.eval(); head.train()
        for layer in layers:
            layer.train()
        order = np.random.default_rng(seed+epoch).permutation(len(fit))
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.
        with measurements.measure("pilot_training_epoch", epoch=epoch, n=len(fit), trainable_parameters=n_trainable):
            for step,start in enumerate(range(0,len(order),config["batch_size"])):
                indices = order[start:start+config["batch_size"]]
                output_logits = logits(batch(fit_tokens,indices), numeric_fit[indices])
                loss = (torch.nn.functional.cross_entropy(output_logits,labels[indices],reduction="none")*weights[indices]).mean()
                total_loss += float(loss.detach())*len(indices)
                (loss/config["gradient_accumulation"]).backward()
                if (step+1)%config["gradient_accumulation"] == 0 or start+config["batch_size"]>=len(order):
                    torch.nn.utils.clip_grad_norm_([p for p in encoder.parameters() if p.requires_grad]+list(head.parameters()),1.)
                    optimizer.step(); optimizer.zero_grad(set_to_none=True)
                if step%64 == 0:
                    log.info("Neural pilot epoch=%d fit progress=%d/%d",epoch,min(start+config["batch_size"],len(fit)),len(fit))
        with measurements.measure("pilot_validation_epoch",epoch=epoch,n=len(val)):
            prediction = predict_validation()
            frame = val.assign(prediction=prediction)
            metrics = score_predictions(frame)
        row = {"epoch":epoch,"scope":"inner_validation_only","training_loss":total_loss/len(fit), **metrics}
        results.append(row)
        pd.DataFrame(results).to_csv(output / "validation_epochs.csv", index=False)
        log.info("Neural pilot epoch=%d inner-validation official=%.4f accuracy=%.4f",epoch,metrics["official_f1"],metrics["accuracy"])
        if best is None or metrics["official_f1"]>best["official_f1"]:
            best = row.copy()
            adapted = {name:p.detach().cpu().clone() for name,p in encoder.named_parameters() if name in trainable_names}
            torch.save({"adapted_encoder":adapted,"head":head.state_dict(),"epoch":epoch},output / "best_pilot.pt")
            export_jsonl(frame,output / "best_validation_predictions.jsonl","prediction")
            export_jsonl(frame,output / "validation_gold.jsonl","label_value")
    saved = torch.load(output / "best_pilot.pt",map_location="cpu",weights_only=True)
    encoder.load_state_dict(saved["adapted_encoder"],strict=False)
    head.load_state_dict(saved["head"])
    replay = predict_validation()
    replay_score = score_predictions(val.assign(prediction=replay))["official_f1"]
    assert np.isclose(replay_score,best["official_f1"],atol=1e-12)
    update_norm = float(sum((saved["adapted_encoder"][name]-trainable_before[name]).square().sum() for name in trainable_names).sqrt())
    assert update_norm>0
    joblib_payload = {"scaler_mean":scaler.mean_.tolist(),"scaler_scale":scaler.scale_.tolist()}
    dump_json(output / "checkpoint_metadata.json",{"model_profile":profile,"trainable_parameter_names":sorted(trainable_names),
        "numeric_scaler":joblib_payload,"checkpoint_sha256":sha256(output / "best_pilot.pt"),"checkpoint_replayed":True,
        "encoder_update_l2":update_norm,"best_epoch":best["epoch"],"scope":"inner_validation_only"})
    dump_json(output / "pilot_summary.json",{"scope":"inner_validation_only_not_outer_test",
        "matched_control_validation_f1":control_score,"best_neural_validation_f1":best["official_f1"],
        "best_epoch":best["epoch"],"validation_gain":best["official_f1"]-control_score,
        "full_run_promising":bool(best["official_f1"]>=control_score+config["minimum_validation_gain_for_full_run"]),
        "test_evaluation_performed":False,"trainable_parameters":n_trainable,"wall_seconds":time.perf_counter()-start_time,
        "notes":config["notes"]})
    log.info("Pilot complete. Best validation %.4f versus control %.4f; outer test not scored",best["official_f1"],control_score)


if __name__ == "__main__":
    main()
