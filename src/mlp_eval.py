import os
import sys
import argparse
import pickle
import numpy as np
import cv2
from collections import defaultdict
from math import inf

import mediapipe as mp


#use with arguments, full sample below

# If sir asks us for the tf lite version we can run it
try:
    from tflite_runtime.interpreter import Interpreter
except Exception:
    try:
        from tensorflow.lite.python.interpreter import Interpreter
    except Exception:
        print("Error: tflite-runtime or tensorflow not installed.")
        sys.exit(1)

# Use CV2 for the cascades
mp_face = mp.solutions.face_detection
detector = mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5)

#Load the piclkle file and get the embedding of the names
def load_embeddings(path):
    """Load embeddings pickle."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict) and "embeddings" in data and "names" in data:
        return np.array(data["embeddings"]), list(data["names"])
    else:
        raise ValueError("Unrecognized embeddings pickle format")

def build_interpreter(model_path):
    interp = Interpreter(model_path=model_path)
    interp.allocate_tensors()
    return interp, interp.get_input_details(), interp.get_output_details()

def get_embedding_from_face(interp, in_d, out_d, face_rgb):
    h, w = in_d[0]["shape"][1], in_d[0]["shape"][2]
    img = cv2.resize(face_rgb, (w, h)).astype("float32")
    img = (img - 127.5) / 128.0
    inp = np.expand_dims(img, axis=0)
    interp.set_tensor(in_d[0]["index"], inp)
    interp.invoke()
    emb = interp.get_tensor(out_d[0]["index"])[0]
    norm = np.linalg.norm(emb)
    if norm == 0 or np.isnan(norm):
        return np.zeros_like(emb, dtype=np.float32)
    return emb / norm

def detect_face_mediapipe(img_bgr):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    results = detector.process(rgb)
    if not results.detections:
        return None
    det = results.detections[0]
    box = det.location_data.relative_bounding_box
    h, w, _ = img_bgr.shape

    x1 = int(box.xmin * w)
    y1 = int(box.ymin * h)
    x2 = x1 + int(box.width * w)
    y2 = y1 + int(box.height * h)

    face = img_bgr[y1:y2, x1:x2]
    return face, (x1, y1, x2-x1, y2-y1)


def compute_class_centers(embeddings, labels, le):
    """Compute mean embedding per class for Euclidean distance evaluation."""
    centers = {}
    y_encoded = le.transform(labels)
    for i, cls_name in enumerate(le.classes_):
        cls_embs = embeddings[y_encoded == i]
        if len(cls_embs) == 0:
            centers[cls_name] = np.zeros(embeddings.shape[1], dtype=np.float32)
        else:
            centers[cls_name] = np.mean(cls_embs, axis=0)
    return centers

def evaluate(args):
    # Load models
    facenet, in_d, out_d = build_interpreter(args.facenet)
    mlp, mlp_in, mlp_out = build_interpreter(args.mlp)

    # Load embeddings and labels for class centers
    embeddings = np.load(args.emb)
    labels = np.load(args.labels)
    with open(args.encoder, "rb") as f:
        le = pickle.load(f)

    class_centers = compute_class_centers(embeddings, labels, le)

    # Prepare test images
    tests = []
    for fname in sorted(os.listdir(args.testdir)):
        if fname.lower().endswith((".jpg", ".jpeg", ".png")):
            person = ''.join([c for c in fname if not c.isdigit()]).split(".")[0].strip().lower()
            tests.append((person, os.path.join(args.testdir, fname)))
    if len(tests) == 0:
        print("No test images found in", args.testdir)
        return

    results = []
    per_person = defaultdict(lambda: {"tp":0,"fn":0,"fp":0,"total":0})
    name_set = sorted(list(set([p for p,_ in tests] + list(le.classes_) + ["Unknown"])))
    name_to_idx = {n:i for i,n in enumerate(name_set)}
    confmat = np.zeros((len(name_set), len(name_set)), dtype=int)

    for true_name, img_path in tests:
        img = cv2.imread(img_path)
        if img is None:
            print("Failed to load", img_path)
            continue

        face_crop_and_box = detect_face_mediapipe(img)
        if face_crop_and_box is None:
            h, w, _ = img.shape
            cx, cy = w//2, h//2
            s = min(w,h)//2
            face_crop = img[cy-s//2:cy+s//2, cx-s//2:cx+s//2]
        else:
            face_crop, _ = face_crop_and_box

        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        emb = get_embedding_from_face(facenet, in_d, out_d, face_rgb)

        #MLP prediction
        inp = np.expand_dims(emb.astype("float32"), axis=0)
        mlp.set_tensor(mlp_in[0]['index'], inp)
        mlp.invoke()
        preds = mlp.get_tensor(mlp_out[0]['index'])[0]
        class_id = int(np.argmax(preds))
        confidence = float(preds[class_id])
        if confidence < args.threshold:
            pred_name = "Unknown"
        else:
            pred_name = le.inverse_transform([class_id])[0].lower()

        #Euclidean distance
        euclidean_dists = {cls: np.linalg.norm(emb - center) for cls, center in class_centers.items()}
        best_class_by_dist = min(euclidean_dists, key=euclidean_dists.get)
        best_dist = euclidean_dists[best_class_by_dist]

        pred_name_clean = pred_name.strip().lower()
        true_name_clean = true_name.strip().lower()
        results.append({
            "image": img_path,
            "true": true_name_clean,
            "pred_mlp": pred_name_clean,
            "mlp_conf": confidence,
            "best_dist_name": best_class_by_dist,
            "euclidean_dist": best_dist
        })

        true_idx = name_to_idx[true_name_clean]
        pred_idx = name_to_idx[pred_name_clean if pred_name_clean in name_to_idx else "Unknown"]
        confmat[true_idx, pred_idx] += 1

        per_person[true_name_clean]["total"] += 1
        if pred_name_clean == true_name_clean:
            per_person[true_name_clean]["tp"] += 1
        else:
            per_person[true_name_clean]["fn"] += 1
        if pred_name_clean != true_name_clean and pred_name_clean != "Unknown":
            per_person[pred_name_clean]["fp"] += 1

    # Overall accuracy
    total = len(results)
    correct = sum(1 for r in results if r["true"] == r["pred_mlp"])
    accuracy = correct / total if total>0 else 0.0

    print(f"\nTotal tests: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Accuracy (MLP): {accuracy:.4f}")

    # Per-person metrics
    print("\nIndividual metrics:")
    for name in sorted(per_person.keys()):
        stats = per_person[name]
        tp, fn, fp, total_p = stats["tp"], stats["fn"], stats["fp"], stats["total"]
        recall = tp / total_p if total_p>0 else 0.0
        precision = tp / (tp+fp) if (tp+fp)>0 else 0.0
        f1 = 2*precision*recall/(precision+recall) if (precision+recall)>0 else 0.0
        print(f"  {name}: total={total_p}  recall={recall:.3f}  precision={precision:.3f}  f1={f1:.3f}")

    # Confusion matrix
    print("\nConfusion matrix rows=true cols=pred:")
    print("Names:", name_set)
    print(confmat)

    # Save results
    import csv
    csv_path = args.output or "eval_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as cf:
        writer = csv.writer(cf)
        writer.writerow(["image","true","pred_mlp","mlp_conf","best_dist_name","euclidean_dist"])
        for r in results:
            writer.writerow([r["image"], r["true"], r["pred_mlp"], r["mlp_conf"], r["best_dist_name"], r["euclidean_dist"]])
    print(f"\nSaved detailed results to {csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--facenet", required=True, help="Path to facenet.tflite")
    parser.add_argument("--mlp", required=True, help="Path to mlp_classifier.tflite")
    parser.add_argument("--encoder", required=True, help="Path to LabelEncoder pickle")
    parser.add_argument("--emb", required=True, help="Path to embeddings.npy")
    parser.add_argument("--labels", required=True, help="Path to labels.npy")
    parser.add_argument("--testdir", required=True, help="Directory with test images")
    parser.add_argument("--threshold", type=float, default=0.6, help="MLP confidence threshold")
    parser.add_argument("--output", default="eval_results.csv", help="CSV output path")
    args = parser.parse_args()
    evaluate(args)
