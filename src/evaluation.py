import os
import sys
import argparse
import pickle
import numpy as np
import cv2
from collections import defaultdict
from math import inf

#use with arguments, full sample below
#python evaluation.py --facenet facenet.tflite --embeddings face_model_tflite.pickle --testdir .\test_dataset --threshold 0.9

# If sir asks us for the tf lite version we can run it
try:
    from tflite_runtime.interpreter import Interpreter
except Exception:
    try:
        from tensorflow.lite.python.interpreter import Interpreter
    except Exception:
        print("Error: tflite-runtime or tensorflow not installed. Install tflite-runtime on Pi.")
        sys.exit(1)

# Use CV2 for the cascades
FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

#Load the piclkle file and get the embedding of the names
def load_embeddings(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict):
        if "embeddings" in data and "names" in data:
            embeddings = np.array(data["embeddings"])
            names = list(data["names"])
            return embeddings, names
        else:
            names = []
            embeddings = []
            for k, v in data.items():
                names.append(k)
                embeddings.append(np.array(v))
            return np.array(embeddings), names
    else:
        raise ValueError("Unrecognized embeddings pickle format")

def build_interpreter(facenet_path):
    interp = Interpreter(model_path=facenet_path)
    interp.allocate_tensors()
    in_d = interp.get_input_details()
    out_d = interp.get_output_details()
    return interp, in_d, out_d


def get_embedding_from_face(interp, in_d, out_d, face_rgb):
    h = in_d[0]["shape"][1]
    w = in_d[0]["shape"][2]
    img = cv2.resize(face_rgb, (w, h)).astype("float32")
    # Normalize images to -1,1 for TFLite (convetional)
    img_norm = (img - 127.5) / 128.0 
    arr = np.expand_dims(img_norm, axis=0)
    interp.set_tensor(in_d[0]["index"], arr)
    interp.invoke()
    emb = interp.get_tensor(out_d[0]["index"])[0]
    norm = np.linalg.norm(emb)
    if norm == 0 or np.isnan(norm):
        return np.zeros_like(emb, dtype=np.float32)
    return emb / norm

def find_best_match(embedding, known_embeddings, known_names, threshold):
    if len(known_embeddings) == 0:
        return "Unknown", inf
    dists = np.linalg.norm(known_embeddings - embedding, axis=1)
    idx = int(np.argmin(dists))
    best = dists[idx]
    if best < threshold:
        return known_names[idx], float(best)
    else:
        return "Unknown", float(best)

def detect_face_opencv(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30,30))
    if len(faces) == 0:
        return None
    x, y, w, h = faces[0]
    return img_bgr[y:y+h, x:x+w], (x, y, w, h)

def evaluate(args):
    interp, in_d, out_d = build_interpreter(args.facenet)
    known_embeddings, known_names = load_embeddings(args.embeddings)

    tests = []

    for fname in sorted(os.listdir(args.testdir)):
        if fname.lower().endswith((".jpg")):
            #Error here that results in a 0 evaluation
            person = ''.join([c for c in fname if not c.isdigit()]).split(".")[0].strip().lower()
            tests.append((person, os.path.join(args.testdir, fname)))

    if len(tests) == 0:
        print("No test images found under", args.testdir)
        return

    results = []
    per_person = defaultdict(lambda: {"tp":0,"fn":0,"fp":0,"total":0})
    
    name_set = sorted(list(set([p for p,_ in tests] + known_names + ["Unknown"])))
    name_to_idx = {n:i for i,n in enumerate(name_set)}
    confmat = np.zeros((len(name_set), len(name_set)), dtype=int)  # rows true, cols pred

    for true_name, img_path in tests:
        img = cv2.imread(img_path)
        if img is None:
            print("Failed to load", img_path)
            continue

        face_and_box = detect_face_opencv(img)
        if face_and_box is None:
            h, w, _ = img.shape
            cx, cy = w//2, h//2
            s = min(w,h)//2
            face_crop = img[cy-s//2:cy+s//2, cx-s//2:cx+s//2]
        else:
            face_crop, (x,y,ww,hh) = face_and_box

        # convert to RGB
        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)

        emb = get_embedding_from_face(interp, in_d, out_d, face_rgb)
        pred_name, dist = find_best_match(emb, known_embeddings, known_names, args.threshold)
        pred_name_clean = pred_name.strip().lower()
        true_name_clean = true_name.strip().lower()

        results.append({
            "image": img_path,
            "true": true_name_clean,
            "pred": pred_name_clean,
            "distance": dist
        })

        true_idx = name_to_idx[true_name_clean]
        pred_idx = name_to_idx[pred_name_clean if pred_name_clean in name_to_idx else "Unknown"]
        confmat[true_idx, pred_idx] += 1

        per_person[true_name]["total"] += 1
        if pred_name_clean == true_name_clean:
            per_person[true_name_clean]["tp"] += 1
        else:
            per_person[true_name_clean]["fn"] += 1
        # false positives
        if pred_name_clean != true_name_clean and pred_name_clean != "Unknown":
            per_person[pred_name_clean]["fp"] += 1

    # Evaluations shown below
    total = len(results)
    correct = sum(1 for r in results if r["true"] == r["pred"])
    accuracy = correct / total if total>0 else 0.0

    print(f"\nTotal tests: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.4f}")

    print("\nIndividual metrics:")
    for name in sorted(per_person.keys()):
        stats = per_person[name]
        tp = stats["tp"]
        fn = stats["fn"]
        fp = stats["fp"]
        total_p = stats["total"]
        recall = tp / total_p if total_p>0 else 0.0
        precision = tp / (tp+fp) if (tp+fp)>0 else 0.0
        f1 = 2*precision*recall/(precision+recall) if (precision+recall)>0 else 0.0
        print(f"  {name}: total={total_p:3d}  recall={recall:.3f}  precision={precision:.3f}  f1={f1:.3f}")

    # Confusion matrix 
    print("\nConfusion matrix rows=true cols=pred:")
    print("Names:", name_set)
    print(confmat)

    import csv
    csv_path = args.output or "eval_results.csv" #Default name
    with open(csv_path, "w", newline="", encoding="utf-8") as cf:
        writer = csv.writer(cf)

        #Per Image
        writer.writerow(["image","true","pred","distance"])
        for r in results:
            writer.writerow([r["image"], r["true"], r["pred"], r["distance"]])

        writer.writerow([])
        # Overall metrics
        writer.writerow(["Overall metrics"])
        writer.writerow(["Total images", total])
        writer.writerow(["Correct predictions", correct])
        writer.writerow(["Accuracy", accuracy])

        writer.writerow([])
        # Per-person metrics
        writer.writerow(["Per-person metrics"])
        writer.writerow(["Name","Total","TP","FN","FP","Recall","Precision","F1"])
        for name in sorted(per_person.keys()):
            stats = per_person[name]
            tp, fn, fp, total_p = stats["tp"], stats["fn"], stats["fp"], stats["total"]
            recall = tp / total_p if total_p>0 else 0.0
            precision = tp / (tp+fp) if (tp+fp)>0 else 0.0
            f1 = 2*precision*recall/(precision+recall) if (precision+recall)>0 else 0.0
            writer.writerow([name, total_p, tp, fn, fp, f"{recall:.3f}", f"{precision:.3f}", f"{f1:.3f}"])
    print(f"\nSaved detailed results to {csv_path}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--facenet", required=True, help="Path to facenet.tflite")
    p.add_argument("--embeddings", required=True, help="Path to embeddings pickle (face_model.pickle or embeddings.pkl)")
    p.add_argument("--testdir", required=True, help="Path to test directory with subfolders for each identity")
    p.add_argument("--threshold", type=float, default=0.9, help="Euclidean distance threshold")
    p.add_argument("--output", default="eval_results.csv", help="CSV output path")
    args = p.parse_args()
    evaluate(args)
