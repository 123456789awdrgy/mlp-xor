# -*- coding: utf-8 -*-
"""2-2-1 MLP 手写实现（numpy，无框架），训练 XOR，把学习过程存成 JSON 供可视化。"""
import json
import numpy as np

X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
T = np.array([[0.0], [1.0], [1.0], [0.0]])

W = np.array([[2.0, -2.0], [-2.0, 2.0]])
b1 = np.array([[-1.0, -1.0]])
v = np.array([[3.0], [3.0]])
b2 = np.array([[-2.5]])

LR = 0.5
EPOCHS = 4000
SNAPSHOTS = [0, 50, 200, 1000, 4000]


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def forward(W, b1, v, b2, x=X):
    h = sigmoid(x @ W + b1)
    p = sigmoid(h @ v + b2)
    return h, p


def loss_of(p):
    eps = 1e-12
    return float(-np.mean(T * np.log(p + eps) + (1 - T) * np.log(1 - p + eps)))


def boundary_grid(W, b1, v, b2, n=40):
    xs = np.linspace(-0.5, 1.5, n)
    grid = []
    for y in xs:
        row = []
        for x in xs:
            h = sigmoid(np.array([x, y]) @ W + b1)
            p = float(sigmoid(h @ v + b2).item())
            row.append(round(p, 4))
        grid.append(row)
    return grid


history = []
snapshots = []
track = []
final_hidden = None
final_probs = None

for epoch in range(EPOCHS + 1):
    h, p = forward(W, b1, v, b2)
    loss = loss_of(p)

    if epoch in SNAPSHOTS:
        snapshots.append({
            "epoch": epoch,
            "loss": round(loss, 6),
            "W": W.round(4).tolist(),
            "b1": b1.round(4).tolist(),
            "v": v.round(4).tolist(),
            "b2": b2.round(4).tolist(),
            "grid": boundary_grid(W, b1, v, b2),
        })

    history.append({"epoch": epoch, "loss": round(loss, 6)})

    if epoch % 50 == 0:
        track.append({
            "epoch": epoch,
            "loss": round(loss, 6),
            "W": W.round(4).ravel().tolist(),
            "v": v.round(4).ravel().tolist(),
        })

    if epoch == EPOCHS:
        final_hidden = h.round(4).tolist()
        final_probs = p.round(4).ravel().tolist()
        break

    dz2 = (p - T) / len(X)
    dv = h.T @ dz2
    db2 = dz2.sum(axis=0, keepdims=True)

    dh = dz2 @ v.T
    dz1 = dh * h * (1 - h)
    dW = X.T @ dz1
    db1 = dz1.sum(axis=0, keepdims=True)

    W -= LR * dW
    b1 -= LR * db1
    v -= LR * dv
    b2 -= LR * db2

preds = [1 if p_ >= 0.5 else 0 for p_ in final_probs]

result = {
    "meta": {
        "lr": LR,
        "epochs": EPOCHS,
        "architecture": "2-2-1, sigmoid + BCE",
        "samples": [
            {"x1": int(a), "x2": int(b), "target": int(t[0])}
            for (a, b), t in zip(X, T)
        ],
    },
    "history": history,
    "snapshots": snapshots,
    "weights_track": track,
    "final": {
        "loss": history[-1]["loss"],
        "W": W.round(4).tolist(),
        "b1": b1.round(4).tolist(),
        "v": v.round(4).tolist(),
        "b2": b2.round(4).tolist(),
        "hidden": final_hidden,
        "probs": final_probs,
        "preds": preds,
        "correct": preds == [int(t[0]) for t in T],
    },
}

with open("mlp_data.json", "w", encoding="utf-8") as fh:
    json.dump(result, fh, ensure_ascii=False)

print("loss:", result["final"]["loss"])
print("W =", W.round(3).tolist())
print("v =", v.round(3).tolist())
print("probs =", [round(p_, 4) for p_ in final_probs])
print("preds =", preds, "correct =", result["final"]["correct"])
print("track =", len(track), "snapshots =", len(snapshots))
