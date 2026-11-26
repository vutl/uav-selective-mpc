"""Paper A (fd_v1) qualitative figure: what the mode decisions of H*, nominal MPC and guarded MPC look like on real
VisDrone frames. Stored results only (no simulation): the modes and outcomes come from the Phase 2 TD delivery logs
(runs/uav_semcom/paperA_fd/phase2/test_TD.deliv.json.gz); payload, PSNR and recall of each mode come from the fd_v1
profile / recall tables. The frames are re-encoded with the same codecs (CPU) only to draw them; the script checks
that the re-encoded payload and the re-detected TP count equal the stored ones and prints any difference.

Occurrences (same seed, same image occurrence, the three arms); chosen by inspection among TD occurrences with >= 30
labelled objects in which all three arms picked a conventional / learned codec (not a generative mode):
  (a) seed 20031, 9999973_00000_d_0000017: nominal MPC fails to deliver in time with a large MBT2018 q8 payload,
      H* delivers on time below the image's Qmin, guarded MPC delivers a compliant MBT2018 q3 image.
  (b) seed 20040, 0000105_02931_d_0000074: all three compliant; guarded MPC sends the smallest payload (AVIF q5)
      and detects fewer objects (the return-recall trade-off of Section 5.5).
These are illustrations, not representative samples.

  python paperA_fd_qualitative.py OUT_DIR [--weights best.pt] [--images DIR]
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import a5_common as C  # noqa: E402
import a5_fidelity as AF  # noqa: E402
import fd_common as FD  # noqa: E402
import imcodecs as IC  # noqa: E402
import paperA_fd_figures  # noqa: E402,F401  (shared rcParams / fonts)
from fd_posthoc import ctx, load  # noqa: E402
from profile_a11 import load as load_img  # noqa: E402

TD_DIR = Path("/home/uservutl/data/visdrone_dl/VisDrone2019-DET-test-dev")
YOLO_W = Path("/home/uservutl/work/a5_det/yolo11s_visdrone_960/weights/best.pt")
EXAMPLES = [dict(seed=20031, image="9999973_00000_d_0000017.jpg", occ=0,
                 title="(a) Rate over-estimation: the nominal MPC over-commits"),
            dict(seed=20040, image="0000105_02931_d_0000074.jpg", occ=0,
                 title="(b) Smaller payloads cost detections")]
ARMS = [("H", "", "H* (fixed AVIF q20)", "#4d4d4d"), ("nc", "+mpcnc", "Nominal MPC", "#d95319"),
        ("ng", "+mpcng", "Guarded MPC", "#0072bd")]
GREEN, RED, INK, MUTED = "#1a9850", "#d73027", "#1a1a1a", "#666666"
CW, CH = 360, 225                                       # crop (pixels, native resolution)


def pretty(m):
    if m.startswith("avif_"):
        return f"AVIF q{m[5:]}"
    if m.startswith("mbt2018-mean_q"):
        return f"MBT2018 q{m[14:]}"
    if m.startswith("jpeg2000_"):
        return f"JPEG2000 r{m[9:]}"
    return m


def family(m):
    for k in ("avif", "mbt2018", "jpeg2000"):
        if m.startswith(k):
            return k
    return "gen"


FAM = {"avif": ("o", "AVIF"), "mbt2018": ("s", "MBT2018"), "jpeg2000": ("^", "JPEG2000"),
       "gen": ("x", "thumbnail / generative")}


def outcome(ok, ot):
    return "compliant" if ok else ("on time, below Qmin" if ot else "not delivered in time")


def crop_box(g, W, H):
    """crop window (CW x CH) holding the most labelled object centres (grid search, 20 px steps)."""
    cx, cy = (g[:, 0] + g[:, 2]) / 2, (g[:, 1] + g[:, 3]) / 2
    best = (-1, 0, 0)
    for x0 in range(0, max(1, W - CW + 1), 20):
        for y0 in range(0, max(1, H - CH + 1), 20):
            n = int(((cx >= x0) & (cx < x0 + CW) & (cy >= y0) & (cy < y0 + CH)).sum())
            best = max(best, (n, -x0, -y0))
    return -best[1], -best[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--weights", default=str(YOLO_W))
    ap.add_argument("--images", default=str(TD_DIR))
    a = ap.parse_args()
    out, src = Path(a.out), Path(a.images)
    t, CX = ctx()
    rec, row_of = CX[0], CX[1]
    P = load("phase2", "test_TD", "TD", CX)
    Hh = json.loads((FD.OUT / "phase2" / "tune.json").read_text())["H"]
    man = json.loads(FD.MANIFEST.read_text())["images"]
    mi_of = {e["name"]: k for k, e in enumerate(man)}
    from ultralytics import YOLO
    model = YOLO(a.weights)
    codecs = {}

    def codec(m):
        if m not in codecs:
            if m.startswith("mbt2018-mean_q"):
                codecs[m] = IC.CompressAICodec("mbt2018-mean", int(m[14:]), "cpu")
            else:
                fmt, lv = m.split("_")
                codecs[m] = IC.PillowCodec({"avif": "AVIF", "jpeg2000": "JPEG2000"}[fmt], int(lv))
        return codecs[m]

    fig = plt.figure(figsize=(7.0, 3.55))
    gs = fig.add_gridspec(2, 5, width_ratios=[1, 1, 1, 0.13, 1.0], left=0.005, right=0.985, top=0.90, bottom=0.115,
                          wspace=0.06, hspace=0.62)
    report = []
    for row, ex in enumerate(EXAMPLES):
        mi = mi_of[ex["image"]]
        r = row_of[mi]
        e = P.seeds.index(ex["seed"])
        img = load_img(src / "images" / ex["image"], 0)
        Hp, Wp = img.shape[:2]
        g, gc, ig = AF.read_gt(src / "annotations" / (Path(ex["image"]).stem + ".txt"))
        x0, y0 = crop_box(g, Wp, Hp)
        ngt = int(rec["ngt"][r])
        chosen = {}
        for col, (k, suf, label, colr) in enumerate(ARMS):
            eps = P.arms[("N5x2", f"h={Hh}{suf}")][-1][1][e]
            x = [d for d in eps if d[0] == mi][ex["occ"]]
            m = C.MODES[x[1]]
            chosen[k] = x[1]
            data = codec(m).encode(img)
            dec = codec(m).decode(data)
            bx, sc, cl = AF.detect(model, dec)
            keep = ~AF.in_ignore(bx, ig) & (sc >= AF.CONF)
            mg, _ = AF.match(bx[keep], cl[keep], g, gc)
            kb, ps, tp = t.bits[r, x[1]] / 1e3, float(t.psnr[r, x[1]]), float(rec["tp"][r, x[1]])
            chk = dict(arm=k, mode=m, kbit_stored=round(kb, 2), kbit_reenc=round(8 * len(data) / 1e3, 2),
                       tp_stored=tp, tp_redet=int(mg.sum()), ngt_stored=ngt, ngt_file=len(g), ok=x[2], ontime=x[3])
            report.append(dict(seed=ex["seed"], image=ex["image"], **chk))
            ax = fig.add_subplot(gs[row, col])
            ax.imshow(dec[y0:y0 + CH, x0:x0 + CW], interpolation="nearest")
            lost = not x[3]
            if lost:
                ax.add_patch(mpatches.Rectangle((0, 0), CW, CH, color="white", alpha=0.62, lw=0))
                ax.text(CW / 2, CH / 2, "not delivered\nby the deadline", ha="center", va="center", fontsize=6.6,
                        fontweight="bold", color=colr)
            else:
                for b, hit in zip(g, mg):
                    if b[2] < x0 or b[0] > x0 + CW or b[3] < y0 or b[1] > y0 + CH:
                        continue
                    ax.add_patch(mpatches.Rectangle((b[0] - x0, b[1] - y0), b[2] - b[0], b[3] - b[1], fill=False,
                                                    lw=0.7, ec=GREEN if hit else RED))
            ax.set_xlim(0, CW)
            ax.set_ylim(CH, 0)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for s in ax.spines.values():
                s.set_edgecolor(colr)
                s.set_linewidth(1.4)
            ax.set_title(label, fontsize=6.6, fontweight="bold", color=colr, pad=2.2)
            det = (f"would detect {tp:.0f}/{ngt}" if lost else f"{tp:.0f}/{ngt} detected")
            ax.text(0.5, -0.035, f"{pretty(m)}, {kb:,.0f} kbit, {ps:.1f} dB\n{det}; {outcome(x[2], x[3])}",
                    transform=ax.transAxes, ha="center", va="top", fontsize=5.6, color=INK, linespacing=1.15)
        # mode space of this image: payload vs detection recall for the 14 modes
        ax = fig.add_subplot(gs[row, 4])
        kb_all = t.bits[r] / 1e3
        rc_all = rec["tp"][r] / ngt
        for fam, (mk, lab) in FAM.items():
            idx = [j for j, m in enumerate(C.MODES) if family(m) == fam]
            ax.scatter(kb_all[idx], rc_all[idx], marker=mk, s=12 if mk != "x" else 10,
                       **(dict(color=MUTED) if mk == "x" else dict(facecolors="none", edgecolors=MUTED)), linewidths=0.7, label=lab,
                       zorder=2)
        offs = {"H": (-7, 0), "nc": (0, -10), "ng": (0, -10)}
        for k, _, label, colr in ARMS:
            j = chosen[k]
            ax.scatter([kb_all[j]], [rc_all[j]], s=34, facecolors="none", edgecolors=colr, linewidths=1.3, zorder=3)
        shared = {}
        for k, _, label, colr in ARMS:
            shared.setdefault(chosen[k], []).append((k, colr))
        for j, lst in shared.items():
            txt = " = ".join({"H": "H*", "nc": "nom.", "ng": "guard."}[k] for k, _ in lst)
            dx, dy = offs[lst[0][0]]
            ax.annotate(txt, (kb_all[j], rc_all[j]), xytext=(dx, dy), textcoords="offset points", fontsize=5.6,
                        color=lst[0][1], fontweight="bold", ha="center" if dx == 0 else ("left" if dx > 0 else "right"), va="center")
        ax.set_xscale("log")
        ax.set_xlim(0.3, 2e4)
        ax.set_ylim(-0.03, 1.0)
        ax.set_xticks([1, 10, 100, 1000, 10000])
        ax.set_xticklabels(["1", "10", "100", "1k", "10k"])
        ax.tick_params(labelsize=5.8, pad=1.5)
        ax.set_xlabel("payload (kbit, log)", fontsize=6.0, labelpad=1)
        ax.set_ylabel("detection recall (frame)", fontsize=6.0, labelpad=1)
        ax.set_title(f"all 14 modes, this frame ({ngt} objects)", fontsize=6.2, pad=2.2, color=INK)
        if row == 0:
            ax.legend(loc="upper left", fontsize=5.0, handletextpad=0.2, borderpad=0.3, labelspacing=0.2,
                      handlelength=1.0)
        y_top = gs[row, 0].get_position(fig).y1
        fig.text(0.006, y_top + 0.052, ex["title"] + f"  (TD frame {ex['image'][:-4]}, Phase 2 seed {ex['seed']})",
                 fontsize=6.8, fontweight="bold", color=INK, ha="left", va="bottom")
    fig.text(0.006, 0.006, "Boxes: labelled objects detected (green) or missed (red) by YOLO11s (conf $\\geq$ 0.25, "
             "IoU $\\geq$ 0.5) in a crop of the native frame; counts over the whole frame.\nOnly compliant images "
             "(on time, PSNR $\\geq$ Qmin) count towards recall.", fontsize=5.6, color=MUTED,
             ha="left", va="bottom")
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "fig_fd_qualitative.pdf", dpi=400)
    fig.savefig(out / "fig_fd_qualitative.png", dpi=220)
    (out / "fig_fd_qualitative.json").write_text(json.dumps(report, indent=1))
    for x in report:
        print(x)


if __name__ == "__main__":
    main()
