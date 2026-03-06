# **Spec (ImageNet \+ COCO only): Distillation Engineering Benchmark Suite**

## **1\) Goals**

1. Compare **teacher vs student** on:  
   * Representation alignment (no labels required)  
   * Transfer with minimal training (head-only)  
   * Efficiency  
2. Use only:  
   * ImageNet-1k (train/val)  
   * COCO 2017 (train/val)

No ADE20K / Cityscapes / NYU / VOC.

---

## **2\) Benchmark Matrix**

### **A) Representation Alignment (no training, no labels needed)**

Run on:

* **ImageNet-val**  
* **COCO-val** (images only)

Report:

* Summary cosine quantiles, MSE, CKA  
* Spatial token cosine quantiles, MSE  
* Retrieval Top1/MRR (batch)  
* HF cosine / energy ratio  
* Edge-alignment corr (Sobel)

(You already have most of this in your distill eval)

loop

model

### **B) ImageNet Linear Probe (minimal training)**

* Freeze backbone, train **linear head** on pooled features  
* Teacher vs student (same protocol)

### **C) COCO Detection Transfer (head-only training; backbone frozen)**

* Train standard detection heads on frozen backbone features.  
* Evaluate mAP on COCO-val.  
* Compare teacher-backed vs student-backed.

### **D) Efficiency**

* params, FLOPs, latency, throughput

---

## **3\) Repo layout (exact)**

bench/  
 README.md  
 common/  
   \_\_init\_\_.py  
   config.py  
   preprocess.py  
   io.py  
   model\_loaders.py  
   features.py  
   metrics.py  
   logging.py  
   timing.py  
 rep/  
   eval\_alignment.py  
 imagenet/  
   extract\_features.py  
   linear\_probe.py  
   knn\_eval.py  
 coco/  
   det\_train\_headonly.py  
   det\_eval.py  
   coco\_index.py  
 eff/  
   profile.py  
 reports/  
   summarize\_runs.py  
   make\_tables.py

All runnable as `python -m bench.<module> ...`.

---

## **4\) Standard preprocessing (must match distillation)**

### **Student preprocessing**

Match your distill loader:

* decode \-\> float \-\> resize bicubic antialias \-\> normalize ImageNet mean/std  
   data


## **5\) Model adapters**

Implement in `bench/common/model_loaders.py`

### **Student loader**

* Load timm ConvNeXt (same as your training runner) and apply checkpoint state dicts  
   launcher  
* Must expose feature taps:  
  * `f2`, `f3`  
  * pooled `f3_pool`  
  * `s_sum`, `s_tokens`, `s_sp` if using full DistillModel wrapper  
     model

### **Teacher loader**

* HF `AutoModel.from_pretrained(teacher_id, trust_remote_code=True)`  
* Must return `(t_summary, t_spatial_tokens)` like in caching  
   cache\_teacher\_outputs

---

## **6\) Module specs**

# **6.1 Representation alignment (ImageNet-val \+ COCO-val)**

### **Script: `bench/rep/eval_alignment.py`**

**Inputs**

* `--teacher_id`  
* `--student_ckpt`  
* `--dataset imagenet_val|coco_val|folder`  
* `--imagenet_root` (if imagenet\_val)  
* `--coco_root` \+ `--coco_split val2017` (if coco\_val)  
* `--size 416 --patch_size 16`  
* `--batch_size --num_workers --amp`  
* `--out_dir`

**Must compute**  
Reuse/port your alignment metrics:

* Summary:  
  * cosine mean/p05/p95  
  * MSE mean  
  * Linear CKA  
  * Batch retrieval Top1 \+ MRR  
     model  
* Spatial:  
  * token cosine mean/p05/p95  
  * spatial MSE  
  * style gram loss  
  * spatial energy ratio  
  * HF cosine \+ HF MSE \+ HF energy ratio  
  * edge alignment Pearson corr  
     loop

**Outputs**

* `metrics.json` (aggregates)  
* `metrics.jsonl` (per batch)  
* Optional TB images: spatial compare grids

**Acceptance**

* Runs on ImageNet-val and COCO-val images without annotation dependency.

---

# **6.2 ImageNet probing**

## **Script: `bench/imagenet/extract_features.py`**

Extract frozen features for teacher and student.

**Inputs**

* `--model teacher|student`  
* model args: `--teacher_id` or `--student_ckpt`  
* `--split train|val`  
* `--imagenet_root`  
* `--feature f3_pool|summary`  
* `--out_dir`

**Output**  
Sharded dumps:

* `out_dir/{split}/shard_00000.pt` with `{features, labels}`

## **Script: `bench/imagenet/linear_probe.py`**

**Head**

* `nn.Linear(C, 1000)`

**Train**

* SGD+momentum  
* 50 epochs default  
* report Top1/Top5

## **Script: `bench/imagenet/knn_eval.py` (optional)**

* cosine kNN on frozen features

---

# **6.3 COCO detection (head-only)**

You have two viable design choices:

### **Option 1 (Recommended, easiest to ship): TorchVision detection with custom backbone wrapper**

Use `torchvision.models.detection.FasterRCNN` with a backbone that returns feature maps in expected dict format (FPN optional).

**Key requirement:** head-only training means:

* `backbone.requires_grad_(False)`  
* train only:  
  * RPN head  
  * ROI heads  
  * box predictor

#### **Script: `bench/coco/det_train_headonly.py`**

**Inputs**

* `--model teacher|student`  
* `--teacher_id` or `--student_ckpt`  
* `--coco_root` (expects annotations in `annotations/instances_train2017.json`)  
* `--train_split train2017 --val_split val2017`  
* `--size 416`  
* `--epochs` (e.g. 12\)  
* `--batch_size` (e.g. 8\)  
* `--lr` (e.g. 0.02 for SGD)  
* `--out_dir`

**Backbone interface**

* Student backbone wrapper:  
  * uses f2 or f3 as feature map  
  * returns `{"0": fmap}` where fmap is (B,C,H,W)  
* Teacher backbone wrapper:  
  * convert `t_spatial_tokens` to (B,Dt,Ht,Wt)  
  * return `{"0": fmap}`

**Notes**

* Because both teacher/student yield different channel dims (Dt), the detector heads will be trained from scratch anyway. That’s fine.

#### **Script: `bench/coco/det_eval.py`**

Runs COCO eval (mAP, AP50, AP75, APS/APM/APL).  
Store results json and summary.

#### **Script: `bench/coco/coco_index.py`**

Utilities to build dataset, collate, category mapping.

**Acceptance**

* mAP computed on val2017  
* teacher-backed \> student-backed expected, but student should be close

---

# **6.4 Efficiency profiling**

### **Script: `bench/eff/profile.py`**

Measures:

* params  
* FLOPs (fvcore or ptflops)  
* latency (bs=1, 416\)  
* throughput (bs=32, 416\)

Outputs `efficiency.json`

---

## **7\) Output conventions (all scripts)**

Each run writes:

out\_dir/  
 run\_meta.json  
 metrics.json  
 metrics.jsonl   \# if training  
 tb/             \# optional  
 checkpoints/    \# for probe head/det head  
---

## **8\) Report aggregator**

### **Script: `bench/reports/summarize_runs.py`**

Given `--runs_root`, crawl subdirs, extract:

* alignment metrics on ImageNet \+ COCO  
* ImageNet probe top1/top5  
* COCO detection mAP  
* efficiency stats

Outputs:

* `summary.csv`  
* `summary.md` (paper-ready tables)

---

## **9\) “Pretrained heads without finetuning” policy (must be explicit)**

Agent must document in `bench/README.md`:

* A downstream head trained on a different backbone learns a **specific feature coordinate system**.  
* Teacher/student features can be aligned to each other but not necessarily to that head.  
* Therefore:  
  * **No-training evaluation** is only valid for **alignment metrics**.  
  * Task evaluation uses **head-only training** (frozen backbone), which is standard.

---

## **10\) Default commands (to include verbatim in README)**

### **Alignment: ImageNet-val**

python \-m bench.rep.eval\_alignment \\  
 \--teacher\_id nvidia/C-RADIOv4-H \\  
 \--student\_ckpt /path/student.pth \\  
 \--dataset imagenet\_val \\  
 \--imagenet\_root /data/imagenet \\  
 \--size 416 \--patch\_size 16 \\  
 \--batch\_size 64 \--num\_workers 8 \--amp \\  
 \--out\_dir /exp/bench/alignment\_imnet

### **Alignment: COCO-val (images only)**

python \-m bench.rep.eval\_alignment \\  
 \--teacher\_id nvidia/C-RADIOv4-H \\  
 \--student\_ckpt /path/student.pth \\  
 \--dataset coco\_val \\  
 \--coco\_root /data/coco \\  
 \--size 416 \--patch\_size 16 \\  
 \--batch\_size 64 \--num\_workers 8 \--amp \\  
 \--out\_dir /exp/bench/alignment\_coco

### **ImageNet features \+ linear probe**

python \-m bench.imagenet.extract\_features \\  
 \--model student \--student\_ckpt /path/student.pth \\  
 \--split train \--imagenet\_root /data/imagenet \\  
 \--feature f3\_pool \--batch\_size 128 \--num\_workers 8 \--amp \\  
 \--out\_dir /exp/bench/features\_student

python \-m bench.imagenet.extract\_features \\  
 \--model student \--student\_ckpt /path/student.pth \\  
 \--split val \--imagenet\_root /data/imagenet \\  
 \--feature f3\_pool \--batch\_size 128 \--num\_workers 8 \--amp \\  
 \--out\_dir /exp/bench/features\_student

python \-m bench.imagenet.linear\_probe \\  
 \--train\_features\_dir /exp/bench/features\_student/train \\  
 \--val\_features\_dir /exp/bench/features\_student/val \\  
 \--epochs 50 \--lr 0.1 \--batch\_size 4096 \\  
 \--out\_dir /exp/bench/linprobe\_student

### **COCO detection head-only**

python \-m bench.coco.det\_train\_headonly \\  
 \--model student \--student\_ckpt /path/student.pth \\  
 \--coco\_root /data/coco \\  
 \--size 416 \--epochs 12 \--batch\_size 8 \--lr 0.02 \\  
 \--out\_dir /exp/bench/coco\_det\_student

### **Efficiency**

python \-m bench.eff.profile \\  
 \--teacher\_id nvidia/C-RADIOv4-H \\  
 \--student\_ckpt /path/student.pth \\  
 \--size 416 \--batch\_size 1 \\  
 \--out\_dir /exp/bench/eff

### **Summarize**

python \-m bench.reports.summarize\_runs \\  
 \--runs\_root /exp/bench \\  
 \--out\_csv /exp/bench/summary.csv \\  
 \--out\_md /exp/bench/summary.md
