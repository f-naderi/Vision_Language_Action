```markdown
# Vision-Language-Action Learning

A lightweight Vision-Language-Action (VLA) project that learns to predict continuous robot actions from visual observations, natural-language instructions, and robot state.

## Features

- Multimodal input: RGB image + language instruction + robot state
- 7D continuous robot action prediction
- Pretrained ResNet-18 visual encoder
- Pretrained DistilBERT language encoder
- Partial fine-tuning of the vision-language backbone
- Test-set evaluation using MSE and MAE


## Project Structure

Vision_Language_Action/
├── libero_spatial_no_noops/       # LIBERO dataset
│
├── pretrained/                    # Local pretrained DistilBERT weights
│
├── src/
│   ├── dataset.py                 # LIBERO dataset loading and preprocessing
│   ├── model.py                   # MiniVLA multimodal architecture
│   ├── train.py                   # Training and validation pipeline
│   ├── evaluate.py                # Test-set evaluation
│   └── inference.py               # Single-sample inference
│
├── results/
│   ├── best_minivla.pt            # Best validation checkpoint
│   └── training_curves.png        # Training/validation loss curves
│
├── VLA.ipynb                      # Interactive Jupyter notebook
└── requirements.txt               # Python dependencies


## Installation

```bash
git clone https://github.com/f-naderi/Vision_Language_Action.git
cd Vision_Language_Action
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Model Architecture:


The MiniVLA model receives three complementary inputs:
```text
RGB Image ──────────► ResNet-18 ───────┐
                                       │
Language Instruction ─► DistilBERT ────┼──► Multimodal Fusion ─► Action Head ─► 7D Action
                                       │
Robot State ─────────► State Encoder ──┘
```

* ResNet-18:   Frozen backbone + trainable final projection
* DistilBERT:  The last two Transformer layers are trainable, while the remaining layers are frozen
* State/Fusion/Action Head: Fully trainable


## Training Details

* Dataset: LIBERO Spatial
* Train/Val/Test: 345 episodes / 43 episodes / 44 episodes
* Original Image Size: 256×256
* Model Input Size: 224×224
* Total Parameters: 78.7M
* Trainable Parameters: 15.3M (19.48%)
* Optimizer: AdamW
* Learning Rate: 1e-4
* Weight Decay: 1e-4
* Batch Size: 64
* Epochs: 50
* Mixed Precision: FP16


## Test Results

* Test MSE: 0.038973
* Test MAE: 0.094021
   
## Requirements

* Python 3.10+
* PyTorch
* Torchvision
* Transformers
* NumPy
* Pillow
* Matplotlib
* tqdm
* tfrecord
* Jupyter Notebook

## References

* LIBERO: Benchmark for Learning in Robotics
* OpenVLA: An Open-Source Vision-Language-Action Model
* He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition.
* Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). DistilBERT, a distilled version of BERT.

```

