import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
import sys, os

from model import MelodyCRNN
from dataset import MelodyDataset
from train import calculate_accuracies
from config import train_config as config

if len(sys.argv) != 2:
    print('Usage: python test_eval.py MODEL_PATH')
    print(len(sys.argv))
    exit()

model_path = Path(sys.argv[1])
if not model_path.exists():
    print(f"{model_path} does not exist.")
    exit()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MelodyCRNN().to(device)
model.load_state_dict(
    torch.load(model_path, map_location=device, weights_only=True)
)

model.eval()
test_loader = DataLoader(
    dataset=MelodyDataset(config['test_set_path']), 
    batch_size=config['batch_size'], 
    shuffle=True, 
    num_workers=min(os.cpu_count() // 2, 4),
    pin_memory=True
)

chroma_correct, octave_correct, voicing_correct = 0, 0, 0
voicing_recall, voicing_falarm = 0, 0
voicing_true, voicing_false = 0, 0
total_frames = 0

pbar = tqdm(test_loader, desc="Validating", unit="batch")
with torch.no_grad():
    for batch in pbar:
        x, y_chroma, y_octave, y_voicing = [b.to(device) for b in batch]
        chroma_logits, octave_logits, voicing_logits = model(x)
        c_cor, o_cor, v_cor, v_rec, v_far, v_true, v_false, frames = calculate_accuracies(
            chroma_logits, octave_logits, voicing_logits,
            y_chroma, y_octave, y_voicing
        )
        chroma_correct += c_cor
        octave_correct += o_cor
        voicing_correct += v_cor
        voicing_recall += v_rec
        voicing_falarm += v_far
        voicing_true += v_true
        voicing_false += v_false

        total_frames += frames

acc_c = (chroma_correct / total_frames) * 100
acc_o = (octave_correct / total_frames) * 100
acc_v = (voicing_correct / total_frames) * 100
rec_v = (voicing_recall / voicing_true) * 100
far_v = (voicing_falarm / voicing_false) * 100
acc = (acc_c + acc_o + acc_v) / 3

print(f"ADC2004 Test Dataset Evaluation Results")
print(f"| overall acc:              {acc:.2f}%")
print(f"| chroma frame-wise acc:    {acc_c:.2f}%")
print(f"| octave frame-wise acc:    {acc_o:.2f}%")
print(f"| voicing recall rate:      {rec_v:.2f}%")
print(f"| voicing false-alarm rate: {far_v:.2f}%")
