import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
import os

from model import MelodyCRNN
from dataset import MelodyDataset
from config import train_config as config

def calculate_loss(chroma_logits, octave_logits, voicing_logits,
                   y_chroma, y_octave, y_voicing, loss_co, loss_v):
    B, T, C_chroma = chroma_logits.shape
    loss_chroma = loss_co(
        chroma_logits.view(B * T, C_chroma), 
        y_chroma.view(B * T)
    )
    
    _, _, C_octave = octave_logits.shape
    loss_octave = loss_co(
        octave_logits.view(B * T, C_octave),
        y_octave.view(B * T)
    )
    
    _, _, C_voicing = voicing_logits.shape
    loss_voicing = loss_v(
        voicing_logits.view(B * T, C_voicing),
        y_voicing.view(B * T)
    )
    
    # equal weight between classifications (this is arbitrary)
    loss = loss_chroma + loss_octave + config['voicing_loss_weight'] * loss_voicing
    
    return loss, loss_chroma, loss_octave, loss_voicing

def calculate_accuracies(chroma_logits, octave_logits, voicing_logits,
                         y_chroma, y_octave, y_voicing):
    
    # predict classes with maximum logits for time frame
    chroma_preds = torch.argmax(chroma_logits, dim=2)
    octave_preds = torch.argmax(octave_logits, dim=2)
    voicing_preds = torch.argmax(voicing_logits, dim=2)

    frames = y_chroma.numel()
    
    c_cor = (chroma_preds == y_chroma).sum().item()
    o_cor = (octave_preds == y_octave).sum().item()
    v_cor = (voicing_preds == y_voicing).sum().item()
    v_rec = ((voicing_preds == 1) & (y_voicing == 1)).sum().item()
    v_far = ((voicing_preds == 1) & (y_voicing == 0)).sum().item()
    v_true = (y_voicing == 1).sum().item()
    v_false = (y_voicing == 0).sum().item()
    
    return c_cor, o_cor, v_cor, v_rec, v_far, v_true, v_false, frames


def train_epoch(model, dataloader, optimizer, loss_co, loss_v, device):
    model.train()
    total_loss = 0.0
    
    pbar = tqdm(dataloader, desc="Training", unit="batch")
    for batch in pbar:
        x, y_chroma, y_octave, y_voicing = [b.to(device) for b in batch]
        optimizer.zero_grad()
        chroma_logits, octave_logits, voicing_logits = model(x)
        loss, loss_chroma, loss_octave, loss_voicing = calculate_loss(
            chroma_logits, octave_logits, voicing_logits,
            y_chroma, y_octave, y_voicing, loss_co, loss_v
        )
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f} (c:{loss_chroma.item():.2f} o:{loss_octave.item():.2f} v:{loss_voicing.item():.2f})")
        
    return total_loss / len(dataloader)

def validate_epoch(model, dataloader, loss_co, loss_v, device):
    model.eval()
    total_loss = 0.0
    
    chroma_correct, octave_correct, voicing_correct = 0, 0, 0
    voicing_recall, voicing_falarm = 0, 0
    voicing_true, voicing_false = 0, 0
    total_frames = 0
    
    pbar = tqdm(dataloader, desc="Validating", unit="batch")
    with torch.no_grad():
        for batch in pbar:
            x, y_chroma, y_octave, y_voicing = [b.to(device) for b in batch]
            chroma_logits, octave_logits, voicing_logits = model(x)
            loss, _, _, _ = calculate_loss(
                chroma_logits, octave_logits, voicing_logits,
                y_chroma, y_octave, y_voicing, loss_co, loss_v
            )
            total_loss += loss.item()
            
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

    avg_loss = total_loss / len(dataloader)
    acc_c = (chroma_correct / total_frames) * 100
    acc_o = (octave_correct / total_frames) * 100
    acc_v = (voicing_correct / total_frames) * 100
    rec_v = (voicing_recall / voicing_true) * 100
    far_v = (voicing_falarm / voicing_false) * 100
    acc = (acc_c + acc_o + acc_v) / 3
    
    return avg_loss, acc, acc_c, acc_o, rec_v, far_v

def main():
    now = datetime.now()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset = MelodyDataset(config['train_set_path'])
    val_dataset = MelodyDataset(config['val_set_path'])
    
    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        num_workers=min(os.cpu_count() // 2, 4),
        pin_memory=True
    )
    val_loader = DataLoader(
        dataset=val_dataset, 
        batch_size=config['batch_size'],
        shuffle=False, 
        num_workers=min(os.cpu_count() // 2, 4),
        pin_memory=True
    )

    model = MelodyCRNN().to(device)
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )

    # weighed loss for voiced/unvoiced decision to prevent overfit on unvoiced
    loss_co = nn.CrossEntropyLoss()
    loss_v = nn.CrossEntropyLoss(
        weight=torch.Tensor([config['unvoiced_weight'], config['voiced_weight']]).to(device)
    )

    model_dir = Path(config['model_dir'])
    model_dir.mkdir(parents=True, exist_ok=True)
    timestamp = f'{now.year}{now.month:02d}{now.day:02d}-{now.hour:02d}{now.minute:02d}{now.second:02d}'
    model_path_start = f'{config['model_dir']}/melody_crnn_{timestamp}'

    best_acc = 0.0
    best_loss = float('inf')
    prev_acc, prev_loss, prev_both = None, None, None
    for epoch in range(config['epochs']):
        print(f"\nepoch {epoch+1}/{config['epochs']}")
        
        train_loss = train_epoch(model, train_loader, optimizer, loss_co, loss_v, device)
        val_loss, acc, acc_c, acc_o, rec_v, far_v = validate_epoch(model, val_loader, loss_co, loss_v, device)
        
        print(f"| train loss: {train_loss:.4f}")
        print(f"| valid loss: {val_loss:.4f}")
        print(f"|  overall acc:              {acc:.2f}%")
        print(f"|  chroma frame-wise acc:    {acc_c:.2f}%")
        print(f"|  octave frame-wise acc:    {acc_o:.2f}%")
        print(f"|  voicing recall rate:      {rec_v:.2f}%")
        print(f"|  voicing false-alarm rate: {far_v:.2f}%")

        # welcome to a world of filename slop
        quality = ''
        if acc > best_acc and val_loss < best_loss:
            if prev_both != None:
                os.remove(prev_both)
        if acc > best_acc:
            if prev_acc != None:
                os.remove(prev_acc)
                prev_acc = None
            quality += f'__acc-{acc:.2f}'
        if val_loss < best_loss:
            if prev_loss != None:
                os.remove(prev_loss)
                prev_loss = None
            quality += f'__val-{val_loss:.4f}'
        
        if quality != '':
            model_path = model_path_start + quality + '.pt'
            torch.save(model.state_dict(), model_path)
            print(f"| saved to {model_path}")
        
        if acc > best_acc and val_loss < best_loss:
            best_acc = acc
            best_loss = val_loss
            prev_both = model_path
        elif acc > best_acc:
            if prev_both != None:
                prev_loss = model_path_start + f'__val-{val_loss:.4f}.pt'
                os.rename(prev_both, prev_loss)
                prev_both = None
            best_acc = acc
            prev_acc = model_path
        elif val_loss < best_loss:
            if prev_both != None:
                prev_acc = model_path_start + f'__acc-{best_acc:.2f}.pt'
                os.rename(prev_both, prev_acc)
                prev_both = None
            best_loss = val_loss
            prev_loss = model_path

    print(f"best validation loss: {best_loss:.4f}")
    print(f"best validation acc:  {best_acc:.2f}")

if __name__ == "__main__":
    main()