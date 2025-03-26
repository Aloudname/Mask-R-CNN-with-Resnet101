import os
import torch
import matplotlib.pyplot as plt
from datetime import datetime

def save_model(model, optimizer, epoch, args, save_dir='E:\\kq\\outputs\\checkpoints'):
    """
    All the checkpoint save jobs.
    params:
        model: model.
        optimizer: optimizer.
        epoch: current epoch.
        args: commandline arguments and hyperarguments.
        save_dir: save directory, defaultly named 'checkpoints'.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # filename with time stamps.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = os.path.join(save_dir, f'maskrcnn_epoch{epoch}_{timestamp}.pth')
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'args': vars(args)  # convert the commandline arguments.
    }, checkpoint_path)
    
    print(f"\nModel checkpoint saved to: {checkpoint_path}")


def save_loss_curve(train_loss_history, val_map_history, save_dir='E:\\kq\\outputs\\plots'):
    """
    All the plotting jobs.
    params:
        train_loss_history: test loss list with time.
        val_map_history: test mAP list with time.
        save_dir: save directory, defaultly named 'plots'.
    """

    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(10, 5))
    
    # loss curve.
    plt.subplot(1, 2, 1)
    plt.plot(train_loss_history, label='Train Loss', color='blue', linestyle='-')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # mAP curve.
    plt.subplot(1, 2, 2)
    plt.plot(val_map_history, label='Validation mAP', color='red', linestyle='-')
    plt.xlabel('Epoch')
    plt.ylabel('mAP@[0.5:0.95]')
    plt.title('Validation mAP Curve')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # adjust the layout.
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(save_dir, f'loss_map_curve_{timestamp}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Loss and mAP curves saved to: {save_path}")
