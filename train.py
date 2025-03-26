import torch
import argparse
import torch.nn as nn
from tqdm.auto import tqdm
import torch.optim as optim
import vision_reference_detection as utils

from model import mask_rcnn
from utils import save_model, save_loss_curve
from BoxMode import train_one_epoch, evaluate
from datasets import train_loader, test_loader
from torchvision.ops.boxes import box_convert
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone


# Activate by python train.py --epochs 20 --lr 0.005 --momentum 0.9 --weight-decay 0.0005
parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epochs', type=int, default=20,
                    help='number of epochs to train network for')
parser.add_argument('--lr', type=float, default=0.005,
                    help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='momentum for SGD')
parser.add_argument('--weight-decay', type=float, default=0.0005,
                    help='weight decay')
args = parser.parse_args()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"\nComputation device: {device}")

model = mask_rcnn(
    pretrained=True,
    fine_tune=True,
    num_classes=14
).to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.SGD(
    params,
    lr=args.lr,
    momentum=args.momentum,
    weight_decay=args.weight_decay
)

lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=3,
    gamma=0.1
)

train_loss_history = []
val_map_history = []

print(f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters")

for epoch in range(args.epochs):
    print(f"\n[Epoch {epoch+1}/{args.epochs}]")
    
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f'Epoch: [{epoch+1}/{args.epochs}]'
    images, targets = [], []
    
    for image, target in tqdm(train_loader, desc="Training"):
        target['boxes'] = box_convert(
            boxes=torch.tensor(target['boxes'].clone().detach()),
            in_fmt='xywh', out_fmt='xyxy')
        images.append(image.to(device))
        targets.append({k: v.to(device) for k, v in target.items()})
        
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        metric_logger.update(loss=losses, **loss_dict)
    
    lr_scheduler.step()
    train_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    train_loss_history.append(train_stats['loss'])
    
    print("\nValidating...")
    coco_evaluator = evaluate(model, test_loader, device=device)
    val_stats = coco_evaluator.coco_eval['bbox'].stats
    val_map_history.append(val_stats[0])  # record: mAP@[0.5:0.95]
    
    print(f"\nTrain Loss: {train_stats['loss']:.4f}")
    print(f"Validation mAP: {val_stats[0]:.4f}")
    print('-' * 80)

torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch,
}, 'mask_rcnn.pth')

save_model(model, optimizer, epoch, args)
save_loss_curve(train_loss_history, val_map_history)
print("TRAINING COMPLETE!")
