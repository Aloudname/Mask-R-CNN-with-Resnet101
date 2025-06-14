import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from torchvision.models.detection import MaskRCNN
from torchvision.ops import RoIAlign, MultiScaleRoIAlign
from torchvision.models.detection.rpn import RPNHead, AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone


def res34(pretrained = True, fine_tune = True, num_classes = 4):
    """
    num_classes: number of classifications.
    """
    if pretrained:
        print('[INFO]: Loading pre-trained weights...')  
    elif not pretrained:
        print('[INFO]: Not loading pre-trained weights...')  
    model = models.resnet34(pretrained=pretrained)
    if fine_tune:
        print('[INFO]: Fine-tuning all layers...')
        for param in model.parameters():
            param.requires_grad = False

        # paragraghs below are selectable "True" for better performance:
        # defreeze the last block.
        for param in model.layer4.parameters():
            param.requires_grad = False
        # defreeze the 3rd block.
        for param in model.layer3.parameters():
            param.requires_grad = False
        # defreeze the 2rd block.    
        for param in model.layer2.parameters():
            param.requires_grad = False
        # defreeze the 1st block.
        for param in model.layer1.parameters():
            param.requires_grad = False

    elif not fine_tune:
        print('[INFO]: Freezing hidden layers...')
        for params in model.parameters():
            params.requires_grad = False

    # model.fc = nn.Linear(512, num_classes)
    model.fc = nn.Sequential(
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, num_classes))
    return model

def res101(pretrained=True, fine_tune=True, num_classes=4):
    """
    num_classes: number of classifications.
    """
    if pretrained:
        print('[INFO]: Loading pre-trained weights...')
    elif not pretrained:
        print('[INFO]: Not loading pre-trained weights...')
    model = models.resnet101(pretrained=pretrained)
    if fine_tune:
        print('[INFO]: Fine-tuning all layers...')
        # freeze all the residual layers.
        for param in model.parameters():
            param.requires_grad = False

        # paragraghs below are selectable "True" for better performance:
        # defreeze the last block.
        for param in model.layer4.parameters():
            param.requires_grad = True
        # defreeze the 3rd block.
        for param in model.layer3.parameters():
            param.requires_grad = True
        # defreeze the 2nd block.
        for param in model.layer2.parameters():
            param.requires_grad = True
        # defreeze the 1st block.
        for param in model.layer1.parameters():
            param.requires_grad = False
            
    elif not fine_tune:
        print('[INFO]: Freezing hidden layers...')
        for param in model.parameters():
            param.requires_grad = False
    
    model.fc = nn.Sequential(
        nn.Linear(2048, 1024),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(1024, num_classes))
    return model

def mask_rcnn(pretrained=True, fine_tune=True, num_classes=4):
    """
    Mask R-CNN based on Resnet.
    
    params:
    - pretrained: If pretrained model is applied.
    - fine_tune: If defreeze some residual layers for fine-tuning.
    - num_classes: Used for segmentation. Counts of pathelogical classes and background, thus classes + 1.
    """
    
    # ResNet101-FPN backbone with default defreeze layer 4,3,2.
    # Possible values = {'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
    # 'resnext50_32x4d', 'resnext101_32x8d', 'wide_resnet50_2', 'wide_resnet101_2'}
    backbone_name = 'resnet101'
    backbone = resnet_fpn_backbone(
        backbone_name=backbone_name,
        pretrained=pretrained,
        trainable_layers=3
    )

    if pretrained:
        print(f'[INFO]: Loading pre-trained {backbone_name}-FPN weights...')
    else:
        print('[INFO]: Training from scratch...')

    # anchor generator.
    anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    anchor_generator = AnchorGenerator(
        sizes=anchor_sizes,
        aspect_ratios=aspect_ratios
    )

    # Compose into mask R-CNN.
    model = MaskRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_detections_per_img=200  # maximum box within an image. 200 defaultly.
    )

    # Fine-tuning adjust within backbone resnet101.
    if fine_tune:
        print('[INFO]: Fine-tuning selected layers...')
        # freeze all parameters.
        for param in model.parameters():
            param.requires_grad = False

        # defreeze FPN layer.
        for param in backbone.fpn.parameters():
            param.requires_grad = True
            
        # defreeze 4th layer.
        for param in backbone.body.layer4.parameters():
            param.requires_grad = True
        # defreeze 3rd layer.
        for param in backbone.body.layer3.parameters():
            param.requires_grad = False
        # defreeze 2nd layer.
        for param in backbone.body.layer2.parameters():
            param.requires_grad = False
        # defreeze 1st layer.
        for param in backbone.body.layer1.parameters():
            param.requires_grad = False
            
        # defreeze RPN layer.
        for param in model.rpn.parameters():
            param.requires_grad = True
            
        # defreeze ROI Heads.
        for param in model.roi_heads.parameters():
            param.requires_grad = True
            
    else:
        print('[INFO]: Freezing all backbone layers...')
        # If not, only ROI heads are trained.
        for param in backbone.parameters():
            param.requires_grad = False

    # ROI heads defination.
    # replace ROI heads.
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # replace mask code heads.
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, 256,  # dim of hidden layer.
        num_classes
    )
    return model

class FastRCNNPredictor(nn.Module):
    """Customized RCNNPredictor (ROI heads) implementation align with the function above."""
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)

    def forward(self, x):
        return self.cls_score(x), self.bbox_pred(x)

class MaskRCNNPredictor(nn.Sequential):
    """Customized RCNN mask predictor with normalization."""
    def __init__(self, in_channels, dim_reduced, num_classes):
        super().__init__(
            nn.Conv2d(in_channels, dim_reduced, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim_reduced),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2),
            nn.Conv2d(dim_reduced, dim_reduced, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim_reduced),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_reduced, num_classes, 1)
        )

class KANLayer(nn.Module):
    """A single KAN layer derived from nn.Module."""
    def __init__(self, input_dim, output_dim, num_basis=5):
        """
        num_basis : Number of basis functions, reflects the complexity of a layer.
        weights: Learnable weights of basis functions.
        basis_functions: Pattern of the basis function. Linear defaultly.
        """
        super().__init__()
        self.num_basis = num_basis
        self.weights = nn.Parameter(torch.randn(output_dim, input_dim, num_basis))
        self.basis_functions = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, 16),
                nn.SiLU(),
                nn.Linear(16, 1)
            ) for _ in range(num_basis)
        ])

    def forward(self, x):
        # x shape: [batch, input_dim]
        batch_size, input_dim = x.shape
        outputs = []
        for out_dim in range(self.weights.shape[0]):
            out = 0.0
            for in_dim in range(input_dim):
                basis_input = x[:, in_dim].unsqueeze(1)  # [batch, 1]
                basis_output = self.basis_functions[in_dim](basis_input)  # [batch, 1]
                weight = self.weights[out_dim, in_dim]  # [num_basis]
                out += torch.sum(weight * basis_output, dim=-1)  # [batch]
            outputs.append(out)
        return torch.stack(outputs, dim = 1)  # [batch, output_dim]

class KAN(nn.Module):
    """KAN module for 512*512 images classification."""
    def __init__(self, in_channels=3, num_classes=2):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.MaxPool2d(2),  # 256x256
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(2),  # 128x128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.MaxPool2d(2),  # 64x64
            nn.Flatten()  # output dim: 128*64*64 = 524288
            )
        
        self.kan_layers = nn.Sequential(
            KANLayer(input_dim=524288, output_dim=256),
            nn.SiLU(),
            KANLayer(input_dim=256, output_dim=64),
            nn.SiLU(),
            KANLayer(input_dim=64, output_dim=16))
        
        self.classifier = nn.Linear(16, num_classes)

    def forward(self, x):
        # input x: [batch, 3, 512, 512]
        features = self.feature_extractor(x)  # [batch, 524288]
        kan_out = self.kan_layers(features)   # [batch, 16]
        logits = self.classifier(kan_out)     # [batch, 2]
        return logits

if __name__ == "__main__":

    model = mask_rcnn(pretrained = True, fine_tune = True, num_classes = 4)
    # inspect trainable params.
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Trainable layer: {name}")
