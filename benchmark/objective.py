"""Original trainer.py objective, preserving reductions, class order and schedule."""
import torch
from torch import nn
from utils import ContrastiveLoss, DiceLoss

LOSS_SPEC = {
    "source": "trainer.py:trainer_Myops; utils.py:DiceLoss,ContrastiveLoss (original snapshot in reference/)",
    "label_order": "legacy: background,myocardium,scar,edema",
    "segmentation": "0.2 CE + 0.8 original Dice; decoder losses summed",
    "dice": "batch/spatial reduction, squared denominator, smooth=1e-5, class weights=[0.5,1,1,1], divide by 4",
    "contrast": "original anchor sampling, modality average, temperature=0.07; ignores=([2,3],[0],[0])",
    "text_anchor_adapter": "repeat four class text anchors twice to reproduce original two-GPU DataParallel gather and prompt_num=2",
    "early": "decoder_loss + contrast_w*contrast_loss",
    "late": "main_loss + 0.5*decoder_loss + 0.5*contrast_w*contrast_loss",
    "schedule": "contrast iff epoch_index > start_contrast_epoch; late iff epoch_index >= region_fusion_start_epoch",
}


class OriginalObjective(nn.Module):
    def __init__(self, contrast_w=0.1, contrast_sample_num=10, start_contrast_epoch=20,
                 region_fusion_start_epoch=100):
        super().__init__()
        self.ce, self.dice, self.contrast = nn.CrossEntropyLoss(), DiceLoss(4), ContrastiveLoss()
        self.contrast_w, self.sample_num = contrast_w, contrast_sample_num
        self.start_contrast, self.start_fusion = start_contrast_epoch, region_fusion_start_epoch

    def do_contrast(self, epoch):
        return epoch > self.start_contrast

    def forward(self, outputs, target, epoch):
        out_pre, dec_seg, feature_lists, text_list = outputs
        contrast = 0
        if self.do_contrast(epoch):
            # Original train.py defaults to two GPUs. DataParallel gathers each
            # four-row text tensor twice, matching ContrastiveLoss(prompt_num=2).
            # On one device reproduce those eight anchors, keeping the loss intact.
            text_list = [text.repeat(2, 1) if text.shape[0] == 4 else text for text in text_list]
            if any(text.shape[0] != 8 for text in text_list):
                raise ValueError("Original contrastive recipe requires two copies of the four legacy class anchors")
            for features, ignore in zip(feature_lists, ([2, 3], [0], [0])):
                contrast += self.contrast(features, target, text_list, ignore, sample_num=self.sample_num)
            contrast = contrast / len(feature_lists)
        main = 0.2 * self.ce(out_pre, target) + 0.8 * self.dice(out_pre, target, softmax=True)
        dec_ce, dec_dice = out_pre.new_zeros(1), out_pre.new_zeros(1)
        for pred in dec_seg:
            dec_ce += self.ce(pred, target)
            dec_dice += self.dice(pred, target, softmax=True)
        dec = 0.2 * dec_ce + 0.8 * dec_dice
        if epoch < self.start_fusion:
            return main * 0.0 + dec + contrast * self.contrast_w
        return main + 0.5 * dec + 0.5 * contrast * self.contrast_w
