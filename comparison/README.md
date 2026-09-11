# Model Comparison

This folder runs inference-only PSNR/SSIM and resource comparisons for T2T-VICL, MAE-VQGAN, Painter, Prompt-Diffusion, InstructDiffusion, VisualCloze, and LVM.
Competitors use their official source and released/default inference settings.

Run everything from the root.

## Prepare and check the 11 x 100 split

```bash
python -m comparison.prepare_same_task_eval
python -m comparison.prepare_same_task_eval --check
python -m comparison.datasets --manifest comparison/competitor_tasks.json --data-root data/tasks --check-images -1 --summary-only
python -m unittest discover -s comparison/tests -p 'test_*.py'
```

The generated competitor JSON is `data/dataset/eval_dataset_same_task.json`. Each same-task demonstration differs from its query. Inpainting is excluded.

## Quality

Set the downloaded checkpoint paths, then run:

```bash
export VICL_WEIGHTS="$PWD/weights"
export PAINTER_CKPT="$VICL_WEIGHTS/Painter/painter_vit_large.pth"
export MAE_CKPT="$VICL_WEIGHTS/MAE-VQGAN/checkpoint-3400.pth"
export PROMPT_DIFFUSION_CKPT="$PWD/third_party/Prompt-Diffusion/ckpts/network-step=04999.ckpt"
export INSTRUCT_CKPT="$PWD/third_party/InstructDiffusion/checkpoints/v1-5-pruned-emaonly-adaption-task.ckpt"
test -f "$PAINTER_CKPT" && test -f "$MAE_CKPT" && test -f "$VICL_WEIGHTS/MAE-VQGAN/model.yaml" && test -f "$VICL_WEIGHTS/MAE-VQGAN/last.ckpt" && test -f "$PROMPT_DIFFUSION_CKPT" && test -f "$INSTRUCT_CKPT"

python -m comparison.quality --adapter painter --conditions official --task-manifest comparison/competitor_tasks.json --checkpoint "$PAINTER_CKPT" --max-samples -1 --resume --output-dir comparison/outputs/quality/painter

python -m comparison.quality --adapter mae-vqgan --conditions official --task-manifest comparison/competitor_tasks.json --checkpoint "$MAE_CKPT" --max-samples -1 --resume --output-dir comparison/outputs/quality/mae_vqgan

python -m comparison.quality --adapter prompt-diffusion --conditions official --task-manifest comparison/competitor_tasks.json --checkpoint "$PROMPT_DIFFUSION_CKPT" --max-samples -1 --resume --output-dir comparison/outputs/quality/prompt_diffusion

python -m comparison.quality --adapter instruct-diffusion --conditions official --task-manifest comparison/competitor_tasks.json --checkpoint "$INSTRUCT_CKPT" --max-samples -1 --resume --output-dir comparison/outputs/quality/instruct_diffusion

python -m comparison.quality --adapter visualcloze --conditions official --task-manifest comparison/competitor_tasks.json --checkpoint "$VICL_WEIGHTS/VisualCloze/visualcloze-384-lora.pth" --max-samples -1 --resume --output-dir comparison/outputs/quality/visualcloze

python -m comparison.quality --adapter lvm --conditions official --task-manifest comparison/competitor_tasks.json --checkpoint "$weights_root/LVM/LVM_ckpts" --lvm-vqvae "$weights_root/LVM/vqvae_ckpts" --max-samples -1 --resume --output-dir comparison/outputs/quality/lvm
```

## Resources

Run the four T2T-VICL backbones separately on the same GPU. This uses the 26 original Task A-to-B directions in `data/dataset/eval_dataset.json`:

```bash
export VICL_WEIGHTS="$PWD/weights"
export PROMPT_CKPT="$PWD/Qwen3-VL/qwen-vl-finetune/output/checkpoint-4875"

python -m comparison.preflight --require-dispatch-flops
python -m comparison.suite --adapter t2t-qwen --conditions ours --task-manifest comparison/t2t_tasks.json --model-id "$VICL_WEIGHTS/Qwen-Image-Edit-2511" --prompt-checkpoint "$PROMPT_CKPT" --max-samples 10 --warmup 5 --resolution 448 --profile-flops --output-dir comparison/outputs/resources/t2t_qwen
python -m comparison.suite --adapter t2t-flux2 --conditions ours --task-manifest comparison/t2t_tasks.json --model-id "$VICL_WEIGHTS/FLUX.2-dev-bnb-4bit" --prompt-checkpoint "$PROMPT_CKPT" --max-samples 10 --warmup 5 --resolution 448 --profile-flops --output-dir comparison/outputs/resources/t2t_flux2
python -m comparison.suite --adapter t2t-omnigen2 --conditions ours --task-manifest comparison/t2t_tasks.json --model-id "$VICL_WEIGHTS/OmniGen2" --prompt-checkpoint "$PROMPT_CKPT" --max-samples 10 --warmup 5 --resolution 448 --profile-flops --output-dir comparison/outputs/resources/t2t_omnigen2
python -m comparison.suite --adapter t2t-firered --conditions ours --task-manifest comparison/t2t_tasks.json --model-id "$VICL_WEIGHTS/FireRed-Image-Edit-1.1" --prompt-checkpoint "$PROMPT_CKPT" --max-samples 10 --warmup 5 --resolution 448 --profile-flops --output-dir comparison/outputs/resources/t2t_firered
```

## Tables and figure

```bash
python -m comparison.quality_report comparison/outputs/quality/*/*_quality_latest.json --output-dir comparison/outputs/paper_tables

python -m comparison.qualitative_grid comparison/outputs/quality/*/*_quality_latest.json --rebuttal-first T2T-VICL=data/output/supplementary/gemini/qwen --output ../latex/fig/competitor_comparison.png

python -m comparison.report comparison/outputs/resources/*/*_suite_latest.json --output-dir comparison/outputs/paper_tables/resources
```

The new competitor rows require 100 completed queries per task. The figure command uses the saved T2T-VICL outputs.
