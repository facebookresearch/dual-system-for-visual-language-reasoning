# dual-system-for-visual-language-reasoning

This is a Pytorch implementation for Peifeng's internship project.

Code folders:

(1) `system1-vision`: Fine-tuning and inference with the vision module.

(2) `system2-lm`: Prompting LM for solving downstream tasks.

## Dependencies

- Python >= 3.6
- PyTorch == 1.12.1
- transformers == 4.29.2
- fairscale == 0.4.6

## Fine-tuning a vision module for visual information extraction

```bash
cd system1-vision
sbatch ./scripts/finetune_deplot.sh 
```
After training, the checkpoint of the vision module is saved to `$VISION_CHECKPOINT='HOME_DIR/outputs/checkpoint'` for later use.

## Prompting LM for downstream tasks

The scripts for different tasks are stored at `system2-lm/scripts`. To run the script,
```bash
cd system2-lm
./script/run_llama_vlqa_chartQA.sh
```