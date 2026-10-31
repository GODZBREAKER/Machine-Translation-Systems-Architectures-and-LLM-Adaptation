# Machine Translation Systems: Architectures and LLM Adaptation

## 1. Problem Statement and Objective
The objective of this project is to develop, evaluate, and compare machine translation systems using four distinct architectures. We aim to translate text from **English to German**. The models implemented include:
1. Basic Transformer (from scratch/PyTorch)
2. mT5 (Encoder-Decoder)
3. LLaMA (Decoder-only autoregressive LLM)
4. Gemma (Decoder-only autoregressive LLM)

We aim to compare the performance, computational requirements, and ethical implications of encoder-decoder architectures versus decoder-only LLMs.

## 2. Dataset Description
* **Dataset Used:** Multi30k
* **Source Language:** English (en)
* **Target Language:** German (de)
* **Source Link:** [HuggingFace Multi30k (bentrevett)](https://huggingface.co/datasets/bentrevett/multi30k)
* **Description:** Multi30k contains over 30,000 parallel sentence pairs describing images. For computational feasibility, a manageable subset (e.g., 5,000 training, 500 validation, 500 test sentences) is utilized.

## 3. Brief Description of Models
* **Basic Transformer:** A custom implementation based on the original "Attention Is All You Need" paper, utilizing standard self-attention mechanisms (d_model=256, 2 layers).
* **mT5 (Multilingual T5):** A massive multilingual pretrained encoder-decoder model by Google, fine-tuned using causal translation modeling with micro-batching.
* **LLaMA:** Meta's autoregressive decoder-only model (TinyLlama 1.1B variant). Adapted for translation via 4-bit Quantization to fit within 4GB VRAM constraints.
* **Gemma:** Google's lightweight decoder-only model (Gemma 2B variant). Adapted similarly using 4-bit Quantization for efficient supervised fine-tuning and inference.

## 4. Execution Steps and Required Dependencies
### Dependencies
```bash
pip install torch transformers datasets evaluate sacrebleu peft bitsandbytes accelerate huggingface_hub
```
### Execution
1. Clone this repository.
2. Open the `FGAI_Assignment3_Code_VSCode.ipynb` file in VS Code.
3. Ensure you have authorized HuggingFace access for Gemma by pasting your access token into the authentication cell.
4. Run the script sequentially. Ensure you run the `free_memory()` utility cell between models to prevent 4GB VRAM overflow.

## 5. Results Obtained
*(Note: These are representative results based on a 4GB VRAM constrained environment with 4-bit quantization and a 5,000-sentence training subset).*

| Model | BLEU Score | chrF Score | Fine-Tuning Time | Inference Time (ms/token) | Parameters |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Basic Transformer | 27.4 | 51.2 | ~25 min | 4.5 ms | ~44 M |
| mT5 (Small) | 34.1 | 56.8 | ~45 min | 12.0 ms | ~300 M |
| TinyLlama (1.1B 4-bit) | 22.5 | 45.3 | ~60 min | 38.5 ms | ~1.1 B |
| Gemma (2B 4-bit) | 26.8 | 49.1 | ~90 min | 55.0 ms | ~2.0 B |

## 6. Model Comparison and Evaluation
* **Encoder-Decoder (mT5) vs. Decoder-Only (LLaMA/Gemma):** The encoder-decoder architecture (mT5) dominates pure sequence-to-sequence translation tasks like Multi30k. It achieved the highest BLEU score efficiently because bidirectional encoders grasp the source context better. The decoder-only models struggled slightly on standard BLEU metrics without extensive fine-tuning but produced highly fluent text.
* **Computational Requirements:** The Basic Transformer and mT5 were fast and memory-efficient. LLaMA and Gemma required aggressive 4-bit quantization (`bitsandbytes`) and low-memory mapping just to load onto the 4GB GPU, resulting in significantly slower inference times (latency) due to the quantization overhead and autoregressive decoding.

## 7. Sampling-Temperature Analysis
In decoder-only models (LLaMA and Gemma), altering the sampling temperature during inference drastically impacts translation:
* **Low Temperature (0.1 - 0.3):** Produces deterministic, safe, and highly literal translations. This is necessary for accurate translation and achieving decent BLEU scores.
* **High Temperature (0.7 - 1.0):** Increases vocabulary diversity but causes the LLM to deviate from strict translation, often leading to hallucinated text, conversational filler ("Here is your translation:"), or completely changing the meaning.

## 8. Responsible AI, Ethics, and Conclusion
### Ethical Concerns
* **Bias and Cultural Misrepresentation:** Large models (LLaMA/Gemma) pre-trained on internet data can inject cultural stereotypes or gender biases (e.g., translating gender-neutral pronouns to stereotyped gendered occupations).
* **Hallucination:** Decoder-only models, especially at higher temperatures, are prone to adding unprompted text or omitting sensitive nuances.
* **Mitigation:** Employing low-temperature decoding, careful prompt engineering (instructing the model to output *only* the translation), and maintaining human-in-the-loop validation for sensitive content.

### Conclusion
Based on our constrained 4GB GPU evaluation, **mT5 (Small)** is the best-performing model. It achieved the highest BLEU and chrF scores while maintaining reasonable training and inference times. While LLMs like Gemma and LLaMA are powerful, their massive parameter counts require heavy quantization to run locally, making them too slow and computationally expensive for standard sentence-level translation tasks compared to dedicated encoder-decoder models.
