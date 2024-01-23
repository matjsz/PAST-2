# PAST-2
**PAST-2** (**PT-AN Seq2Seq Transformer Translator**) is a neural network for translation from Portuguese to my fictional language called Angrarosskesh.

## About Angrarosskesh
Angrarosskesh is my fictional language for a literary project (aka the book I'm writing). It's a mix of English (Anglo/Angra) and Russian (Rossci/Rosske).

## Why am I training a neural network for this?
I often find myself lost within my own notes to get a grip from this language, even if it's me who is creating it, so if I teach a neural network how to speak it, then I don't need to rely on my confusing and messy notes on my equally confusing and messy notebook.

---

## About the model
**PAST-2** comes from **PT-AN** (Portuguese to Angrarosskesk) **Seq2Seq** (Sequence to Sequence encoding-decoding algorithm-based) **Transformer** (it's not a RNN, although pretty similar, and neither a CNN, even if it's also pretty similar, it relies on transformer-based algorithms) **Translator** (you already know it) and **2** (because of the two S and T in Seq2Seq and Transformer-Translator, just a nice touch).

- **Developed b:y** Matheus J. G. Silva
- **Model type**: Language model
- **License:** MIT License
- **Related Models:** ConvS2S (https://arxiv.org/abs/1705.03122)

## Uses
Unless you really wanna know how Angrarosskesh sounds, it's just a nice AI model. You can also fine-tune this model, I don't have problems with it, just remember to cite me or link this repository, it matters a lot!

## Bias, Risks, and Limitations
Still don't know it. WIP.

# Training Details

## Training Data

The model is going to be trained based on my own dataset (nobody whave a PT-AN dataset available, sadly), which will be available at HuggingFace Hub at **matjs/pt_to_an**.

## Training Procedure

The model involves a training procedure that brings together the approaches used in most of the Seq2Seq models out there, which is:
- Dataset preparation
- Tokenization
- Preparation of the embedding and encoding layers
- Preparation of the global self attention layer
- Interpretation of data on feed foward network
- Final preparation of encoding and decoding layers
- Transformer hyper parameters preparation
- Loss and metrics
- Training (based on the dataset created)
- Inference
- **Translate it!**

# Evaluation
Still no one. WIP.

## Testing Data, Factors & Metrics
Still no one. WIP.

## Results
Still no one. WIP.

## Environmental Impact
- **Hardware Type:** AWS EC2 Cloud GPU Instancing
- **Hours used:** WIP 
- **Cloud Provider:** AWS
- **Compute Region:** Brazil
- **Carbon Emitted:** Carbon emissions can be estimated using the Machine Learning Impact calculator presented in Lacoste et al. (2019).

# Citation

## BibTeX:
@misc {<br>
&emsp;&emsp;2024PAST-2,<br>
&emsp;&emsp;title   = {MATJSZ/PAST-2: PAST-2 (PT-an Seq2Seq Transformer Translator) - A neural network for translation from Portuguese to Angrarosskesh, a fictional language.},<br>
&emsp;&emsp;url     = {https://github.com/matjsz/PAST-2},<br>
&emsp;&emsp;journal = {GitHub},<br>
&emsp;&emsp;author  = {Silva, Matheus J. G.},<br>
&emsp;&emsp;year    = {2024},<br>
&emsp;&emsp;month   = {Jan}<br>
}

# APA:
Silva, MATJSZ/PAST-2: PAST-2 (PT-an Seq2Seq transformer translator) - A neural network for translation from Portuguese to Angrarosskesh, a fictional language. 2024.
