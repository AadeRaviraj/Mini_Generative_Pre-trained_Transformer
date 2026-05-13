# Mini GPT - Next Word Prediction using Decoder-Only Transformer

A deep learning project that builds a Mini GPT model from scratch using a Decoder-Only Transformer architecture. The model learns next-word prediction from simple text sentences and can generate new text autoregressively — predicting one word at a time.

---

## Table of Contents

- [Project Overview](#project-overview)
- [How It Works](#how-it-works)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Key Concepts Explained](#key-concepts-explained)
- [Technologies Used](#technologies-used)
- [Getting Started](#getting-started)
- [Training Details](#training-details)
- [Next Word Prediction](#next-word-prediction)
- [Text Generation](#text-generation)
- [Project File Structure](#project-file-structure)
- [License](#license)

---

## Project Overview

This project implements a simplified version of GPT (Generative Pre-trained Transformer) using a Decoder-Only Transformer built entirely from scratch with TensorFlow and Keras. The model is trained on a small set of sentences and learns to predict the next word given a sequence of previous words.

It also supports autoregressive text generation — where each predicted word is fed back into the model to generate the next word, exactly how real GPT models work.

---

## How It Works

```
Input sentence  :  "i love artificial"
Target sentence :  "love artificial intelligence"

The model learns:
  After "i"                 → predict "love"
  After "i love"            → predict "artificial"
  After "i love artificial" → predict "intelligence"
```

This is called **next-word prediction** or **causal language modeling** — the same core concept used in GPT-2 and GPT-3.

---

## Dataset

The model is trained on 20 hand-crafted sentences covering topics like artificial intelligence, machine learning, transformers, and programming:

```
i love artificial intelligence
i love machine learning
i love deep learning
machine learning is powerful
deep learning is amazing
artificial intelligence is future
transformer is powerful model
decoder transformer generates text
gpt is decoder only transformer
students learn artificial intelligence
students learn python programming
python is useful language
ai helps students learn
transformer learns word relations
masked attention prevents future words
language model predicts next word
gpt predicts next token
decoder only model generates output
self attention understands context
positional encoding gives word order
```

---

## Model Architecture

The model is built using the Keras Functional API and has three main components.

### 1. Token and Positional Embedding

| Component            | Details                                                   |
|----------------------|-----------------------------------------------------------|
| Token Embedding      | Converts word indices into dense vectors                  |
| Positional Embedding | Adds position information so the model knows word order   |
| Output               | Token embedding + Positional embedding (element-wise sum) |

### 2. Decoder-Only Transformer Block (stacked x2)

Two identical Transformer blocks are stacked on top of each other:

| Component                   | Details                                              |
|-----------------------------|------------------------------------------------------|
| Masked Multi-Head Attention | 2 heads — each word attends only to previous words   |
| Feed Forward Network        | Dense(64, ReLU) followed by Dense(embed_dim)         |
| Layer Normalization         | Applied after attention and after feed forward       |
| Dropout                     | 10% dropout after attention and after feed forward   |

### 3. Output Layer

| Component | Details                                                  |
|-----------|----------------------------------------------------------|
| Dense     | vocab_size neurons with Softmax activation               |
| Output    | Probability distribution over the entire vocabulary      |

### Model Configuration

| Parameter          | Value                           |
|--------------------|---------------------------------|
| Vocabulary size    | 1000                            |
| Sequence length    | 6                               |
| Embedding dim      | 32                              |
| Number of heads    | 2                               |
| Feed forward dim   | 64                              |
| Transformer blocks | 2                               |
| Loss function      | Sparse Categorical Crossentropy |
| Optimizer          | Adam                            |
| Epochs             | 300                             |
| Batch size         | 2                               |

---

## Key Concepts Explained

### Causal Mask (Masked Self-Attention)

The causal mask ensures the model cannot see future words during training. For a sequence of length 6, the mask matrix looks like this:

```
[[1 0 0 0 0 0]
 [1 1 0 0 0 0]
 [1 1 1 0 0 0]
 [1 1 1 1 0 0]
 [1 1 1 1 1 0]
 [1 1 1 1 1 1]]

1 = word is visible
0 = word is hidden (masked)
```

- Word at position 0 can only see position 0
- Word at position 1 can see positions 0 and 1
- Word at position 2 can see positions 0, 1 and 2

This is what makes it a **Decoder-Only** Transformer — the same design principle used in real GPT models.

### Autoregressive Text Generation

```
Start text : "i love"
Step 1     : predict "artificial"   → "i love artificial"
Step 2     : predict "intelligence" → "i love artificial intelligence"
Step 3     : predict next word      → continues...
```

Each predicted word is appended to the input and fed back into the model to predict the next word.

---

## Technologies Used

| Technology        | Purpose                                  |
|-------------------|------------------------------------------|
| Python 3.x        | Core programming language                |
| TensorFlow/Keras  | Deep learning and Transformer framework  |
| NumPy             | Numerical operations and array handling  |

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/mini-gpt-decoder-transformer.git
cd mini-gpt-decoder-transformer
```

### 2. Install Dependencies

```bash
pip install tensorflow numpy
```

### 3. Run the Project

```bash
python Transformer_Generative_Pre_trained_Transformer_Decoder_Only.py
```

The script runs through all 11 steps automatically:

1. Create and display the training dataset
2. Tokenize and pad sentences using TextVectorization
3. Prepare input and target sequences
4. Build the Token and Positional Embedding layer
5. Demonstrate the causal mask matrix
6. Build and compile the Mini GPT model
7. Train the model for 300 epochs
8. Build the reverse vocabulary (index to word mapping)
9. Run next-word predictions on test inputs
10. Run autoregressive text generation

---

## Training Details

The model is trained using next-word prediction at every position in the sequence simultaneously.

```
Input  :  [i, love, artificial, intelligence, [pad], [pad]]
Target :  [love, artificial, intelligence, [pad], [pad], [pad]]
```

| Setting    | Value                           |
|------------|---------------------------------|
| Loss       | Sparse Categorical Crossentropy |
| Optimizer  | Adam                            |
| Epochs     | 300                             |
| Batch size | 2                               |

Training for 300 epochs on this small dataset is intentional — the model is meant to memorize the sentence patterns so it can generate coherent word completions from the training vocabulary.

---

## Next Word Prediction

The `predict_next_word()` function takes a partial sentence and returns the next predicted word:

```python
predict_next_word("i love")
# Input Text     : i love
# Predicted Word : artificial

predict_next_word("machine learning is")
# Input Text     : machine learning is
# Predicted Word : powerful

predict_next_word("gpt is")
# Input Text     : gpt is
# Predicted Word : decoder

predict_next_word("language model predicts")
# Input Text     : language model predicts
# Predicted Word : next

predict_next_word("masked attention prevents")
# Input Text     : masked attention prevents
# Predicted Word : future
```

**How it works internally:**

1. Tokenizes the input text using the trained vectorizer
2. Feeds the token indices into the model
3. Reads the output probability at the last input word position
4. Returns the word with the highest probability (argmax)

---

## Text Generation

The `generate_text()` function generates multiple words autoregressively:

```python
generate_text("i love", number_of_words=4)
# Start Text     : i love
# Generated Text : i love artificial intelligence future

generate_text("decoder transformer", number_of_words=4)
# Start Text     : decoder transformer
# Generated Text : decoder transformer generates text model

generate_text("students learn", number_of_words=4)
# Start Text     : students learn
# Generated Text : students learn artificial intelligence

generate_text("gpt", number_of_words=4)
# Start Text     : gpt
# Generated Text : gpt is decoder only transformer
```

**How it works internally:**

1. Starts with the given seed text
2. Predicts the next word using `predict_next_word()`
3. Appends the predicted word to the existing text
4. Repeats for the requested number of words
5. Stops early if the model produces an unknown token `[UNK]`

---

## Project File Structure

```
mini-gpt-decoder-transformer/
├── Transformer_Generative_Pre_trained_Transformer_Decoder_Only.py      <- Main project script (all 11 steps in one file)
└── README.md
```

---

## License

This project is intended for educational purposes.
