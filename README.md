# Season of Code : "Building LLMs from Scratch" 
This is a summary of what I learned so far in the four weeks of this project.

First of all while studying all the history of transformers and LLMs in detail, I came across a number of research papers that led to the development of powerful LLMs in such a small period of time. I have uploaded all of them in this repository for my future references, and to take a look at them everytime i want to study a concept in detail. Therefore, these can be skipped for evaluation purposes.
---

## WEEK-1 
1. This week, we shall primarily focus on getting our basics clear on LLMs, python and pytorch.
2. LLMs, as we know is a giant tweaked neural network which works on DL algorithms, so in order to get started we focused on clearing our basics on neural networks.
3. We reviewed basic python syntax and flow control in python programs, learned basic numpy functions and then spent some time learning basic pytorch.
4. Moving on, we learned basic terminology used in the this domain like AI, ML, DL, and LLMs and the basic difference between them.
5. Secret sauce of Large Language Models -- **Transformer Architecture**
6. Stages of Building LLMs which consists of :
   - *Pretraining* : training a model on a large, diverse dataset. Followed by
   - *Finetuning* :  refinement by training on narrower datasets, specific to a particular task or domain. Finetuned LLM can be used as a chatbot, personal assistant etc.
7. **Large** in LLMs signifies that the model is trained on billions of parameters.
8. **Language Models** in LLMs signifies that they do a wide range of NLP tasks like question-answering, sentiment analysis, translation etc.
9. **Basic intro to Transformers** : concept introduced in 2017 paper *Attention is all you need*.
    - Simple transformer architecture consists of an encoder(encodes input text into vectors) and a decoder(generates output text from encoded vctors).
    - *Self-attention mechanism* is a key part of transformers which allows the model to capture *lomg-range dependencies*.
    - GPT = Generative Pretrained Transformers is a variation of Transformer archotecture developed from the 2017 paper on transformers which mainly focuses on predicting the next word in a sequence of input text.
10. Learned how GPT-3 works :
    - *Zero-shot learning*: ability to generalise to completely unseen tasks without any prior specific examples.
    - *Few-shot learning*: learning from a minimum number of examples which the user provides as an input.
    - GPT-3 is a few-shot learner as it generates more accurate and relevant responses when provided a few examples. This does not mean that it cannot do zero-shot tasks, just that its a better few-shot learner.
    - Total pretraining cost of GPT-3 was around **4.6 million dollars**.
    - GPT models are simply trained on *next-word prediction tasks*.
    - Next word prediction is an example of **SELF SUPERVISED LEARNING**.
    - **Auto-regressive model** : uses previous outputs as inputs for future predictions.
    - Pretraining of GPT-3 model is *unsupervised* and *auto-regressive*.
    - GPT architecture has no encoder, we just have a decoder.
    - GPT shows **Emergent Behaviour** which is the ability of a model to perform tasks that the model wasn't explicitly trained to perform.
 Understood the basic flow of stages of building an LLM from scratch.
  ---

## WEEK-2 Building the entire Data Preprocessing Pipeline of Large Language Models (LLMs) 
1. Any Model, let it be a vision model or voice models or language models perhaps, only knows to input Numbers and process on numbers. Models don’t know what are pixels, sound waves or even words which can be perceived by humans very smoothly. We focused on converting our english Words into a specific kind of data, which can be processed by models by a process called **Tokenization**. 

2. Next we dealt with a concept called **Embeddings**. After tokenizing the data, it becomes important to store the meaning of those words during training process. Every token has an n-dimensional array, ie a kind of a vector in multi dimensions to store the meaning of the token assigned with language. 

3. Four Key Stages Covered:
   - **Tokenization**:
     - Different approaches: word-based, subword-based (including Byte Pair Encoding (BPE)), and character-based tokenizers.
     - Built a word-based tokenizer from scratch in Python, including handling punctuation and special characters as separate tokens.
     - Discussed the limitations of word-based tokenization, such as handling out-of-vocabulary words and large vocabulary size.
     - Got introduced to special context tokens like <unk> (unknown) and <eot> (end of text) to handle unknown words and document boundaries.
     - GPT models use subword-based tokenization, specifically BPE, which balances vocabulary size and the retention of root word meanings.
     - BPE works by iteratively merging frequent character pairs and allows handling of unknown words without explicit unknown tokens.
     - The use of the OpenAI tiktoken library for BPE tokenization, as used in GPT-2 and later models.

   - **Token Embeddings**:
     - Converts token IDs into high-dimensional vectors, enabling the model to process tokens numerically.

   - **Positional Embeddings**:
     - Adds information about the position of each token in a sequence, which is crucial since LLMs process sequences in parallel and need to understand order.

   - **Input Embeddings**:
     - The final input to the model is the sum of token embeddings and positional embeddings, forming the input embedding vector for each token.
Discussed how these preprocessing steps are foundational for effective LLM training and performance.
---

## WEEK-3 
1. Now it’s the time to build the heart and soul of the transformer architecture- namely **ATTENTION MECHANISM**.
2. Traditional models used sequence models like RNNs and LSTMs to process data sequentially but they used to struggle with long range dependencies. This changed all of a sudden when researchers at google, on 12th June 2017 released a research paper, *Attention is all you need*. It introduced us to the transformer architecture, and it revolutionised the field of natural language processing for ever. 
3. This week we shall dive more into this new architecture, starting with **self attention**. It enables the model to weigh and aggregate information across different positions of a single input sequence. It helps in building a richer context in long sequences. Next we shall go through causal and multi head attention to capture even more deep meaning in sentences and paragraphs. 
4. **Introduction to Attention Mechanism in LLMs**:
   - Learned why the attention mechanism is crucial for modern language models.
   - The evolution from RNNs and LSTMs to attention-based models.
   - Discussed four types of attention mechanisms and their roles.
   - The limitations of RNNs in modeling long sequences and how attention overcomes these issues.
   - Introduction to Bahdanau Attention and the concept of self-attention, which is foundational for transformers.
5. **Simplified Attention Mechanism (No Trainable Weights)**
   - Coding a basic attention mechanism from scratch, focusing on theory and practical intuition.
   - Key concepts: context vectors, attention scores (via dot product), and attention weights.
   - Understood how to normalize attention scores (simple division and softmax) for interpretability and stability.
   - The calculation of context vectors as weighted sums of input embeddings.
   - Emphasized on the importance of normalization (especially softmax) for effective learning and interpretability.
6. **Self-Attention with Trainable Weights**:
   - Expands the simplified mechanism by introducing trainable weight matrices (queries, keys, values).
   - Learned how these matrices allow the model to learn which parts of the input to focus on for each token.
   - The computation of attention scores using learned projections and the role of backpropagation in optimizing these weights.
7. **Causal (Masked) Attention**:
   - The need for causal (or masked) attention in autoregressive models (like GPT), ensuring that each token only attends to previous tokens, not future ones.
   - Learned how masking is implemented in the attention score matrix to prevent information "leakage" from future tokens during training and inference.
8. **Multi-Head Attention**:
   - Introduction to the concept of multi-head attention, where multiple attention mechanisms (heads) run in parallel on different projections of the input.
   - Learned how this allows the model to capture diverse relationships and dependencies in the data.
   - Detailed understanding of the concatenation and linear transformation steps that combine the outputs of all heads.
   - Learned about the benefits of multi-head attention in improving model expressiveness and performance.
---

## WEEK-4 
1. After last week’s exploration of attention mechanisms, we now have a strong understanding of how models like Transformers capture contextual meaning in sequences. This week, we build on that foundation to construct the full architecture behind modern language models.
2. We begin with a high-level overview of the Transformer architecture, understanding how various components interact to process and generate text. We then implement a complete Transformer block—including self-attention, feedforward networks, normalization, and residual connections—from scratch using PyTorch.
3. **Birds Eye View of the LLM Architecture**:
   - Had an overview of the architecture of Large Language Models (LLMs), focusing on how components like token embeddings, transformer blocks, feedforward neural networks, and the output layer interact for next-word prediction.
   - Understood the GPT-2 architecture, detailing how input text is tokenized, embedded, passed through transformer blocks, and finally processed to produce output logits.
4. **Layer Normalization in the LLM Architecture**:
   - Learned about the purpose and mechanics of layer normalization, a critical component for stable and efficient training in deep neural networks.
   - Discussed how layer normalization prevents vanishing/exploding gradients and internal covariate shift, leading to faster convergence.
   - Got to know from the flowchart where layer normalization fits in the transformer block and the overall GPT architecture, emphasizing its independence from batch size.
5. **GELU Activation Function in Transformers**:
   - Introduced the *Gaussian Error Linear Unit (GELU)* activation function, commonly used in transformer models.
   - Learned the mathematical intuition behind GELU, its advantages over ReLU, and why it is preferred in LLMs.
   - Demonstrated implementation and integration of GELU in the feedforward layers of the transformer block.
6. **Feedforward Neural Networks in Transformer Blocks**:
   - Had an overview of the structure of the feedforward neural network (FFN) within each transformer block.
   - Learned how FFNs process the output of attention layers, typically using two linear layers with a non-linear activation (like GELU) in between.
7. **Shortcut (Residual) Connections**:
   - Understood the concept of shortcut or residual connections, which add the input of a layer to its output.
   - Learned how these connections help mitigate the vanishing gradient problem, enabling the training of very deep networks.
   - Finally saw in code how to implement residual connections in the transformer block.
8. **Putting It All Together: The Transformer Block**:
   - Combined all previously discussed components—multi-head attention, layer normalization, feedforward networks, GELU activation, and residual connections—into a complete transformer block.
---
