# Season of Code : "Building LLMs from Scratch" 
This is a summary of what I learned so far in the four weeks of this project.

NOTE : The "combined_code.ipynb" file contains all the code written over the weeks of progress, it is still not fully completed as was intended to be at the start of the project, but a major part is done and this submission is made by me after concerning my mentor. The two files required to run this combined file are "the-verdict.txt" and "gpt_download3.py".

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


## WEEK-5 Topics focused on: 
1. Measuring the LLM Loss Function :
   - We focused on understanding the role of loss functions in LLM training, specifically the use of cross entropy loss to measure model fit.
   - Learned about the input-target construction for next-token prediction: each input sequence leads to several prediction tasks (not just the last token to make the LLM look more human).
   - Learned about how LLM outputs logits (pre-softmax values per token), which are compared with target tokens to compute loss.
   - Set a foundation for integrating gradient-based training (backpropagation) by defining the loss quantitatively.

2. Evaluating LLM Performance on a Real Dataset :
   - Performed a hands-on project using a real storybook dataset ("the verdict") to evaluate LLM loss.
   - The project demonstrates tokenizing the dataset using Byte Pair Encoding (BPE), and how to split data into training and validation sets.
   - Also shows how to create input-target pairs using context windows and stride, vital for proper AutoRegressive language modeling.
   - Implemented PyTorch DataLoader for batching and efficient data feeding.
   - Measured training and validation loss without full-blown training—setting up the workflow to benchmark improvement as later training proceeds.

3. Coding the Entire LLM Pre-training Loop :
   - Revised how input-target pairs are generated and how cross entropy loss is defined.
   - Implemented the full pre-training loop: batching data, passing input through the model, calculating logits, computing loss, and performing backpropagation using PyTorch.
   - Discussed batching, epochs, and parameter optimization (with hundreds of millions of parameters).
     
4. Temperature Scaling in LLMs :
   - Got introduced to temperature scaling as a decoding strategy during text generation, controlling the randomness and creativity of model outputs.
   - Showed that dividing logits by a "temperature" parameter before softmax can sharpen (low T) or flatten (high T) the probability distribution over next tokens.
   - Learned the difference between deterministic (argmax/greedy) and probabilistic (sampling) decoding.

5. Top-k Sampling in LLMs :
   - top-k sampling (often combined with temperature scaling) is used to further control output randomness by limiting candidate tokens at each step to the top k likely ones.
   - Discussed how this manages the tradeoff between creativity and coherence and reduces the chance of generating nonsensical or memorized output.
---

## WEEK 6 
1. Loading Pre-trained Weights from OpenAI GPT-2 :
   - Loaded OpenAI GPT-2 pre-trained weights into a custom-built GPT architecture, improving output coherence.
   - Explaination of the structure and content of GPT-2 checkpoint files (including weights, vocab, BPE merges, model config).
   - Learned the conversion and integration of TensorFlow-saved GPT-2 weights into a PyTorch-based LLM pipeline.
   - Tested the model with loaded GPT-2 weights, comparing text generation before and after integration; significantly improved text fluency and coherence.

2. Introduction to LLM Fine-tuning : 
   - Introduction to LLM fine-tuning and why it’s needed after pre-training for application-specific tasks.
   - Two main types of fine-tuning:
        - Instruction-based fine-tuning (model follows prompt instructions, suitable for diverse tasks).
        - Classification-based fine-tuning (model classifies input into fixed categories, e.g., spam/ham).
   - Learned hands-on start to classification finetuning by downloading and preprocessing a spam vs. non-spam email dataset.
   - Full fine-tuning workflow to follow (data prep, model init, training, evaluation, usage on new data).

3. Dataloaders in LLM Classification Finetuning :
   - The importance of consistent input length for batching—demonstrates padding/truncating emails to the max sequence length using GPT-2’s end-of-text token.
   - Implemented a PyTorch Dataset class to tokenize emails and pad them as needed.
   - Implemented PyTorch DataLoader objects to create batches for training, validation, and testing.

4. Coding the Model Architecture for LLM Classification Fine-tuning :
   - Modified the original GPT model to support classification (spam detection) by adding a classification head (output layer with two neurons for spam/not spam).
   - Loaded OpenAI GPT-2 pre-trained weights into the modified architecture.
   - Understood the explanation for updating only specific (selected) layers during fine-tuning.
   - The code shows how to extract the final (last token’s) output for classification, as it encodes information from the entire input due to the transformer’s attention mechanism.

5. Coding a Fine-tuned LLM Spam Classification Model : 
   - Implemented full training (fine-tuning) loop for the spam classifier on the labeled dataset, using cross-entropy loss and the AdamW optimizer.
   - Learned how to convert logits to label predictions (using argmax over the output layer).
   - Calculated accuracy and loss throughout training and validation phases.
   - Analyzed results: decreasing loss curves, increasing accuracy, and checks for overfitting by comparing validation and training results.
   - Finalized a practical, working LLM-based spam classifier tested on new data, concluding the full from-scratch fine-tuning pipeline.
---

## WEEK 7
1. Introduction to Instruction Fine-Tuning:
   - Instruction fine-tuning: adapting a pre-trained LLM to follow specific instructions rather than generic text completion.
   - Practical application: E-commerce chatbots, personalized healthcare assistants, and domain-specific question answering require fine-tuning on relevant instructions.
   - Dataset preparation: Uses a set of 1,100 instruction-response pairs, each with an instruction, optional input, and expected output.
   - Data is split into train (85%), test (10%), and validation (5%) sets.

2. Data Batching and Formatting:
   - Illustrates how instruction, input, and output fields are merged into a single prompt following the "Alpaca prompt style".
   - Steps in batching:
        - Formatting each prompt using the Alpaca template.
        - Tokenizing the formatted prompts (using OpenAI’s tiktoken/BPE).
        - Sequences of varying length are padded to a uniform length for batch training.
        - Target IDs (for output/labels) are created with masked tokens (ignore_index = -100) to avoid loss computation on padding.
        - Explains how batching enables efficient model training.

3. Data Loaders for Efficient Training:
   - Shows the use of PyTorch DataLoader classes to handle batches of tokenized, padded data.
   - DataLoader provides iterators for training, validation, and testing datasets.

4. Loading Pre-Trained Model Weights:
   - Explained the importance of initializing the LLM with pre-trained weights (GPT-2 355M parameter "medium" model) before fine-tuning.
   - Details downloading and integrating GPT-2 weights for all model layers.
   - Discussed the advantage: faster and more effective fine-tuning as the model starts from a knowledgeable state instead of random parameters.

5. Fine-Tuning Training Loop:
   - Implements the standard PyTorch training loop: batch input, compute predictions, cross-entropy loss, backward pass, and optimizer update.
   - Target: Model learns to predict the next token for each instruction (next-token prediction task).
   - Loss function: Cross-entropy, with masking for padded locations.
   - Training and validation loss are tracked; results show significant loss reduction and improved response accuracy after fine-tuning.
   - Shows qualitative improvements in model responses, such as converting sentences from active to passive voice and generating similes.

6. Evaluating the Fine-Tuned LLM:
   - Moved to systematic evaluation after training.
   - Discussed three evaluation strategies:
     - MMLU: Standardized benchmarks.
     - Manual/human preference: Comparing outputs by hand.
     - LLM-as-judge: Using another LLM (e.g., Llama 3 via Ollama) for automated qualitative scoring.
    

I trained the model after instruction finetuning for 2 epochs and the results were quite improved than 1 epoch earlier.
I also intended to use ollama's llama3 to evaluate my model but it required a lot of storage space and laptop had some storage issues, so had to stop here in code.
That and the deployment part was mainly what couldn't be achieved during the project, otherwise the project is mostly complete. Thank for your time !!

End of the project 
---
