# Season of Code : "Building LLMs from Scratch" 
This is a summary of what I learned so far in the four weeks of this project.
First of all while studying all the history of transformers and LLMs in detail, i came across a number of research papers that led to the development of powerful LLMs in such a small period of time. Ihave uploaded all of them in this repository for my future references, and to take a look at them everytime i want to study a concept in detail. Therefore, these can be skipped for evealuation purposes.
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

  ## WEEK-2
  
