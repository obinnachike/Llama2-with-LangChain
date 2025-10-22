#  LLM Prompt Engineering with LangChain and Llama-2

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/156QAcdh8hBxCpIrRqDuAJGY_bPRdsQYl)

---

<p align="center">
  <img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" height="70">
  <img src="https://python.langchain.com/_static/logo.png" height="70">
  <img src="https://upload.wikimedia.org/wikipedia/en/6/6a/Meta_Platforms_Inc._logo.svg" height="70">
  <img src="https://storage.googleapis.com/gweb-cloudblog-publish/images/Google_Cloud_logo.width-400.png" height="70">
  <img src="https://upload.wikimedia.org/wikipedia/en/3/33/NVIDIA_logo.svg" height="70">
  <img src="https://upload.wikimedia.org/wikipedia/commons/1/10/PyTorch_logo_icon.svg" height="70">
</p>

---

##  Overview

This project demonstrates how to build a **prompt-driven text-generation workflow** using **Llama-2 (7B)** with **LangChain** and **Hugging Face Transformers**.
It shows how to:

* Initialize a quantized large language model for inference,
* Create reusable prompt templates,
* Chain user inputs to LLM calls for dynamic responses.

---

##  Environment Setup

Check GPU availability:

```bash
!nvidia-smi
```

Install all dependencies:

```bash
!pip install -q transformers einops accelerate langchain bitsandbytes
```

Authenticate your Hugging Face account:

```bash
!huggingface-cli login
```

---

##  Import Libraries

Install additional community integrations:

```bash
!pip install langchain_community
```

Import required modules:

```python
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer
import transformers
import torch
import warnings

warnings.filterwarnings('ignore')
```

---

##  Load and Configure the Model

Select a pretrained **Llama-2-7B-Chat** model:

```python
# model = "meta-llama/Llama-2-7b-chat-hf"
model = "daryl149/llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model)
```

Build the text-generation pipeline:

```python
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
    max_length=1000,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id
)
```

Wrap it into a LangChain LLM interface:

```python
llm = HuggingFacePipeline(pipeline=pipeline, model_kwargs={'temperature': 0})
```

---

##  Example Prompts

Generate creative names:

```python
prompt = "What would be a good name for a company that makes colorful socks?"
print(llm(prompt))
```

Generate themed names:

```python
prompt = "I want to open a restaurant for Indian food. Suggest me a fancy name for this."
print(llm(prompt))
```

---

##  Prompt Templates with LangChain

LangChain helps you standardize and reuse prompts in your LLM applications.
Instead of writing the whole prompt each time, define **PromptTemplates**.

```python
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
```

Create and format templates:

```python
prompt_template1 = PromptTemplate(
    input_variables=["cuisine"],
    template="I want to open a restaurant for {cuisine} food. Suggest a fancy name for this."
)

input_prompt = prompt_template1.format(cuisine="indian")
print(input_prompt)
```

Book-summary example:

```python
prompt_template2 = PromptTemplate(
    input_variables=["book_name"],
    template="Provide me a concise summary of the book {book_name}."
)

input_prompt = prompt_template2.format(book_name="Alchemist")
print(input_prompt)
```

---

##  Build a Prompt Chain

Combine the LLM and prompt template into a single callable chain:

```python
chain = LLMChain(llm=llm, prompt=prompt_template2, verbose=True)
response = chain.run("Harry Potter")
print(response)
```

---

##  Summary

You have now:
 Initialized a **Llama-2 model** through the **Hugging Face Transformers** API.
 Integrated it with **LangChain** for scalable prompt management.
 Executed real-time text generation using structured prompts.

---

Developed by **Chiejina Chike Obinna**.

---
