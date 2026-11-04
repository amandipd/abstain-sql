# Model checks (plan step 0.3)

Generated 2026-11-04 by `train/inspect_models.py` with transformers 5.19.0. Configs and tokenizers only; no weights were downloaded.

## Qwen/Qwen3.5-2B

- Architecture (the name vLLM keys on): `Qwen3_5ForConditionalGeneration`
- Model type: `qwen3_5`
- Vocabulary size: 248320
- Max context (positions): 262144
- License: apache-2.0 (https://huggingface.co/Qwen/Qwen3.5-2B/blob/main/LICENSE)
- Tokenizer class: `Qwen3_5Tokenizer`, 248077 tokens incl. added ones

Chat template on one sample question, with the thinking switch:

**default:**

```
<|im_start|>user
How many customers live in Austin?<|im_end|>
<|im_start|>assistant
<think>

</think>

```

**thinking on:**

```
<|im_start|>user
How many customers live in Austin?<|im_end|>
<|im_start|>assistant
<think>
```

**thinking off:**

```
<|im_start|>user
How many customers live in Austin?<|im_end|>
<|im_start|>assistant
<think>

</think>

```

## HuggingFaceTB/SmolLM3-3B

- Architecture (the name vLLM keys on): `SmolLM3ForCausalLM`
- Model type: `smollm3`
- Vocabulary size: 128256
- Max context (positions): 65536
- License: apache-2.0
- Tokenizer class: `TokenizersBackend`, 128256 tokens incl. added ones

Chat template on one sample question, with the thinking switch:

**default:**

```
<|im_start|>system
## Metadata

Knowledge Cutoff Date: June 2025
Today Date: 04 November 2026
Reasoning Mode: /think

## Custom Instructions

You are a helpful AI assistant named SmolLM, trained by Hugging Face. Your role as an assistant involves thoroughly exploring questions through a systematic thinking process before providing the final precise and accurate solutions. This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracking, and iteration to develop well-considered thinking process. Please structure your response into two main sections: Thought and Solution using the specified format: <think> Thought section </think> Solution section. In the Thought section, detail your reasoning process in steps. Each step should include detailed considerations such as analysing questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, and revisiting previous steps. In the Solution section, based on various attempts, explorations, and reflections from the Thought section, systematically present the final solution that you deem correct. The Solution section should be logical, accurate, and concise and detail necessary steps needed to reach the conclusion.

<|im_start|>user
How many customers live in Austin?<|im_end|>
<|im_start|>assistant
```

**thinking on:**

```
<|im_start|>system
## Metadata

Knowledge Cutoff Date: June 2025
Today Date: 04 November 2026
Reasoning Mode: /think

## Custom Instructions

You are a helpful AI assistant named SmolLM, trained by Hugging Face. Your role as an assistant involves thoroughly exploring questions through a systematic thinking process before providing the final precise and accurate solutions. This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracking, and iteration to develop well-considered thinking process. Please structure your response into two main sections: Thought and Solution using the specified format: <think> Thought section </think> Solution section. In the Thought section, detail your reasoning process in steps. Each step should include detailed considerations such as analysing questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, and revisiting previous steps. In the Solution section, based on various attempts, explorations, and reflections from the Thought section, systematically present the final solution that you deem correct. The Solution section should be logical, accurate, and concise and detail necessary steps needed to reach the conclusion.

<|im_start|>user
How many customers live in Austin?<|im_end|>
<|im_start|>assistant
```

**thinking off:**

```
<|im_start|>system
## Metadata

Knowledge Cutoff Date: June 2025
Today Date: 04 November 2026
Reasoning Mode: /no_think

## Custom Instructions

You are a helpful AI assistant named SmolLM, trained by Hugging Face.

<|im_start|>user
How many customers live in Austin?<|im_end|>
<|im_start|>assistant
<think>

</think>
```

## google/gemma-4-E2B-it

- Architecture (the name vLLM keys on): `Gemma4ForConditionalGeneration`
- Model type: `gemma4`
- Vocabulary size: 262144
- Max context (positions): 131072
- License: apache-2.0 (https://ai.google.dev/gemma/docs/gemma_4_license)
- Tokenizer class: `GemmaTokenizer`, 262144 tokens incl. added ones

Chat template on one sample question, with the thinking switch:

**default:**

```
<bos><|turn>user
How many customers live in Austin?<turn|>
<|turn>model
```

**thinking on:**

```
<bos><|turn>system
<|think|>
<turn|>
<|turn>user
How many customers live in Austin?<turn|>
<|turn>model
```

**thinking off:**

```
<bos><|turn>user
How many customers live in Austin?<turn|>
<|turn>model
```
