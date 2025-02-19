## Ideas for future examples/demos

- Use e.g. Typescript compiler to check validity of completions at generation time
    - checking validity on a per-token basis is impractical, so you'd need a lineage of completions to go 3-5 tokens without generating compilable Typescript before you could cull it.
    - maybe you need to break at control characters, e.g. parentheses for Typescript, quotes for JSON, etc
    - backtracking n characters/tokens until you can get syntactically valid json? still requires unidirectional parsing support
- Generate valid JSON
- Single-embedding steering: Subtract 0.5 from cosine similarity score, multiply resulting steering constant by 100 or so. OR you might need to calculate the cosine similarity relative to the other proto-completions.
- Is steering based on maximizing the delta between the embedding of the string minus the token and the embedding of the string plus the token viable?
- Fill in the blanks, maybe while also optimizing for a maximum possible text regressor score
- Train a context-specific regressor (simple dictionary?) that rates how likely a token is to appear in a given context, and use that to steer. You could sample from a GPT2 finetune trained on the corpus and evaluate by logits? or sample from both and evaluate by a combination likelihood score * logits a la the prompt intersection example. In principle, maybe the model's logits will be less "confident" on tokens that require context to predict, allowing the context-specific regressor to step in on such tokens (as long as the contextual token is being sampled.)
- GAN: generate a completion, then check if it is distinguishable from other completions in a set. Modify hyperparameters/steering embedding to make it theoretically more similar to the other set members. Gradient descend.
- Does weight represent likelihood assigned by the model?
- Web/API interface: build HFPPL model via web code editor (or CLI?), assign to unique ID, get completion by calling an API with the HFPPL model ID, LLM from HF hub, and the model signature (prompt, etc.) Should look kinda like Apify.
    - would need to deploy the hfppl code to a server, instantiate the transformers model/tokenizer
    - upload the PPL decoding model via API/CLI, store in db for future use
    - call the API with PPL model ID, transformers model ID, and prompt
    - get completion by instantiating PPL model, calling using hf transformers' native support for AWQ models. static batching is probably fine initially. continuous batching offers 2x throughput gain @ 32 max 32 tokens -> 10x gain @ 512 max tokens.
- If I'm understanding correctly, llama.cpp's grammar sampling works by zeroing out the logits of sampled tokens that don't conform to the BNF grammar.