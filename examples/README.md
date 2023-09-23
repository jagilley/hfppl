## Ideas for future examples/demos

- Use e.g. Typescript compiler to check validity of completions at generation time
- Generate valid JSON
- Is single-embedding steering viable? Subtract 0.5 from cosine similarity score, multiply resulting steering constant by 100 or so. OR you might need to calculate the cosine similarity relative to the other proto-completions.
    - Relatedly, is steering with a single text regressor viable?
- Is steering based on maximizing the differential between the embedding of the string minus the token and the embedding of the string plus the token viable?
- Fill in the blanks, maybe while also optimizing for a maximum possible text regressor score
- Train a context-specific regressor (simple dictionary?) that rates how likely a token is to appear in a given context, and use that to steer. You could sample from a Markov chain model trained on the corpus and evaluate by logits? or sample from both and evaluate by a combination likelihood score * logits.