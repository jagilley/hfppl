import json
from hfppl import Model, LMContext, TokenCategorical, CachedCausalLM
from hfppl.util import show_graph
from string import punctuation
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

with torch.no_grad():
    tok = AutoTokenizer.from_pretrained("gpt2")
    mod = AutoModelForCausalLM.from_pretrained("gpt2", do_sample=True, device_map="auto", load_in_8bit=True)

lm = CachedCausalLM.from_pretrained(mod, tok)
lm.batch_size = 40

# make forbidden tokens array of all tokens that are not valid JSON characters
forbidden_tokens = [i for (i, v) in enumerate(lm.vocab) if not all(c in '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ{}[]:," ' for c in v)]

class JsonModel(Model):
    def __init__(self, lm, prompt):
        super().__init__()
        self.lm = lm
        self.context = LMContext(lm, prompt)
        self.decay = 4 # you can go 4 tokens without generating valid JSON

    async def step(self):
        token = await self.sample(self.context.next_token(), proposal=await self.proposal())

        # condition on all characters being valid JSON characters
        # self.condition(all(c in '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ{}[]:," ' for c in str(token)))

        if self.decay == 0:
            print("Decay limit reached, culling branch")
            print(self.context.s + str(token))
            # Cull branch
            self.finish()

        print(self.context.s + str(token))

        # Check if the generated string is valid JSON
        try:
            json.loads(str(self.context.s + str(token)))
            self.decay = 4
        except json.JSONDecodeError:
            # If not, condition on the token containing only valid JSON characters
            self.decay -= 1
            self.condition(all(c in '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ{}[]:," ' for c in str(token)))

        # Check for EOS or end of sentence
        if token.token_id == self.lm.tokenizer.eos_token_id or str(token) in ['.', '!', '?']:
            print("End of sentence")
            # Finish generation
            self.finish()

    async def proposal(self):
        logits = self.context.next_token_logprobs.copy()
        # Set the log probabilities of forbidden tokens to -inf
        logits[forbidden_tokens] = float('-inf')
        return TokenCategorical(self.context.lm, logits)

# To run the model:

import asyncio
from hfppl import smc_steer

async def main():
    model = JsonModel(lm, '<|endoftext|>Here is a syntactically valid JSON object representing a car:\n{"car": {"make": "Honda", "model": "Civic", "year":')

    particles = await smc_steer(model, 10, 3)

    # print the final particles and their associated weights
    for particle in particles:
        print(f"{particle.context.s} (weight: {particle.weight})")

    show_graph(lm)

asyncio.run(main())