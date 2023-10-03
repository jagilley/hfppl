from hfppl  import Model, CachedCausalLM, Token, LMContext, smc_standard
from string import punctuation
import asyncio
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

with torch.no_grad():
    tok = AutoTokenizer.from_pretrained("gpt2")
    mod = AutoModelForCausalLM.from_pretrained("gpt2", do_sample=True, device_map="auto", load_in_8bit=True)

LLM = CachedCausalLM.from_pretrained(mod, tok)
LLM.batch_size = 40

MASKS = set()
for j, v in enumerate(LLM.vocab):
    # Skip if j is the end-of-sentence token id
    if j == LLM.tokenizer.eos_token_id:
        continue
    # Skip if v contains a newline character
    if '\n' in v:
        continue
    # Skip if v does not contain an alphabetic character or punctuation
    if not any(c.isalpha() or c in punctuation for c in v):
        continue
    # Skip if v contains the character "e"
    if 'e' in v or 'E' in v:
        continue
    # If none of the conditions are met, add j to the set
    MASKS.add(j)

class ConstraintModel(Model):
    def __init__(self, prompt, max_tokens):
        super().__init__()
        self.lm         = LMContext(LLM, prompt)
        self.q          = LMContext(LLM, prompt)
        self.prompt_len = len(str(self.lm.s))
        self.max_tokens = max_tokens

    async def step(self):
        # Which tokens are allowed?
        mask = self.active_constraint_mask()

        # Generate proposed token.
        token = await self.sample(self.lm.next_token(),
                                  proposal = await self.proposal(mask))

        # Condition on constraint — a no-op since proposal already guarantees the constraint
        self.condition(token.token_id in mask)

        # Reduce number of max tokens remaining
        self.max_tokens -= 1

        print(str(self.lm.s)[self.prompt_len:])

        # Check if done
        if token == LLM.tokenizer.eos_token_id or self.max_tokens == 0:
            self.finish()

    def active_constraint_mask(self):
        return MASKS

    async def proposal(self, mask):
        string_so_far = str(self.lm.s)

        # Force the proposal StatefulLM to adhere to this mask
        await self.intervene(self.q.mask_dist(mask), True)

        # Return the proposal's modified next-token distribution
        return self.q.next_token()
    
if __name__=="__main__":
    # From Politico.com
    prompt = """<|endoftext|>3 things to watch …

    1. The return of the House means new energy for the GOP’s Biden impeachment push, and Democrats are starting their pushback early. Rep. Jamie Raskin (D-Md.) is out this morning with a 14-page rebuttal memo that seeks to paint the GOP campaign as a “complete and total bust” and an attempt at distracting from the “overwhelming evidence of [Trump’s] criminal and corrupt conduct during his term of office.”

    2. The Senate is back this evening for a bed-check vote. With Minority Leader Mitch McConnell having successfully quieted (public) chatter about his health, expect senators to be quizzed anew about Sen. Tommy Tuberville’s (R-Ala.) Pentagon nominee blockade, especially with the Joint Chiefs chair, Gen. Mark Milley, just weeks away from retirement and the confirmation of his successor, Gen. C.Q. Brown, in limbo.

    3."""

    LLM.cache_kv(LLM.tokenizer.encode(prompt))

    async def main():
        constraint_model = ConstraintModel(prompt, 50)
        particles = await smc_standard(constraint_model, 20)
        for p in particles:
            print(str(p.lm.s)[p.prompt_len:])
    
    asyncio.run(main())