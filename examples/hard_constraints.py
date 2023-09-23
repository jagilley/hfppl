import string
import asyncio
from hfppl import Model, CachedCausalLM, Token, LMContext, smc_standard
from string import punctuation

# Load the language model. 
# Vicuna is an open model; to use a model with restricted access, like LLaMA 2,
# pass your HuggingFace API key as the optional `auth_token` argument:
#    LLM = CachedCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", auth_token=HF_AUTH_TOKEN)
LLM = CachedCausalLM.from_pretrained("lmsys/vicuna-7b-v1.5")
LLM.batch_size = 40

MASKS = {}
for i in range(6):
    MASKS[i] = set()
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
        # Skip if the length of v, after stripping leading and trailing whitespace, is more than 5
        if len(v.strip()) > 4:
            continue
        # Skip if the first character of v is alphabetic and i plus the length of v is more than 5
        if v[0].isalpha() and i + len(v) > 4:
            continue
        # If none of the conditions are met, add j to the set for the current i
        MASKS[i].add(j)

class ConstraintModel(Model):
    def __init__(self, prompt, max_tokens):
        super().__init__()
        self.context = LMContext(LLM, prompt)
        self.q       = LMContext(LLM, prompt)
        self.max_tokens = max_tokens

    async def step(self):
        # Which tokens are allowed?
        mask = self.active_constraint_mask()
        
        # Generate proposed token.
        token = await self.sample(self.context.next_token(), proposal=await self.proposal(mask))

        # Condition on constraint — a no-op since proposal already guarantees the constraint
        self.condition(token.token_id in mask)
        
        # Reduce number of max tokens remaining
        self.max_tokens -= 1
        
        print(f"{self.context}")

        # Check if done
        if token == LLM.tokenizer.eos_token_id or self.max_tokens == 0:
            self.finish()
            return
    
    def active_constraint_mask(self):
        string_so_far = str(self.context.s)
        words = string_so_far.split()
        last_word = words[-1] if len(words) > 0 else ""
        return MASKS[min(5, len(last_word))]
    
    async def proposal(self, mask):
        string_so_far = str(self.context)
        
        # Force the proposal LMContext to adhere to this mask
        await self.intervene(self.q.mask_dist(mask), True)
        
        # Return the proposal's modified next-token distribution
        return self.q.next_token()

        
# From Politico.com
prompt = """3 things to watch …

1. The return of the House means new energy for the GOP’s Biden impeachment push, and Democrats are starting their pushback early. Rep. Jamie Raskin (D-Md.) is out this morning with a 14-page rebuttal memo that seeks to paint the GOP campaign as a “complete and total bust” and an attempt at distracting from the “overwhelming evidence of [Trump’s] criminal and corrupt conduct during his term of office.”

2. The Senate is back this evening for a bed-check vote. With Minority Leader Mitch McConnell having successfully quieted (public) chatter about his health, expect senators to be quizzed anew about Sen. Tommy Tuberville’s (R-Ala.) Pentagon nominee blockade, especially with the Joint Chiefs chair, Gen. Mark Milley, just weeks away from retirement and the confirmation of his successor, Gen. C.Q. Brown, in limbo.

3."""

LLM.cache_kv(LLM.tokenizer.encode(prompt))

async def main():
    constraint_model = ConstraintModel(prompt, 50)
    particles = await smc_standard(constraint_model, 40)
    for p in particles:
        print(f"{p.context}")

asyncio.run(main())