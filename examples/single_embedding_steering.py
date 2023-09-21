import asyncio
from hfppl import Model, CachedCausalLM, Token, LMContext, smc_standard
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

LLM = CachedCausalLM.from_pretrained("gpt2")
LLM.batch_size = 40

emb_model = SentenceTransformer('thenlper/gte-base')

target_sentence = "Grr, I'm so angry right now :("
target_embed_1 = emb_model.encode(target_sentence)

class CosineSteeringModel2(Model):
    def __init__(self, prompt, max_tokens):
        super().__init__()
        self.lm         = LMContext(LLM, prompt)
        self.q          = LMContext(LLM, prompt)
        self.prompt_len = len(str(self.lm.s))
        self.max_tokens = max_tokens

    async def step(self):
        # Generate proposed token.
        token = await self.sample(self.lm.next_token(),proposal = await self.proposal())
        token_embed = emb_model.encode(str(token))
        sim = cos_sim(target_embed_1, token_embed)[0][0].item()
        diff = (sim - 0.5) * 100
        self.twist(diff)

        # Reduce number of max tokens remaining
        self.max_tokens -= 1

        print(str(self.lm.s)[self.prompt_len:], diff)

        # Check if done
        if token == LLM.tokenizer.eos_token_id or self.max_tokens == 0:
            self.finish()

    async def proposal(self):
        string_so_far = str(self.lm.s)

        # Return the proposal's modified next-token distribution
        return self.q.next_token()
    
prompt = "<|endoftext|>Today, I'm feeling very"

LLM.cache_kv(LLM.tokenizer.encode(prompt))

async def main():
    constraint_model = CosineSteeringModel2(prompt, 50)
    # constraint_model = ConstraintModel(prompt, 50)
    particles = await smc_standard(constraint_model, 20)
    for p in particles:
        print(str(p.lm.s)[p.prompt_len:])

# Run the model
asyncio.run(main())

"""
satisfied 25.88883638381958
bullish 26.007390022277832
guilty 26.325392723083496
lonely 27.52317786216736
confident 25.12357234954834
sick 25.914227962493896
tired 26.806026697158813
sick 25.914227962493896
good 24.812287092208862
relaxed 29.482483863830566
focused 25.90562105178833
good 24.812287092208862
strongly 27.22303867340088
guilty 26.325392723083496
excited 27.805256843566895
proud 26.67599320411682
worldly 26.754385232925415
positive 27.338671684265137
tired 26.806026697158813
religious 29.59536910057068
relaxed. 26.004862785339355
religious. 26.004862785339355
"""