import asyncio
from hfppl import Model, CachedCausalLM, Token, LMContext, smc_standard
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

LLM = CachedCausalLM.from_pretrained("gpt2")
LLM.batch_size = 40

emb_model = SentenceTransformer('thenlper/gte-base')

target_sentence = "Grr, I'm so angry right now :("
target_embed_1 = emb_model.encode(target_sentence)
t2 = "Yay, I'm so happy right now :)"
target_embed_2 = emb_model.encode(t2)

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
        sim1 = cos_sim(target_embed_1, token_embed)[0][0].item()
        sim2 = cos_sim(target_embed_2, token_embed)[0][0].item()
        diff = (sim2 - sim1) * 1000 # sim2 - sim1 for positive sentiment, sim1 - sim2 for negative sentiment
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

# Examples of weights after this is run in negative sentiment mode:
"""
guilty 16.538679599761963
sad 21.520018577575684
honored -2.478301525115967
excited 15.129446983337402
fortunate 18.187284469604492
sad 21.520018577575684
lucky 27.285993099212646 # odd that this scores so highly!
good -25.420784950256348
grateful -12.510061264038086
tired 3.169834613800049
angry 24.075865745544434
bullish 5.2909255027771
happy -66.2958025932312
special -4.4966936111450195
dis 8.99726152420044
stressed 9.49770212173462
good -25.420784950256348
nervous 5.048990249633789
stressed 9.49770212173462
grateful -12.510061264038086
"""