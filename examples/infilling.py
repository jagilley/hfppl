import asyncio
from hfppl import Model, CachedCausalLM, Token, LMContext, smc_standard, Geometric

LLM = CachedCausalLM.from_pretrained("gpt2")
LLM.batch_size = 40

class Infilling(Model):
    def __init__(self, words):
        super().__init__()
        firstword = words.pop(0)
        self.context = LMContext(LLM, "<|endoftext|>" + firstword)
        self.generation = 0
        self.s = firstword
        self.tokenized_words = [LLM.tokenizer.encode(word) for word in words]
    
    async def step(self):
        n = await self.sample(Geometric(0.5)) + 1
        # generate n tokens
        for _ in range(n):
            self.s += await self.sample(self.context.next_token())

        # add the next token from the tokenized words
        for token in self.tokenized_words.pop(0):
            self.s += await self.observe(self.context.next_token(), token)

        print(str(self.s))
        self.generation += 1

        if len(self.tokenized_words) == 0:
            self.observe(self.context.next_token(), LLM.tokenizer.eos_token_id)
            self.finish()
    
    def proposal(self):
        return self.context.next_token()
    
async def main():
    words = ["Well, you see, every", " he", " to", " another", "!"]
    model = Infilling(words)
    particles = await smc_standard(model, 20)

if __name__=="__main__":
    asyncio.run(main())