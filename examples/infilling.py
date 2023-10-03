import asyncio
from hfppl import Model, CachedCausalLM, Token, LMContext, smc_standard, Geometric
from hfppl.util import show_graph
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

with torch.no_grad():
    tok = AutoTokenizer.from_pretrained("gpt2")
    mod = AutoModelForCausalLM.from_pretrained("gpt2", do_sample=True, device_map="auto", load_in_8bit=True)

LLM = CachedCausalLM.from_pretrained(mod, tok)
LLM.batch_size = 40

class Infilling(Model):
    def __init__(self, words):
        super().__init__()
        firstword = words.pop(0)
        self.context = LMContext(LLM, "<|endoftext|>" + firstword)
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

        if len(self.tokenized_words) == 0:
            await self.observe(self.context.next_token(), LLM.tokenizer.eos_token_id)
            self.finish()
    
    def proposal(self):
        return self.context.next_token()
    
async def main():
    words = ["If you're in", ", pivot to", "."]
    model = Infilling(words)
    particles = await smc_standard(model, 20)

    show_graph(LLM)

if __name__=="__main__":
    asyncio.run(main())

"""
If you're in high school, pivot to
If you're in Toronto,, pivot to
If you're in an area, pivot to
If you're in the mood, pivot to
If you're in a little, pivot to
If you're in Canada and, pivot to
If you're in a hurry, pivot to
If you're in a factory, pivot to
If you're in Washington,, pivot to
If you're in Croatia or, pivot to
If you're in the midst, pivot to
If you're in New York, pivot to
If you're in North Carolina, pivot to
If you're in Pennsylvania, you, pivot to
If you're in desperate need of, pivot to
If you're in the market for, pivot to
If you're in agreement that consuming, pivot to
If you're in need of Real, pivot to
If you're in any doubt, either, pivot to
If you're in business, etiquette may contain some fairly big ling, pivot to
If you're in a hurry, pivot to one of.
If you're in a hurry, pivot to the original.
If you're in a hurry, pivot to MakeUse.
If you're in a hurry, pivot to a major.
If you're in high school, pivot to Google News.
If you're in a hurry, pivot to a secondary.
If you're in a factory, pivot to the safety.
If you're in a hurry, pivot to this post.
If you're in a hurry, pivot to your keyboard.
If you're in North Carolina, pivot to 4chan. # this one's pretty good lmao
If you're in North Carolina, pivot to Fox News.
If you're in a hurry, pivot to the safer.
If you're in a hurry, pivot to the newcomer list.
If you're in North Carolina, pivot to AA & PP.
If you're in a factory, pivot to the most convenient and.
If you're in an area, pivot to the SATA3 (likely.
If you're in a hurry, pivot toset your permit so that.
If you're in North Carolina, pivot to States One and You Cause.
If you're in a hurry, pivot to a world that is light on pro-LGBT.
If you're in a hurry, pivot to downtrendlets (up now) or downt.
"""