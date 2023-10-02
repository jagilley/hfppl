from parsimonious.grammar import Grammar
from parsimonious.exceptions import ParseError
from hfppl import Model, CachedCausalLM, Token, LMContext, smc_standard, TokenCategorical, smc_steer
from hfppl.util import show_graph
import asyncio

def check_gbnf_conformance(gbnf, text):
    try:
        grammar = Grammar(gbnf)
        grammar.parse(text)
        return True
    except ParseError:
        return False

ts_gbnf = """root ::= CarAndOwner
Car ::= "{"   ws   "\"make\":"   ws   string   ","   ws   "\"model\":"   ws   string   ","   ws   "\"year\":"   ws   number   ","   ws   "\"colors\":"   ws   stringlist   "}"
Carlist ::= "[]" | "["   ws   Car   (","   ws   Car)*   "]"
CarAndOwner ::= "{"   ws   "\"car\":"   ws   Car   "}"
CarAndOwnerlist ::= "[]" | "["   ws   CarAndOwner   (","   ws   CarAndOwner)*   "]"
string ::= "\""   ([^"]*)   "\""
boolean ::= "true" | "false"
ws ::= [ \t\n]*
number ::= [0-9]+   "."?   [0-9]*
stringlist ::= "["   ws   "]" | "["   ws   string   (","   ws   string)*   ws   "]"
numberlist ::= "["   ws   "]" | "["   ws   string   (","   ws   number)*   ws   "]"
"""
    
class BNF(Model):
    def __init__(self, lm, prompt):
        super().__init__()
        self.lm = lm
        self.context = LMContext(lm, prompt)

    async def step(self):
        token = await self.sample(self.context.next_token(), proposal=self.proposal())

        self.condition(check_gbnf_conformance(ts_gbnf, str(self.context.s + str(token))))

        print(self.context.s + str(token))

        # Check for EOS or end of sentence
        if token.token_id == self.lm.tokenizer.eos_token_id or str(token) in ['.', '!', '?']:
            print("End of sentence")
            # Finish generation
            self.finish()

    def proposal(self):
        logits = self.context.next_token_logprobs.copy()
        return TokenCategorical(self.context.lm, logits)
    
lm = CachedCausalLM.from_pretrained("gpt2")

model = BNF(lm, '<|endoftext|>')

particles = asyncio.run(smc_steer(model, 5, 3))

# print the final particles and their associated weights
for particle in particles:
    print(f"{particle.context.s} (weight: {particle.weight})")

show_graph(lm)