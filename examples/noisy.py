# WIP
import asyncio
from hfppl import Model, CachedCausalLM, Token, LMContext, smc_standard, TokenCategorical
import numpy as np
import re
from Levenshtein import distance as lev

LLM = CachedCausalLM.from_pretrained("gpt2")
LLM.batch_size = 40

def is_similar_levenshtein(threshold=1):
    """Make predicate for whether Levenshtein distance is (at or) below threshold"""

    def f(s, token):
        if isinstance(token, Token):
            token = str(token)
        if isinstance(s, Token):
            s = str(s)
        if isinstance(s, list):
            s = "".join(str(x) for x in s)
        # if s != '' and s[0] != token[0]:
        # return False
        if lev(token, s) <= threshold:
            return True
        else:
            return False

    return f


def split_before_whitespace(string):
    """
    Simple whitespace-preserving word-tokenizer: splits string every location
    that follows a non-whitespace char and precedes a whitespace char.
    E.g.: 'Here, an example.' -> ['Here,', ' an', ' example.']
    """
    return re.split(r"(?<=\S)(?=\s)", string)


class NoisyTokenModel(Model):
    """
    Simple model of noisy observation (by token).
    Samples continuations of `prompt` conditioning on each successive
    token being similar to the corresponding token from `string` input, using
    similarity function `is_similar`.
    """

    def __init__(self, prompt, string, is_similar=is_similar_levenshtein(1)):
        super().__init__()
        self.s = prompt
        self.ctx = self.new_context(self.s)
        self.remaining_tokens = self.llama.tokenize(string)
        self.is_similar = is_similar

    def step(self):
        # Actual next token
        true_token = self.remaining_tokens.pop(0)

        # Sample noisy version that is similar
        sampled_token = self.sample(
            llp.Transformer(self.ctx), proposal=self.proposal(true_token)
        )

        # Condition on constraint (not necessary if proposal is correct)
        # self.condition(self.is_similar(true_token, sampled_token))

        # Update generated string
        self.s += sampled_token

        # Check if done
        if len(self.remaining_tokens) == 0:
            self.observe(llp.Transformer(self.ctx), LLM.tokenizer.eos_token_id)
            self.finish()
            return

    def proposal(self, true_token):
        lm_logits = self.ctx.logits()
        mask = np.array(
            [0.0 if self.is_similar(true_token, v) else -np.inf for v in self.vocab()]
        )
        logprobs = llp.lognormalize(lm_logits + mask)
        return TokenCategorical(logprobs)


class NoisyWordModel(Model):
    """
    Noisy observations of whole words rather than single tokens.
    TODO: allow noisy word to be different number of tokens than observation.
    """

    def __init__(self, prompt, string, is_similar):
        super().__init__()
        self.s = prompt
        self.ctx = self.new_context(self.s)
        self.remaining_words = split_before_whitespace(string)
        self.is_similar = is_similar

    def step(self):
        true_word = self.remaining_words.pop(0)
        true_tokens = self.llama.tokenize(true_word)

        for i, tok in enumerate(true_tokens):
            self.s += self.sample(
                llp.Transformer(self.ctx), proposal=self.proposal(tok)
            )

        if len(self.remaining_words) == 0:
            self.observe(llp.Transformer(self.ctx), LLM.tokenizer.eos_token_id)
            self.finish()
            return

    def proposal(self, true_token):
        lm_logits = self.ctx.logits()
        mask = np.array(
            [0.0 if self.is_similar(true_token, v) else -np.inf for v in self.vocab()]
        )
        logprobs = llp.lognormalize(lm_logits + mask)
        return TokenCategorical(logprobs)