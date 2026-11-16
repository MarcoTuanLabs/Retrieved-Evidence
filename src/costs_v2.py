"""Dense linear/attention FLOP estimate with explicit query-head width.

Qwen3 head_dim * num_attention_heads need not equal hidden_size. This revision
corrects attention matmul accounting, while retaining the disclosed dense
upper convention rather than pretending to measure hardware kernel FLOPs.
"""
from dataclasses import dataclass

@dataclass(frozen=True)
class DenseForwardCost:
    transformer_parameters:int
    head_parameters:int
    layers:int
    hidden_size:int
    attention_width:int

    def __post_init__(self):
        assert all(isinstance(x,int) and x>0 for x in (self.transformer_parameters,self.head_parameters,self.layers,self.hidden_size,self.attention_width))

    def incremental(self,new_tokens,past_tokens=0):
        assert new_tokens>=1 and past_tokens>=0
        return 2*self.transformer_parameters*new_tokens+4*self.layers*self.attention_width*new_tokens*(past_tokens+new_tokens)+2*self.head_parameters

    def prefill(self,tokens):return self.incremental(tokens)

    def generation(self,prompt_tokens,generated_tokens):
        extra=max(generated_tokens-1,0)
        return (self.prefill(prompt_tokens)+2*(self.transformer_parameters+self.head_parameters)*extra
            +4*self.layers*self.attention_width*(extra*prompt_tokens+extra*(extra+1)/2))

    def recheck(self,prompt_tokens,generated_tokens,score_tokens):
        return self.generation(prompt_tokens,generated_tokens)+self.prefill(score_tokens)

    @classmethod
    def from_config_total(cls,config,total):
        hidden=config['hidden_size'];head=config['vocab_size']*hidden
        tied=config.get('tie_word_embeddings',False)
        body=total-head-(0 if tied else head)
        width=config.get('head_dim',hidden//config['num_attention_heads'])*config['num_attention_heads']
        return cls(body,head,config['num_hidden_layers'],hidden,width)
