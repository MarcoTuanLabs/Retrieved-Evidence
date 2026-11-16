"""Common serving candidate: batch tokenization, unchanged semantic windows.

The original frozen implementation stays available for exact parity checks.
This optimization changes no evidence policy and must be shared by all methods.
"""
from semantic_paragraph_baseline import SemanticParagraphRanker


class BatchedSemanticParagraphRanker(SemanticParagraphRanker):
    def windows(self, question, paragraphs):
        tok = self.tokenizer
        # One Rust tokenizer batch per request; no other question is in this batch.
        texts = [question] + [p['section']+'\n'+p['text'] for p in paragraphs]
        encoded = tok(texts, add_special_tokens=False, padding=False,
                      truncation=False, return_attention_mask=False,
                      return_token_type_ids=False, verbose=False)['input_ids']
        raw_q, paragraph_ids = encoded[0], encoded[1:]
        q = raw_q[:64]
        capacity = 512-len(q)-tok.num_special_tokens_to_add(pair=True)
        overlap = 64
        assert capacity > overlap
        pairs, owners, lengths = [], [], []
        for paragraph, ids in zip(paragraphs, paragraph_ids):
            lengths.append(len(ids))
            offset = 0
            while True:
                piece = ids[offset:offset+capacity]
                # Keep the exact frozen special-token and mask construction.
                pair = tok.prepare_for_model(q, piece, add_special_tokens=True,
                                             return_attention_mask=True,
                                             truncation=False)
                assert len(pair['input_ids']) <= 512
                pairs.append(pair)
                owners.append({'paragraph_id': paragraph['id'], 'start': offset,
                               'end': offset+len(piece)})
                if offset+capacity >= len(ids):
                    break
                offset += capacity-overlap
        return pairs, owners, {'question_tokens': len(raw_q),
            'question_truncated': len(raw_q)>64, 'paragraph_tokens': lengths,
            'max_pair_tokens': 512, 'overlap': overlap}
