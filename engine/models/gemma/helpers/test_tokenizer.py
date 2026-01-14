import sentencepiece as spm
import os

class GemmaTokenizer:
    def __init__(self, model_path: str):
        """
        Initialize the Gemma tokenizer.
        
        Args:
            model_path: Path to the tokenizer.model file
        """
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(model_path)
        
    @property
    def vocab_size(self):
        return self.sp.GetPieceSize()
    
    @property
    def bos_id(self):
        return self.sp.bos_id()
    
    @property
    def eos_id(self):
        return self.sp.eos_id()
    
    @property
    def pad_id(self):
        return self.sp.pad_id()
    
    def encode(self, text: str, add_bos: bool = True, add_eos: bool = False):
        """
        Encode text to token IDs.
        
        Args:
            text: Input text string
            add_bos: Whether to add beginning-of-sequence token
            add_eos: Whether to add end-of-sequence token
            
        Returns:
            List of token IDs
        """
        ids = self.sp.EncodeAsIds(text)
        
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
            
        return ids
    
    def decode(self, ids: list):
        """
        Decode token IDs to text.
        
        Args:
            ids: List of token IDs
            
        Returns:
            Decoded text string
        """
        return self.sp.DecodeIds(ids)
    
    def id_to_piece(self, id: int):
        """Get the token string for a given ID."""
        return self.sp.IdToPiece(id)
    
    def piece_to_id(self, piece: str):
        """Get the ID for a given token string."""
        return self.sp.PieceToId(piece)


if __name__ == "__main__":
    tokenizer_path = os.path.join(os.path.dirname(__file__), "gemma-3-270m", "tokenizer.model")
    
    tokenizer = GemmaTokenizer(tokenizer_path)
    
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"BOS ID: {tokenizer.bos_id}")
    print(f"EOS ID: {tokenizer.eos_id}")
    print(f"PAD ID: {tokenizer.pad_id}")
    
    test_text = "Hello, world! This is a test."
    print(f"\nOriginal text: {test_text}")
    
    tokens = tokenizer.encode(test_text)
    print(f"Encoded tokens: {tokens}")
    print(f"Number of tokens: {len(tokens)}")
    
    # Show first few token pieces
    print(f"\nFirst 10 tokens as strings:")
    for i, token_id in enumerate(tokens[:10]):
        piece = tokenizer.id_to_piece(token_id)
        print(f"  {i}: {token_id} -> '{piece}'")
    
    decoded_text = tokenizer.decode(tokens)
    print(f"\nDecoded text: {decoded_text}")
    
    # Test another example
    print("\n" + "="*60)
    test_text2 = "The quick brown fox jumps over the lazy dog."
    print(f"Text: {test_text2}")
    tokens2 = tokenizer.encode(test_text2, add_bos=True, add_eos=False)
    print(f"Tokens: {tokens2}")
    print(f"Token count: {len(tokens2)}")
    decoded2 = tokenizer.decode(tokens2)
    print(f"Decoded: {decoded2}")
