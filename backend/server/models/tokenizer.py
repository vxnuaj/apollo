import sentencepiece as spm
import os

class SentencePieceTokenizer:
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
    
class GemmaTokenizer(SentencePieceTokenizer):
    def __init__(self, model_path:str):
        super().__init__(model_path)
        
        