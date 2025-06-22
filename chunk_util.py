from typing import List, Dict

CHUNK_SIZE = 1000  # Maximum size of each text chunk, can be adjusted as needed
CHUNK_OVERLAP = 100  # Overlap size for text chunks, can be adjusted as needed

class ChunkUtil:

    @staticmethod
    def chunk_document(doc: Dict, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> List[Dict]:
        text = doc.get('text', "")
        if not text:
            return [doc]

        splits = ChunkUtil._recursive_split(text, chunk_size, chunk_overlap)
        
        chunks = []
        doc_id = doc.get("id", "unknown")
        for i, chunk_text in enumerate(splits):
            chunk = doc.copy()
            chunk["text"] = chunk_text
            chunk["chunk_id"] = f"{doc_id}_{i}"
            chunks.append(chunk)
        return chunks

    @staticmethod
    def _recursive_split(text: str, chunk_size: int, chunk_overlap: int, depth=0, max_depth=5) -> List[str]:
        separators = ["\n\n", "\n", ".", "!", "?", ",", " ", ""]  # from coarse to fine

        def split_on_separator(text, separator):
            if separator == "":
                # fallback: split by fixed length if no separator found
                return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
            parts = text.split(separator)
            # add separator back except for last part
            return [p + (separator if i < len(parts) - 1 else "") for i, p in enumerate(parts)]

        if depth > max_depth:
            return [text]

        for sep in separators:
            parts = split_on_separator(text, sep)
            chunks = []
            current_chunk = ""

            for part in parts:
                if len(current_chunk) + len(part) > chunk_size:
                    if current_chunk:
                        chunks.append(current_chunk)
                        current_chunk = current_chunk[-chunk_overlap:] + part
                    else:
                        current_chunk = part
                else:
                    current_chunk += part

            if current_chunk:
                chunks.append(current_chunk)

            # Only return if multiple chunks and all within chunk_size
            if len(chunks) > 1 and all(len(chunk) <= chunk_size for chunk in chunks):
                return chunks

        # If none of the separators worked to split into multiple chunks, recurse deeper or fallback
        new_chunks = []
        for chunk in [text]:
            if len(chunk) > chunk_size and depth < max_depth:
                new_chunks.extend(ChunkUtil._recursive_split(chunk, chunk_size, chunk_overlap, depth + 1, max_depth))
            else:
                new_chunks.append(chunk)

        return new_chunks