import re

class TextCleaner:
    @staticmethod
    def clean_extracted_text(text: str, truncate_incomplete=True) -> str:
        # Fix mojibake (bad UTF-8 rendering)
        try:
            text = text.encode('latin1').decode('utf-8', errors='replace')
        except Exception:
            pass  # fallback if input is already clean

        # Replace \n and \\n with space
        text = text.replace('\n', ' ').replace('\\n', ' ')

        # Collapse multiple spaces
        text = re.sub(r'\s+', ' ', text).strip()

        # Truncate at last full stop
        if truncate_incomplete and '.' in text:
            last_dot = text.rfind('.')
            text = text[:last_dot + 1]

        return text
