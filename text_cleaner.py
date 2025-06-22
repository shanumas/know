import codecs, re
class TextCleaner:
    @staticmethod
    def clean_extracted_text(text: str, truncate_incomplete=True) -> str:
            # Decode unicode escape characters (e.g., \u201c to “)
            text = codecs.decode(text, 'unicode_escape')

            # Replace \n with space
            text = text.replace('\n', ' ').replace('\\n', ' ')

            # Collapse multiple spaces
            text = re.sub(r'\s+', ' ', text).strip()

            # Optionally truncate at the last full stop to remove incomplete endings
            if truncate_incomplete and '.' in text:
                last_dot = text.rfind('.')
                text = text[:last_dot + 1]

            return text