import re
from typing import List, Dict, Tuple
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import tiktoken


class TextChunker:
    def __init__(self):
        # Use tiktoken for token counting (OpenAI's tokenizer which is widely used)
        self.tokenizer = tiktoken.get_encoding("cl100k_base")  # Good for code and text

    def chunk_text(
        self,
        text: str,
        max_tokens: int = 1000,
        min_tokens: int = 500,
        overlap_tokens: int = 100,
        preserve_code_blocks: bool = True
    ) -> List[Dict]:
        """
        Intelligent text chunker that:
        - Maintains 500-1000 tokens per chunk
        - Uses 50-100 token overlap
        - Splits on headers
        - Preserves code blocks
        """
        chunks = []

        if preserve_code_blocks:
            # First, separate code blocks from text to preserve them
            parts = self._separate_code_blocks(text)
        else:
            parts = [{'type': 'text', 'content': text}]

        for part in parts:
            if part['type'] == 'code':
                # Code blocks are kept as single chunks if they're not too large
                if self._count_tokens(part['content']) <= max_tokens:
                    chunks.append({
                        'content': part['content'],
                        'token_count': self._count_tokens(part['content']),
                        'type': 'code',
                        'metadata': {'preserved_code_block': True}
                    })
                else:
                    # If code block is too large, we'll need to chunk it anyway
                    code_chunks = self._chunk_large_code_block(
                        part['content'],
                        max_tokens,
                        overlap_tokens
                    )
                    chunks.extend(code_chunks)
            else:  # text type
                text_chunks = self._chunk_text_content(
                    part['content'],
                    max_tokens,
                    min_tokens,
                    overlap_tokens
                )
                chunks.extend(text_chunks)

        return chunks

    def _separate_code_blocks(self, text: str) -> List[Dict]:
        """
        Separate text content from code blocks to preserve code blocks during chunking
        """
        parts = []
        lines = text.split('\n')
        current_text = []
        i = 0

        while i < len(lines):
            line = lines[i]

            # Check if this line starts a code block
            if line.strip().startswith('```'):
                # Add any accumulated text before the code block
                if current_text:
                    parts.append({
                        'type': 'text',
                        'content': '\n'.join(current_text).strip()
                    })
                    current_text = []

                # Find the end of the code block
                code_block = [line]
                i += 1
                while i < len(lines):
                    code_block.append(lines[i])
                    if lines[i].strip().startswith('```') and i > 0 and not lines[i-1].strip().startswith('\\```'):
                        break
                    i += 1

                if len(code_block) > 1:  # Make sure we have a complete code block
                    parts.append({
                        'type': 'code',
                        'content': '\n'.join(code_block).strip()
                    })
            else:
                current_text.append(line)

            i += 1

        # Add any remaining text
        if current_text:
            parts.append({
                'type': 'text',
                'content': '\n'.join(current_text).strip()
            })

        return parts

    def _chunk_text_content(
        self,
        text: str,
        max_tokens: int,
        min_tokens: int,
        overlap_tokens: int
    ) -> List[Dict]:
        """
        Chunk text content based on headers and token count
        """
        # First, split by headers to maintain document structure
        sections = self._split_by_headers(text)

        chunks = []
        current_chunk = ""
        current_token_count = 0
        current_headers = []

        for section in sections:
            section_text = section['header'] + '\n' + section['content'] if section['header'] else section['content']
            section_tokens = self._count_tokens(section_text)

            # If the section itself is too large, we need to split it further
            if section_tokens > max_tokens:
                # Split the large section into smaller pieces
                sub_chunks = self._split_large_section(section_text, max_tokens, overlap_tokens)
                for sub_chunk in sub_chunks:
                    chunks.append({
                        'content': sub_chunk,
                        'token_count': self._count_tokens(sub_chunk),
                        'type': 'text',
                        'metadata': {'headers': section.get('headers', [])}
                    })
            else:
                # Check if adding this section would exceed max tokens
                if current_token_count + section_tokens <= max_tokens:
                    current_chunk += section_text + '\n'
                    current_token_count += section_tokens
                    if section['header']:
                        current_headers.append(section['header'])
                else:
                    # If the current chunk is already substantial (>= min_tokens), save it
                    if current_token_count >= min_tokens:
                        chunks.append({
                            'content': current_chunk.strip(),
                            'token_count': current_token_count,
                            'type': 'text',
                            'metadata': {'headers': current_headers.copy()}
                        })
                        # Start new chunk with overlap
                        if chunks:
                            # Add overlap by including part of the previous chunk
                            overlap_chunk = self._create_overlap_chunk(
                                current_chunk,
                                overlap_tokens,
                                section_text
                            )
                            current_chunk = overlap_chunk + section_text + '\n'
                            current_token_count = self._count_tokens(current_chunk)
                            current_headers = [section['header']] if section['header'] else []
                        else:
                            current_chunk = section_text + '\n'
                            current_token_count = section_tokens
                            current_headers = [section['header']] if section['header'] else []
                    else:
                        # If the current chunk is small, add the section anyway
                        current_chunk += section_text + '\n'
                        current_token_count += section_tokens
                        if section['header']:
                            current_headers.append(section['header'])

        # Add the final chunk if it has content
        if current_chunk.strip():
            chunks.append({
                'content': current_chunk.strip(),
                'token_count': current_token_count,
                'type': 'text',
                'metadata': {'headers': current_headers}
            })

        return chunks

    def _split_by_headers(self, text: str) -> List[Dict]:
        """
        Split text by markdown headers to maintain document structure
        """
        lines = text.split('\n')
        sections = []
        current_section_content = []
        current_header = ""

        for line in lines:
            # Check if this line is a header
            header_match = re.match(r'^(#{1,6})\s+(.+)$', line)
            if header_match:
                # Save the previous section if it has content
                if current_section_content or current_header:
                    sections.append({
                        'header': current_header,
                        'content': '\n'.join(current_section_content).strip()
                    })

                # Start new section with the header
                current_header = line
                current_section_content = []
            else:
                current_section_content.append(line)

        # Add the last section
        if current_section_content or current_header:
            sections.append({
                'header': current_header,
                'content': '\n'.join(current_section_content).strip()
            })

        return sections

    def _split_large_section(self, section_text: str, max_tokens: int, overlap_tokens: int) -> List[str]:
        """
        Split a large section that exceeds max tokens into smaller pieces
        """
        sentences = re.split(r'(?<=[.!?])\s+', section_text)
        chunks = []
        current_chunk = ""
        current_token_count = 0

        for sentence in sentences:
            sentence_token_count = self._count_tokens(sentence)

            if current_token_count + sentence_token_count <= max_tokens:
                current_chunk += sentence + " "
                current_token_count += sentence_token_count
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())

                # Create overlap by including part of the previous sentence
                if chunks and overlap_tokens > 0:
                    prev_chunk = chunks[-1]
                    overlap_part = self._get_overlap_part(prev_chunk, overlap_tokens)
                    current_chunk = overlap_part + sentence + " "
                else:
                    current_chunk = sentence + " "

                current_token_count = self._count_tokens(current_chunk)

        # Add the final chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks

    def _create_overlap_chunk(self, previous_chunk: str, overlap_tokens: int, new_content: str) -> str:
        """
        Create an overlap by taking the end portion of the previous chunk
        """
        tokens = self.tokenizer.encode(previous_chunk)
        if len(tokens) <= overlap_tokens:
            return previous_chunk
        else:
            overlap_tokens_slice = tokens[-overlap_tokens:]
            overlap_text = self.tokenizer.decode(overlap_tokens_slice)
            return overlap_text

    def _get_overlap_part(self, text: str, overlap_tokens: int) -> str:
        """
        Get the last N tokens from text as a string
        """
        tokens = self.tokenizer.encode(text)
        if len(tokens) <= overlap_tokens:
            return text
        else:
            overlap_tokens_slice = tokens[-overlap_tokens:]
            overlap_text = self.tokenizer.decode(overlap_tokens_slice)
            return overlap_text

    def _chunk_large_code_block(self, code: str, max_tokens: int, overlap_tokens: int) -> List[Dict]:
        """
        Chunk a large code block that exceeds max tokens
        """
        lines = code.split('\n')
        chunks = []
        current_chunk = "```\n"  # Start with opening code block
        current_token_count = self._count_tokens(current_chunk)

        i = 1  # Skip the first '```'
        while i < len(lines):
            line = lines[i]
            line_with_newline = line + '\n'
            line_tokens = self._count_tokens(line_with_newline)

            if current_token_count + line_tokens <= max_tokens:
                current_chunk += line_with_newline
                current_token_count += line_tokens
            else:
                # Close the current code block and save it
                current_chunk += "```"
                chunks.append({
                    'content': current_chunk,
                    'token_count': self._count_tokens(current_chunk),
                    'type': 'code',
                    'metadata': {'preserved_code_block': True, 'is_chunked_code': True}
                })

                # Start new chunk with overlap
                if overlap_tokens > 0 and len(chunks) > 0:
                    prev_chunk = chunks[-1]['content']
                    overlap_part = self._get_overlap_part(prev_chunk, overlap_tokens)
                    current_chunk = "```\n" + overlap_part + line_with_newline
                else:
                    current_chunk = "```\n" + line_with_newline

                current_token_count = self._count_tokens(current_chunk)

            i += 1

        # Add the final code block if it has content
        if len(current_chunk) > 3:  # More than just '```'
            if not current_chunk.endswith('```'):
                current_chunk += '```'
            chunks.append({
                'content': current_chunk,
                'token_count': self._count_tokens(current_chunk),
                'type': 'code',
                'metadata': {'preserved_code_block': True, 'is_chunked_code': True}
            })

        return chunks

    def _count_tokens(self, text: str) -> int:
        """
        Count tokens in text using tiktoken
        """
        return len(self.tokenizer.encode(text))


# Global instance
text_chunker = TextChunker()