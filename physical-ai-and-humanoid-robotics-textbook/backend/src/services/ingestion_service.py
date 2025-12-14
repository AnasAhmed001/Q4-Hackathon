import os
import re
import frontmatter
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import time
from ..utils.text_chunker import text_chunker
from ..services.embedding_service import embedding_service
from ..services.qdrant_service import qdrant_service
from ..services.database_service import SessionLocal, db_ops, DocumentCreate
from sqlalchemy.orm import Session
from uuid import uuid4


# ============================================================================
# Markdown Parser
# ============================================================================

class MarkdownParser:
    def __init__(self):
        pass

    def parse_docusaurus_file(self, file_path: str) -> Dict:
        """
        Parse a Docusaurus markdown file and extract content with metadata
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Parse frontmatter
        post = frontmatter.loads(content)

        # Extract main content without frontmatter
        main_content = post.content

        # Extract title from frontmatter or from first H1
        title = post.get('title', self._extract_title_from_content(main_content))

        # Extract module and section from path
        path_parts = Path(file_path).parts
        module = self._extract_module_name(path_parts)
        section = self._extract_section_name(path_parts, title)

        # Extract headers and content structure
        headers = self._extract_headers(main_content)

        # Create document structure
        document = {
            'title': title,
            'module': module,
            'section': section,
            'url': self._generate_url(path_parts),
            'frontmatter': dict(post.metadata),
            'headers': headers,
            'content': main_content,
            'raw_content': content
        }

        return document

    def _extract_title_from_content(self, content: str) -> str:
        """Extract title from the first H1 header if not in frontmatter"""
        h1_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        if h1_match:
            return h1_match.group(1).strip()
        return "Untitled Document"

    def _extract_module_name(self, path_parts: Tuple[str, ...]) -> str:
        """Extract module name from file path"""
        for part in path_parts:
            if 'module' in part.lower() or 'intro' in part.lower() or 'capstone' in part.lower():
                return part
        return 'unknown_module'

    def _extract_section_name(self, path_parts: Tuple[str, ...], title: str) -> str:
        """Extract section name from file path or title"""
        filename = Path(path_parts[-1]).stem
        if filename.lower() in ['index', 'readme']:
            return title
        return filename.replace('-', ' ').title()

    def _generate_url(self, path_parts: Tuple[str, ...]) -> str:
        """Generate URL from file path"""
        docs_index = -1
        for i, part in enumerate(path_parts):
            if part == 'docs':
                docs_index = i
                break

        if docs_index != -1:
            url_parts = path_parts[docs_index:]
            url = '/'.join(url_parts[:-1])
            filename = Path(url_parts[-1]).stem
            if filename != 'index':
                url = f"{url}/{filename}" if url else filename
            return f"/{url}"
        else:
            return f"/{Path(path_parts[-1]).stem}"

    def _extract_headers(self, content: str) -> List[Dict]:
        """Extract headers and their positions in the content"""
        headers = []
        lines = content.split('\n')

        for i, line in enumerate(lines):
            header_match = re.match(r'^(#{1,6})\s+(.+)$', line)
            if header_match:
                level = len(header_match.group(1))
                title = header_match.group(2).strip()
                headers.append({
                    'level': level,
                    'title': title,
                    'line_number': i,
                    'content': title
                })

        return headers


# Global markdown parser instance
markdown_parser = MarkdownParser()


# ============================================================================
# Ingestion Service
# ============================================================================

class IngestionService:
    def __init__(self):
        self.qdrant_service = qdrant_service
        self.embedding_service = embedding_service
        self.markdown_parser = markdown_parser

    def parse_and_chunk_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Parse a file and chunk it using the markdown parser and text chunker
        """
        # Parse the file using the markdown parser
        parsed_document = self.markdown_parser.parse_docusaurus_file(file_path)

        # Chunk the content using the text chunker
        chunks = text_chunker.chunk_text(
            parsed_document['content'],
            max_tokens=1000,
            min_tokens=500,
            overlap_tokens=100,
            preserve_code_blocks=True
        )

        # Format chunks for ingestion
        formatted_chunks = []
        for i, chunk in enumerate(chunks):
            formatted_chunks.append({
                'id': str(uuid4()),  # Generate unique ID for the chunk
                'content': chunk['content'],
                'module': parsed_document['module'],
                'section': parsed_document['section'],
                'url': parsed_document['url'],
                'title': parsed_document['title'],
                'metadata': {
                    'original_file': file_path,
                    'chunk_index': i,
                    'token_count': chunk['token_count'],
                    'type': chunk['type']
                }
            })

        return formatted_chunks

    def process_document_for_ingestion(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Complete parse→chunk→embed→store workflow for a single document
        """
        # Step 1: Parse and chunk the file
        chunks = self.parse_and_chunk_file(file_path)

        # Step 2: Generate embeddings for all chunks
        texts_to_embed = [chunk['content'] for chunk in chunks]
        embeddings = self.embedding_service.embed_batch(
            texts_to_embed,
            input_type="search_document"
        )

        # Step 3: Format for Qdrant storage
        qdrant_points = []
        for chunk, embedding in zip(chunks, embeddings):
            qdrant_points.append({
                'id': chunk['id'],
                'vector': embedding,
                'payload': {
                    'content': chunk['content'],
                    'module': chunk['module'],
                    'section': chunk['section'],
                    'url': chunk['url'],
                    'title': chunk['title'],
                    'metadata': chunk['metadata']
                }
            })

        return qdrant_points

    def ingest_file(self, file_path: str, check_duplicates: bool = True) -> Dict[str, Any]:
        """
        Ingest a single file with duplicate detection
        """
        try:
            # Check for duplicates if requested
            if check_duplicates:
                db: Session = SessionLocal()
                try:
                    existing_doc = db_ops.get_document_by_url(db, url=self.markdown_parser.parse_docusaurus_file(file_path)['url'])
                    if existing_doc:
                        return {
                            'status': 'skipped',
                            'message': f'Duplicate document found: {file_path}',
                            'url': existing_doc.url
                        }
                finally:
                    db.close()

            # Process the document
            qdrant_points = self.process_document_for_ingestion(file_path)

            # Store in Qdrant
            self.qdrant_service.upsert_vectors(qdrant_points)

            # Store document metadata in database
            db: Session = SessionLocal()
            try:
                parsed_doc = self.markdown_parser.parse_docusaurus_file(file_path)

                # Use frontmatter if available, otherwise empty dict
                metadata = parsed_doc.get('frontmatter', {})
                
                doc_in = DocumentCreate(
                    url=parsed_doc['url'],
                    title=parsed_doc['title'],
                    module=parsed_doc['module'],
                    section=parsed_doc['section'],
                    content=parsed_doc['content'][:500] + "..." if len(parsed_doc['content']) > 500 else parsed_doc['content'],  # Store first 500 chars as preview
                    metadata_json=str(metadata)
                )
                db_ops.create_document(db, obj_in=doc_in)
            finally:
                db.close()

            return {
                'status': 'success',
                'message': f'Successfully ingested {len(qdrant_points)} chunks from {file_path}',
                'chunks_count': len(qdrant_points),
                'url': parsed_doc['url']
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Error ingesting {file_path}: {str(e)}',
                'error': str(e)
            }

    def ingest_directory(self, directory_path: str, recursive: bool = True, delay_seconds: float = 2.0) -> Dict[str, Any]:
        """
        Ingest all markdown files in a directory with rate limiting
        
        Args:
            directory_path: Path to directory containing markdown files
            recursive: Whether to search recursively
            delay_seconds: Delay between processing files to avoid rate limits (default: 2 seconds)
        """
        # Find all markdown files in the directory
        if recursive:
            md_files = list(Path(directory_path).rglob("*.md"))
        else:
            md_files = list(Path(directory_path).glob("*.md"))

        results = {
            'total_files': len(md_files),
            'successful': 0,
            'skipped': 0,
            'failed': 0,
            'details': []
        }

        for i, file_path in enumerate(md_files):
            result = self.ingest_file(str(file_path))
            results['details'].append(result)

            if result['status'] == 'success':
                results['successful'] += 1
            elif result['status'] == 'skipped':
                results['skipped'] += 1
            else:
                results['failed'] += 1
            
            # Add delay between files to avoid rate limits (except after last file)
            if i < len(md_files) - 1 and result['status'] == 'success':
                time.sleep(delay_seconds)

        return results

    def ingest_from_docs_directory(self) -> Dict[str, Any]:
        """
        Run initial ingestion of all website/docs/ content to populate Qdrant collection
        """
        # Assuming the docs directory is relative to the project root
        # In a real implementation, this path would be configurable
        docs_path = os.path.join(os.getcwd(), "website", "docs")
        if not os.path.exists(docs_path):
            # Try alternative paths
            docs_path = os.path.join(os.getcwd(), "..", "website", "docs")
            if not os.path.exists(docs_path):
                docs_path = os.path.join(os.getcwd(), "docs")
                if not os.path.exists(docs_path):
                    raise FileNotFoundError(f"Docs directory not found at expected locations. Searched: {os.path.join(os.getcwd(), 'website', 'docs')}, {os.path.join(os.getcwd(), '..', 'website', 'docs')}, {os.path.join(os.getcwd(), 'docs')}")

        return self.ingest_directory(docs_path, recursive=True)


# Global instance
ingestion_service = IngestionService()