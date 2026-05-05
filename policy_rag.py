import re
import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class PolicyChunk:
    chunk_id: str
    section_id: str
    section_title: str
    subsection_id: str
    subsection_title: str
    text: str


class PolicyRetriever:
    def __init__(
        self,
        policy_path: str = "policy/gaggia_policy.txt",
        model_name: str = "all-MiniLM-L6-v2",
    ):
        self.policy_path = Path(policy_path)
        self.model = SentenceTransformer(model_name)
        self.chunks: List[PolicyChunk] = []
        self.embeddings: Optional[np.ndarray] = None

    def load_policy(self) -> str:
        if not self.policy_path.exists():
            raise FileNotFoundError(
                f"Policy file not found: {self.policy_path.resolve()}"
            )

        return self.policy_path.read_text(encoding="utf-8")

    def chunk_policy(self, text: str) -> List[PolicyChunk]:
        """
        Chunk policy by SECTION and top-level subsection.

        Example:
        SECTION 14 — IDENTITY VERIFICATION & TRUST TIERS

        14.1. OVERVIEW
        14.2. TEAM BLUE — TRUSTED USERS
            14.2.1. Definition
            14.2.2. Permissions

        This keeps 14.2.1, 14.2.2, etc. inside the 14.2 chunk.
        """

        text = text.replace("\r\n", "\n").replace("\r", "\n")

        section_pattern = re.compile(
            r"(?im)^SECTION\s+(\d+)\s+[—-]\s+(.+?)\s*$"
        )

        # IMPORTANT:
        # This matches only top-level subsections like 14.2.
        # It does NOT match nested items like 14.2.1.
        subsection_pattern = re.compile(
            r"(?im)^(\d+\.\d+)\.?\s+(.+?)\s*$"
        )

        sections = list(section_pattern.finditer(text))
        chunks: List[PolicyChunk] = []
        chunk_num = 0

        if not sections:
            raise ValueError(
                "No SECTION headings found. Expected format like: SECTION 14 — IDENTITY VERIFICATION & TRUST TIERS"
            )

        intro = text[:sections[0].start()].strip()
        if intro:
            chunks.append(
                PolicyChunk(
                    chunk_id=f"chunk_{chunk_num}",
                    section_id="intro",
                    section_title="Document Header",
                    subsection_id="intro",
                    subsection_title="Document Header",
                    text=intro,
                )
            )
            chunk_num += 1

        for section_idx, section_match in enumerate(sections):
            section_id = section_match.group(1).strip()
            section_title = section_match.group(2).strip()

            section_start = section_match.end()
            section_end = (
                sections[section_idx + 1].start()
                if section_idx + 1 < len(sections)
                else len(text)
            )

            section_body = text[section_start:section_end].strip()
            section_heading = f"SECTION {section_id} — {section_title}"

            # Only accept subsection headers that belong to this section.
            all_subsections = list(subsection_pattern.finditer(section_body))
            subsections = [
                m for m in all_subsections
                if m.group(1).split(".")[0] == section_id
            ]

            if not subsections:
                full_text = f"{section_heading}\n\n{section_body}".strip()

                chunks.append(
                    PolicyChunk(
                        chunk_id=f"chunk_{chunk_num}",
                        section_id=section_id,
                        section_title=section_title,
                        subsection_id=section_id,
                        subsection_title=section_title,
                        text=full_text,
                    )
                )
                chunk_num += 1
                continue

            preamble = section_body[:subsections[0].start()].strip()
            if preamble:
                full_text = f"{section_heading}\n\n{preamble}".strip()

                chunks.append(
                    PolicyChunk(
                        chunk_id=f"chunk_{chunk_num}",
                        section_id=section_id,
                        section_title=section_title,
                        subsection_id=f"{section_id}.0",
                        subsection_title="Section Preamble",
                        text=full_text,
                    )
                )
                chunk_num += 1

            for sub_idx, sub_match in enumerate(subsections):
                subsection_id = sub_match.group(1).strip()
                subsection_title = sub_match.group(2).strip()

                sub_start = sub_match.start()
                sub_end = (
                    subsections[sub_idx + 1].start()
                    if sub_idx + 1 < len(subsections)
                    else len(section_body)
                )

                subsection_text = section_body[sub_start:sub_end].strip()

                full_text = (
                    f"{section_heading}\n\n"
                    f"{subsection_text}"
                ).strip()

                chunks.append(
                    PolicyChunk(
                        chunk_id=f"chunk_{chunk_num}",
                        section_id=section_id,
                        section_title=section_title,
                        subsection_id=subsection_id,
                        subsection_title=subsection_title,
                        text=full_text,
                    )
                )
                chunk_num += 1

        return chunks

    def build_index(self) -> None:
        text = self.load_policy()
        self.chunks = self.chunk_policy(text)

        if not self.chunks:
            raise ValueError("No policy chunks were created. Check section formatting.")

        chunk_texts = [chunk.text for chunk in self.chunks]

        self.embeddings = self.model.encode(
            chunk_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if self.embeddings is None or not self.chunks:
            raise RuntimeError("Index not built. Call build_index() first.")

        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        scores = cosine_similarity(query_embedding, self.embeddings)[0]
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []

        for idx in top_indices:
            chunk = self.chunks[idx]
            results.append(
                {
                    "score": float(scores[idx]),
                    "chunk_id": chunk.chunk_id,
                    "section_id": chunk.section_id,
                    "section_title": chunk.section_title,
                    "subsection_id": chunk.subsection_id,
                    "subsection_title": chunk.subsection_title,
                    "text": chunk.text,
                }
            )

        return results

    def save_index(self, output_dir: str = "policy_index") -> None:
        if self.embeddings is None:
            raise RuntimeError("Index not built. Call build_index() first.")

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        np.save(output_path / "embeddings.npy", self.embeddings)

        with open(output_path / "chunks.json", "w", encoding="utf-8") as f:
            json.dump([asdict(chunk) for chunk in self.chunks], f, indent=2)

    def load_index(self, index_dir: str = "policy_index") -> None:
        index_path = Path(index_dir)

        self.embeddings = np.load(index_path / "embeddings.npy")

        with open(index_path / "chunks.json", "r", encoding="utf-8") as f:
            chunk_dicts = json.load(f)

        self.chunks = [PolicyChunk(**chunk) for chunk in chunk_dicts]


if __name__ == "__main__":
    retriever = PolicyRetriever("policy/gaggia_policy.txt")
    retriever.build_index()
    retriever.save_index()

    print(f"Built index with {len(retriever.chunks)} chunks.")

    for chunk in retriever.chunks:
        print(
            chunk.chunk_id,
            "| Section:",
            chunk.section_id,
            "| Subsection:",
            chunk.subsection_id,
            "|",
            chunk.subsection_title,
        )

    # test_results = retriever.retrieve(
    #     "Can a Team Red user reset a password?",
    #     top_k=3,
    # )

    # for result in test_results:
    #     print("\n---")
    #     print(f"Score: {result['score']:.4f}")
    #     print(f"Section: {result['section_id']} - {result['section_title']}")
    #     print(f"Subsection: {result['subsection_id']} - {result['subsection_title']}")
    #     print(result["text"][:500])