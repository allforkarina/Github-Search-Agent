"""Offline tests for the actual RAG retrieval implementation."""
import importlib
import sys
import types
import unittest
from dataclasses import replace
from unittest.mock import Mock, patch

import numpy as np


PACKAGE = "reposcouts.rag"
retrieval = importlib.import_module(PACKAGE + ".retrieval")
embeddings = importlib.import_module(PACKAGE + ".embeddings")
cache_module = importlib.import_module(PACKAGE + ".index_cache")
schemas = importlib.import_module(PACKAGE + ".schemas")
RagConfig = importlib.import_module(PACKAGE + ".config").RagConfig


def chunk(chunk_id, text, **updates):
    values = dict(chunk_id=chunk_id, full_name="owner/repo", commit_sha="a" * 40,
                  path=chunk_id + ".py", start_line=1, end_line=1, text=text)
    return schemas.EvidenceChunk(**(values | updates))


class FakeEmbedder:
    fingerprint = "test-embedding@revision1"

    def __init__(self, documents, query=None):
        self.documents = documents
        self.query = [[1.0, 0.0]] if query is None else query
        self.document_calls = 0

    def embed_documents(self, chunks):
        self.document_calls += 1
        if isinstance(self.documents, Exception):
            raise self.documents
        return np.asarray(self.documents)

    def embed_query(self, query):
        if isinstance(self.query, Exception):
            raise self.query
        return np.asarray(self.query)


class RetrievalTests(unittest.TestCase):
    def test_lexical_tokens_cover_identifiers_and_chinese(self):
        tokens = set(retrieval.lexical_tokens("getRepositoryInfo persistent_memory 持久化"))
        self.assertTrue({"getrepositoryinfo", "get", "repository", "info",
                         "persistent_memory", "persistent", "memory", "持久", "久化"} <= tokens)

    def test_rrf_uses_one_based_ranks_and_boosts_shared_hits(self):
        scores = dict(retrieval.rrf_fuse(["a", "b"], ["b", "c"], k=60))
        self.assertAlmostEqual(scores["a"], 1 / 61)
        self.assertAlmostEqual(scores["b"], 1 / 62 + 1 / 61)
        self.assertEqual(retrieval.rrf_fuse(["a", "b"], ["b", "c"])[0][0], "b")

    def test_rrf_deduplicates_each_branch_and_breaks_ties_stably(self):
        result = retrieval.rrf_fuse(["b", "b"], ["a"])
        self.assertEqual([item[0] for item in result], ["a", "b"])
        self.assertAlmostEqual(result[0][1], result[1][1])

    def test_hybrid_returns_both_rank_sources(self):
        docs = [chunk("memory", "persistent_memory checkpoint sqlite\n"),
                chunk("web", "FastAPI HTTP route\n"), chunk("intro", "project introduction\n")]
        index = retrieval.HybridIndex(docs, embedder=FakeEmbedder([[4, 0], [0, 3], [0.1, 0.9]]))
        result = index.search("persistent_memory")
        self.assertEqual(result.mode, "hybrid")
        hit = result.hits[0]
        self.assertEqual(hit.chunk.chunk_id, "memory")
        self.assertEqual(hit.dense_rank, 1)
        self.assertEqual(hit.bm25_rank, 1)
        self.assertAlmostEqual(hit.dense_score, 1.0)
        self.assertAlmostEqual(hit.rrf_score, 2 / 61)

    def test_real_bm25_package_returns_single_document_keyword_hit(self):
        index = retrieval.HybridIndex([chunk("one", "SqliteSaver checkpoint\n")])
        result = index.search("SqliteSaver")
        self.assertEqual(result.mode, "bm25")
        self.assertEqual(len(result.hits), 1)
        self.assertGreater(result.hits[0].bm25_score, 0)
        self.assertIsNone(result.hits[0].dense_rank)

    def test_bm25_does_not_pad_results_with_nonmatching_documents(self):
        index = retrieval.HybridIndex([chunk("one", "checkpoint sqlite\n"), chunk("two", "FastAPI HTTP\n")])
        self.assertEqual([hit.chunk.chunk_id for hit in index.search("sqlite").hits], ["one"])
        self.assertEqual(index.search("unfindableword").hits, [])

    def test_chinese_keyword_retrieval(self):
        index = retrieval.HybridIndex([chunk("one", "支持持久化记忆\n"), chunk("two", "图像生成\n")])
        self.assertEqual(index.search("持久化").hits[0].chunk.chunk_id, "one")

    def test_document_embedding_error_falls_back_without_error_text(self):
        index = retrieval.HybridIndex(
            [chunk("one", "checkpoint\n")],
            embedder=FakeEmbedder(RuntimeError("Bearer secret-value")),
        )
        result = index.search("checkpoint")
        self.assertEqual(result.mode, "bm25")
        self.assertEqual(result.warnings, ["document_embedding_failed"])
        self.assertNotIn("secret-value", result.model_dump_json())

    def test_query_embedding_error_falls_back_for_that_query(self):
        embedder = FakeEmbedder([[1, 0]], query=RuntimeError("provider failed"))
        index = retrieval.HybridIndex([chunk("one", "checkpoint\n")], embedder=embedder)
        self.assertEqual(index.search("checkpoint").mode, "bm25")
        embedder.query = [[1, 0]]
        self.assertEqual(index.search("checkpoint").mode, "hybrid")

    def test_invalid_document_vectors_are_rejected(self):
        invalid = ([[0, 0]], [[np.nan, 1]], [[np.inf, 1]], [[1, 0], [0, 1]], [1, 0])
        for vectors in invalid:
            with self.subTest(vectors=vectors):
                index = retrieval.HybridIndex([chunk("one", "checkpoint\n")], embedder=FakeEmbedder(vectors))
                self.assertEqual(index.search("checkpoint").mode, "bm25")

    def test_query_dimension_mismatch_is_visible_fallback(self):
        index = retrieval.HybridIndex(
            [chunk("one", "checkpoint\n")], embedder=FakeEmbedder([[1, 0]], [[1, 0, 0]]),
        )
        result = index.search("checkpoint")
        self.assertEqual(result.mode, "bm25")
        self.assertIn("query_embedding_failed", result.warnings)

    def test_empty_index_is_unavailable_and_blank_queries_are_rejected(self):
        index = retrieval.HybridIndex([])
        self.assertEqual(index.search("anything").mode, "unavailable")
        self.assertEqual(index.search("anything").hits, [])
        with self.assertRaises(ValueError):
            index.search("  ")

    def test_top_k_and_copy_boundaries(self):
        original = chunk("one", "checkpoint\n")
        index = retrieval.HybridIndex([original, chunk("two", "checkpoint sqlite\n")],
                                      config=replace(RagConfig(), evidence_top_k=1))
        original.text = "changed\n"
        result = index.search("checkpoint")
        self.assertEqual(len(result.hits), 1)
        result.hits[0].chunk.text = "mutated result\n"
        self.assertNotIn("mutated result", index.search("checkpoint").hits[0].chunk.text)
        self.assertEqual(index.chunks[0].text, "checkpoint\n")

    def test_mixed_versions_and_duplicate_ids_fail_fast(self):
        first = chunk("one", "checkpoint\n")
        for other in (first, chunk("two", "other\n", commit_sha="b" * 40),
                      chunk("three", "other\n", full_name="another/repo")):
            with self.subTest(other=other.chunk_id), self.assertRaises(ValueError):
                retrieval.HybridIndex([first, other])


class CacheTests(unittest.TestCase):
    def test_reuses_index_without_embedding_documents_again(self):
        cache = cache_module.RequestIndexCache()
        embedder = FakeEmbedder([[1, 0]])
        docs = [chunk("one", "checkpoint\n")]
        first, hit = cache.get_or_build(docs, embedder=embedder)
        self.assertFalse(hit)
        second, hit = cache.get_or_build(docs, embedder=embedder)
        self.assertTrue(hit)
        self.assertIs(first, second)
        self.assertEqual(embedder.document_calls, 1)

    def test_changed_sha_text_model_or_config_invalidates_cache(self):
        cache = cache_module.RequestIndexCache()
        embedder = FakeEmbedder([[1, 0]])
        docs = [chunk("one", "checkpoint\n")]
        original, _ = cache.get_or_build(docs, embedder=embedder)
        for changes in ({"commit_sha": "b" * 40}, {"text": "changed\n"}):
            changed = docs[0].model_copy(update=changes)
            rebuilt, hit = cache.get_or_build([changed], embedder=embedder)
            self.assertFalse(hit)
            self.assertIsNot(original, rebuilt)
        changed_model = FakeEmbedder([[1, 0]])
        changed_model.fingerprint = "test-embedding@revision2"
        _, hit = cache.get_or_build(docs, embedder=changed_model)
        self.assertFalse(hit)
        _, hit = cache.get_or_build(docs, embedder=embedder, config=replace(RagConfig(), chunk_tokens=300))
        self.assertFalse(hit)

    def test_lru_limit_and_clear_release_request_indexes(self):
        cache = cache_module.RequestIndexCache(max_entries=1)
        first = [chunk("one", "checkpoint\n")]
        cache.get_or_build(first)
        cache.get_or_build([chunk("two", "FastAPI\n")])
        _, hit = cache.get_or_build(first)
        self.assertFalse(hit)
        cache.clear()
        _, hit = cache.get_or_build(first)
        self.assertFalse(hit)


class LocalAdapterTests(unittest.TestCase):
    def setUp(self):
        embeddings._load_model.cache_clear()
        self.addCleanup(embeddings._load_model.cache_clear)
        self.model = Mock()
        self.model.max_seq_length = 512
        self.model.tokenizer.side_effect = lambda text, **kwargs: {"input_ids": [0] * (len(text) + 2)}
        self.model.encode.side_effect = lambda texts, **kwargs: np.tile([[1, 0]], (len(texts), 1))
        self.factory = Mock(return_value=self.model)
        module = types.ModuleType("sentence_transformers")
        module.SentenceTransformer = self.factory
        patcher = patch.dict(sys.modules, {"sentence_transformers": module})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_pinned_revision_and_shared_local_model(self):
        first = embeddings.LocalEmbedder()
        second = embeddings.LocalEmbedder()
        self.factory.assert_called_once_with(
            RagConfig().embedding_model, revision=RagConfig().embedding_revision,
            trust_remote_code=False, local_files_only=True,
        )
        self.assertEqual(first.fingerprint, second.fingerprint)

    def test_passage_query_prefixes_and_tokenizer_are_explicit(self):
        embedder = embeddings.LocalEmbedder()
        embedder.embed_documents([chunk("one", "checkpoint\n")])
        document_text = self.model.encode.call_args.args[0][0]
        self.assertEqual(document_text, "passage: one.py\n\ncheckpoint\n")
        embedder.embed_query("persistent memory")
        self.assertEqual(self.model.encode.call_args.args[0], ["query: persistent memory"])
        self.assertEqual(self.model.encode.call_args.kwargs["prompt"], "")
        self.assertTrue(self.model.encode.call_args.kwargs["normalize_embeddings"])
        for call in self.model.tokenizer.call_args_list:
            self.assertFalse(call.kwargs["truncation"])
            self.assertTrue(call.kwargs["add_special_tokens"])

    def test_long_query_is_rejected_before_encoding(self):
        embedder = embeddings.LocalEmbedder()
        with self.assertRaises(ValueError):
            embedder.embed_query("x" * 600)
        self.model.encode.assert_not_called()

    def test_chunk_budget_cannot_exceed_actual_model_limit(self):
        with self.assertRaises(ValueError):
            embeddings.LocalEmbedder(replace(RagConfig(), chunk_tokens=600))


if __name__ == "__main__":
    unittest.main()
