import concurrent.futures
import subprocess
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest import mock

from backend import recommendation_engine as engine
from backend import storage
from backend import chatbot
from backend import llm


ROOT = Path(__file__).resolve().parents[1]


class SentenceModelTests(unittest.TestCase):
    def setUp(self):
        self.old_model = engine._sent_model
        self.old_available = engine._sent_available
        engine._sent_model = None
        engine._sent_available = None

    def tearDown(self):
        engine._sent_model = self.old_model
        engine._sent_available = self.old_available

    def test_loads_model_on_first_attempt(self):
        instance = object()
        constructor = mock.Mock(return_value=instance)
        fake_module = types.ModuleType("sentence_transformers")
        fake_module.SentenceTransformer = constructor
        with mock.patch.dict(sys.modules, {"sentence_transformers": fake_module}):
            self.assertTrue(engine._load_sentence_model())
            self.assertIs(engine._sent_model, instance)
            self.assertTrue(engine._sent_available)
        constructor.assert_called_once_with("all-MiniLM-L6-v2")

    def test_failed_load_is_cached_without_retry(self):
        constructor = mock.Mock(side_effect=RuntimeError("mock load failure"))
        fake_module = types.ModuleType("sentence_transformers")
        fake_module.SentenceTransformer = constructor
        with mock.patch.dict(sys.modules, {"sentence_transformers": fake_module}):
            with mock.patch("builtins.print"):
                self.assertFalse(engine._load_sentence_model())
                self.assertFalse(engine._load_sentence_model())
        constructor.assert_called_once()
        self.assertIs(engine._sent_available, False)

    def test_concurrent_first_load_is_only_attempted_once(self):
        instance = object()
        def construct(_name):
            time.sleep(0.05)
            return instance
        constructor = mock.Mock(side_effect=construct)
        fake_module = types.ModuleType("sentence_transformers")
        fake_module.SentenceTransformer = constructor
        with mock.patch.dict(sys.modules, {"sentence_transformers": fake_module}):
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                results = list(pool.map(lambda _: engine._load_sentence_model(), range(16)))
        self.assertTrue(all(results))
        constructor.assert_called_once()

    def test_discovery_uses_semantic_path_when_model_available(self):
        class FakeModel:
            def encode(self, texts, normalize_embeddings):
                self.texts = texts
                self.normalized = normalize_embeddings
                return engine.loaded_movie_embeddings[:1]
        model = FakeModel()
        with mock.patch.object(engine, "_sent_model", model), \
             mock.patch.object(engine, "_load_sentence_model", return_value=True):
            results = engine.discover_movies("a semantic query", limit=2)
        self.assertTrue(results)
        self.assertIn("Semantic_Score", results[0])
        self.assertTrue(model.normalized)

    def test_discovery_keeps_keyword_fallback_when_model_unavailable(self):
        with mock.patch.object(engine, "_load_sentence_model", return_value=False):
            with mock.patch.object(engine, "_keywords_from", return_value=["action"]):
                results = engine.discover_movies("action query", limit=2)
        self.assertTrue(results)
        self.assertIn("Relevance", results[0])
        self.assertNotIn("Semantic_Score", results[0])


class ChatbotProviderTests(unittest.TestCase):
    def test_missing_key_keeps_fallback_without_client_creation(self):
        with mock.patch.object(llm, "Groq") as groq_cls:
            with self.assertRaises(llm.LLMNotConfigured):
                llm._create_client("  ", 30.0, "https://provider.invalid/v1")
        groq_cls.assert_not_called()

    def test_existing_provider_is_initialized_with_configured_values(self):
        client = object()
        with mock.patch.object(llm, "Groq", return_value=client) as groq_cls:
            result = llm._create_client("mock-key", 12.5, "https://provider.invalid/v1")
        self.assertIs(result, client)
        groq_cls.assert_called_once_with(
            api_key="mock-key", timeout=12.5, base_url="https://provider.invalid/v1"
        )


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.old_data_dir = storage._DATA_DIR
        storage._DATA_DIR = Path(self.tempdir.name)

    def tearDown(self):
        storage._DATA_DIR = self.old_data_dir
        self.tempdir.cleanup()

    def test_atomic_write_cleans_temporary_file_after_replace_failure(self):
        with mock.patch.object(storage.os, "replace", side_effect=OSError("mock replace failure")):
            with self.assertRaises(OSError):
                storage.save_json("sample.json", {"value": 1})
        self.assertEqual(list(Path(self.tempdir.name).glob("*.tmp")), [])

    def test_read_modify_write_does_not_lose_cross_process_updates(self):
        workers = 4
        increments_per_worker = 20
        code = (
            "import sys; from pathlib import Path; from backend import storage; "
            "storage._DATA_DIR = Path(sys.argv[1]); "
            "[(storage.update_json('counter.json', {'count': 0}, "
            "lambda d: ({'count': d['count'] + 1}, None))) for _ in range(int(sys.argv[2]))]"
        )
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(
                subprocess.run,
                [sys.executable, "-c", code, self.tempdir.name, str(increments_per_worker)],
                cwd=ROOT, check=True, capture_output=True, text=True,
            ) for _ in range(workers)]
            for future in futures:
                future.result(timeout=60)
        result = storage.load_json("counter.json", {"count": 0})
        self.assertEqual(result["count"], workers * increments_per_worker)


if __name__ == "__main__":
    unittest.main()
