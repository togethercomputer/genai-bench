"""Tests for JSONL dataset loading functionality."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from genai_bench.data.config import DatasetConfig, DatasetSourceConfig
from genai_bench.data.loaders.text import TextDatasetLoader
from genai_bench.data.sources import FileDatasetSource


class TestJSONLLoading:
    """Test JSONL file loading functionality."""

    def test_load_jsonl_file_success(self):
        """Test successful JSONL file loading."""
        # Create test JSONL data
        test_data = [
            {"messages": [{"role": "user", "content": "Hello"}]},
            {"messages": [{"role": "user", "content": "How are you?"}]},
            {"messages": [{"role": "user", "content": "What's the weather?"}]},
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for item in test_data:
                f.write(json.dumps(item) + '\n')
            temp_path = f.name

        try:
            config = DatasetSourceConfig(type="file", path=temp_path, file_format="jsonl")
            source = FileDatasetSource(config)
            result = source.load()
            
            assert len(result) == 3
            assert result[0]["messages"][0]["content"] == "Hello"
            assert result[1]["messages"][0]["content"] == "How are you?"
            assert result[2]["messages"][0]["content"] == "What's the weather?"
        finally:
            Path(temp_path).unlink()

    def test_load_jsonl_file_invalid_json(self):
        """Test JSONL file loading with invalid JSON."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write('{"valid": "json"}\n')
            f.write('{"invalid": json}\n')  # Missing quotes
            temp_path = f.name

        try:
            config = DatasetSourceConfig(type="file", path=temp_path, file_format="jsonl")
            source = FileDatasetSource(config)
            
            with pytest.raises(ValueError, match="Invalid JSON on line 2"):
                source.load()
        finally:
            Path(temp_path).unlink()

    def test_load_jsonl_file_empty_lines(self):
        """Test JSONL file loading with empty lines."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write('{"messages": [{"role": "user", "content": "Hello"}]}\n')
            f.write('\n')  # Empty line
            f.write('{"messages": [{"role": "user", "content": "World"}]}\n')
            temp_path = f.name

        try:
            config = DatasetSourceConfig(type="file", path=temp_path, file_format="jsonl")
            source = FileDatasetSource(config)
            result = source.load()
            
            assert len(result) == 2
            assert result[0]["messages"][0]["content"] == "Hello"
            assert result[1]["messages"][0]["content"] == "World"
        finally:
            Path(temp_path).unlink()


class TestMessagesProcessing:
    """Test messages format processing in TextDatasetLoader."""

    def test_process_messages_data_simple(self):
        """Test processing simple messages data."""
        test_data = [
            {"messages": [{"role": "user", "content": "Hello"}]},
            {"messages": [{"role": "user", "content": "How are you?"}]},
        ]
        
        config = DatasetConfig(
            source=DatasetSourceConfig(type="file", path="test.jsonl", file_format="jsonl")
        )
        loader = TextDatasetLoader(config)
        
        result = loader._process_messages_data(test_data)
        
        assert result == ["Hello", "How are you?"]

    def test_process_messages_data_conversation(self):
        """Test processing conversation data with multiple messages."""
        test_data = [
            {
                "messages": [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there!"},
                    {"role": "user", "content": "What's the weather?"}
                ]
            }
        ]
        
        config = DatasetConfig(
            source=DatasetSourceConfig(type="file", path="test.jsonl", file_format="jsonl")
        )
        loader = TextDatasetLoader(config)
        
        result = loader._process_messages_data(test_data)
        
        assert result == ["What's the weather?"]

    def test_process_messages_data_no_user_message(self):
        """Test processing data with no user messages."""
        test_data = [
            {"messages": [{"role": "system", "content": "You are helpful."}]},
            {"messages": [{"role": "assistant", "content": "Hello there!"}]},
        ]
        
        config = DatasetConfig(
            source=DatasetSourceConfig(type="file", path="test.jsonl", file_format="jsonl")
        )
        loader = TextDatasetLoader(config)
        
        result = loader._process_messages_data(test_data)
        
        assert result == []

    def test_process_messages_data_empty_messages(self):
        """Test processing data with empty messages."""
        test_data = [
            {"messages": []},
            {"other_field": "value"},
        ]
        
        config = DatasetConfig(
            source=DatasetSourceConfig(type="file", path="test.jsonl", file_format="jsonl")
        )
        loader = TextDatasetLoader(config)
        
        result = loader._process_messages_data(test_data)
        
        assert result == []

    def test_process_loaded_data_detects_messages_format(self):
        """Test that _process_loaded_data detects messages format automatically."""
        test_data = [
            {"messages": [{"role": "user", "content": "Hello"}]},
            {"messages": [{"role": "user", "content": "World"}]},
        ]
        
        config = DatasetConfig(
            source=DatasetSourceConfig(type="file", path="test.jsonl", file_format="jsonl")
        )
        loader = TextDatasetLoader(config)
        
        result = loader._process_loaded_data(test_data)
        
        assert result == ["Hello", "World"]

    def test_process_loaded_data_fallback_to_regular_processing(self):
        """Test that _process_loaded_data falls back to regular processing for non-messages data."""
        test_data = ["Hello", "World", "Test"]
        
        config = DatasetConfig(
            source=DatasetSourceConfig(type="file", path="test.txt", file_format="txt")
        )
        loader = TextDatasetLoader(config)
        
        result = loader._process_loaded_data(test_data)
        
        assert result == ["Hello", "World", "Test"]


class TestJSONLIntegration:
    """Test end-to-end JSONL integration."""

    def test_jsonl_dataset_loading_integration(self):
        """Test complete JSONL dataset loading integration."""
        # Create test JSONL file
        test_data = [
            {
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "What is Python?"},
                    {"role": "assistant", "content": "Python is a programming language."},
                    {"role": "user", "content": "Tell me more about it."}
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "How do I install Python?"}
                ]
            }
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for item in test_data:
                f.write(json.dumps(item) + '\n')
            temp_path = f.name

        try:
            # Create dataset config
            config = DatasetConfig(
                source=DatasetSourceConfig(type="file", path=temp_path, file_format="jsonl")
            )
            
            # Create loader and load data
            loader = TextDatasetLoader(config)
            result = loader.load_request()
            
            # Should extract the last user message from each conversation
            assert result == ["Tell me more about it.", "How do I install Python?"]
        finally:
            Path(temp_path).unlink()

    def test_jsonl_auto_detection(self):
        """Test automatic JSONL format detection from file extension."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write('{"messages": [{"role": "user", "content": "Hello"}]}\n')
            temp_path = f.name

        try:
            # Test auto-detection via from_cli_args
            config = DatasetConfig.from_cli_args(dataset_path=temp_path)
            
            assert config.source.type == "file"
            assert config.source.file_format == "jsonl"
        finally:
            Path(temp_path).unlink()
