# JSONL Dataset Support

This guide explains how to use JSONL (JSON Lines) format datasets with genai-bench, including support for conversation-style data with messages format.

## Overview

JSONL format is ideal for large datasets where each line contains a separate JSON object. This is particularly useful for:

- **Conversation datasets**: Multi-turn conversations with system, user, and assistant messages
- **Large datasets**: Streaming processing of datasets with thousands of entries
- **Real-world data**: Datasets exported from chat applications or conversation platforms

## JSONL Format Specification

### Basic Structure

Each line in a JSONL file must contain a valid JSON object. For GenAI benchmarking, we support two main formats:

#### 1. Simple Prompt Format
```jsonl
{"prompt": "What is machine learning?"}
{"prompt": "Explain quantum computing"}
{"prompt": "How does a neural network work?"}
```

#### 2. Messages Format (Recommended for Conversations)
```jsonl
{"messages": [{"role": "user", "content": "What is machine learning?"}]}
{"messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Explain quantum computing"}]}
{"messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there!"}, {"role": "user", "content": "How are you?"}]}
```

### Messages Format Details

The messages format follows the OpenAI chat completion format:

- **`role`**: One of `"system"`, `"user"`, or `"assistant"`
- **`content`**: The text content of the message
- **Order**: Messages should be in chronological order of the conversation

## Configuration

### Basic JSONL Configuration

```json
{
  "source": {
    "type": "file",
    "path": "/path/to/dataset.jsonl",
    "file_format": "jsonl"
  },
  "prompt_column": "messages"
}
```

### Messages Format Configuration

For conversation datasets using the messages format:

```json
{
  "source": {
    "type": "file",
    "path": "/path/to/conversations.jsonl",
    "file_format": "jsonl"
  },
  "prompt_column": "messages",
  "conversation_mode": true
}
```

## Usage Examples

### Command Line Usage

```bash
# Auto-detect JSONL format from file extension
genai-bench --dataset_path /path/to/dataset.jsonl --task text-to-text

# Explicitly specify JSONL format
genai-bench --dataset_path /path/to/dataset.jsonl --file_format jsonl --task text-to-text
```

### Programmatic Usage

```python
from genai_bench.data.config import DatasetConfig, DatasetSourceConfig

# Create configuration
config = DatasetConfig(
    source=DatasetSourceConfig(
        type="file",
        path="/path/to/conversations.jsonl",
        file_format="jsonl"
    ),
    prompt_column="messages"
)

# Load dataset
from genai_bench.data.loaders.factory import DataLoaderFactory
data = DataLoaderFactory.load_data_for_task("text-to-text", config)
```

## Message Processing

When using the messages format, genai-bench automatically:

1. **Detects messages format**: Checks if data contains `"messages"` arrays
2. **Extracts prompts**: Uses the last user message as the primary prompt
3. **Preserves context**: Maintains conversation history for advanced use cases

### Prompt Extraction Strategies

- **Last User Message** (default): Extracts the final user message as the prompt
- **Full Conversation**: Uses the entire conversation as context
- **System + User**: Combines system instructions with user messages

## Example Datasets

### Simple Q&A Dataset
```jsonl
{"messages": [{"role": "user", "content": "What is the capital of France?"}]}
{"messages": [{"role": "user", "content": "How do you calculate the area of a circle?"}]}
{"messages": [{"role": "user", "content": "Explain photosynthesis"}]}
```

### Multi-turn Conversation Dataset
```jsonl
{"messages": [{"role": "system", "content": "You are a helpful coding assistant."}, {"role": "user", "content": "How do I write a Python function?"}, {"role": "assistant", "content": "Here's how to write a Python function..."}, {"role": "user", "content": "Can you show me an example with error handling?"}]}
{"messages": [{"role": "user", "content": "What's the difference between a list and a tuple?"}]}
```

### Long Context Dataset
```jsonl
{"messages": [{"role": "system", "content": "You are a personal browser assistant..."}, {"role": "user", "content": "How does the GitHub integration work?"}]}
```

## Performance Considerations

- **Memory Efficient**: JSONL files are processed line-by-line, making them suitable for large datasets
- **Streaming**: No need to load entire dataset into memory at once
- **Error Handling**: Invalid JSON lines are reported with line numbers for easy debugging

## Error Handling

The JSONL loader provides detailed error messages:

```
ValueError: Invalid JSON on line 42: Expecting ',' delimiter: line 1 column 15 (char 14)
```

This helps identify and fix malformed JSON in your dataset files.

## Migration from Other Formats

### From CSV
If you have conversation data in CSV format, convert to JSONL:

```python
import csv
import json

# Convert CSV to JSONL
with open('conversations.csv', 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    with open('conversations.jsonl', 'w') as jsonlfile:
        for row in reader:
            messages = [
                {"role": "user", "content": row['user_message']},
                {"role": "assistant", "content": row['assistant_message']}
            ]
            jsonlfile.write(json.dumps({"messages": messages}) + '\n')
```

### From JSON
If you have a JSON array, convert to JSONL:

```python
import json

# Convert JSON array to JSONL
with open('dataset.json', 'r') as jsonfile:
    data = json.load(jsonfile)
    
with open('dataset.jsonl', 'w') as jsonlfile:
    for item in data:
        jsonlfile.write(json.dumps(item) + '\n')
```

## Best Practices

1. **Validate your JSONL files** before using them with genai-bench
2. **Use consistent message formats** across your dataset
3. **Include system messages** when relevant for context
4. **Test with small subsets** before running full benchmarks
5. **Monitor memory usage** for very large datasets

## Troubleshooting

### Common Issues

**Q: My JSONL file isn't being detected as messages format**
A: Ensure your JSON objects contain a `"messages"` key with an array of message objects.

**Q: I'm getting JSON parsing errors**
A: Use a JSON validator to check for malformed JSON. The error message will indicate the problematic line number.

**Q: Only some prompts are being extracted**
A: Check that your messages contain user messages with the `"role": "user"` key.

**Q: Performance is slow with large files**
A: This is expected for very large datasets. Consider splitting into smaller files or using streaming processing.
