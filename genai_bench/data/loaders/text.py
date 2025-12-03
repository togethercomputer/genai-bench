from typing import Any, Dict, List, Set, Tuple, Union

from genai_bench.data.loaders.base import DatasetFormat, DatasetLoader
from genai_bench.logging import init_logger

logger = init_logger(__name__)


class TextDatasetLoader(DatasetLoader):
    """
    This datasetLoader is responsible for loading prompts from a data source.

    TODO: Add support for prompt lambdas similar to ImageDatasetLoader.
    """

    supported_formats: Set[DatasetFormat] = {
        DatasetFormat.TEXT,
        DatasetFormat.CSV,
        DatasetFormat.JSON,
        DatasetFormat.JSONL,
        DatasetFormat.HUGGINGFACE_HUB,
    }
    media_type = "Text"

    def _process_loaded_data(
        self, data: Any
    ) -> Union[List[str], List[Tuple[str, Dict[str, Any]]]]:
        """Process data loaded from dataset source."""
        # Handle data from dataset sources
        if isinstance(data, list):
            # Check if this is JSONL data with messages format
            if data and isinstance(data[0], dict) and "messages" in data[0]:
                return self._process_messages_data(data)
            return data

        # Handle dictionary data (from CSV files) or HuggingFace datasets
        prompt_column = self.dataset_config.prompt_column
        try:
            column_data = data[prompt_column]
            # Ensure we return a list of strings
            if isinstance(column_data, list):
                return [str(item) for item in column_data]
            else:
                # For HuggingFace datasets, convert to list
                return list(column_data)
        except (ValueError, KeyError) as e:
            # Provide helpful error message with available columns
            if isinstance(data, dict):
                available_columns = list(data.keys())
                raise ValueError(
                    f"Column '{prompt_column}' not found in CSV file. "
                    f"Available columns: {available_columns}"
                ) from e
            else:
                raise ValueError(
                    f"Cannot extract prompts from data: {type(data)}, error: {str(e)}"
                ) from e

    def _process_messages_data(
        self, data: List[Dict[str, Any]]
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """Process messages format data for benchmarking.
        
        Returns a list of tuples where each tuple contains:
        - prompt: The last user message (str)
        - additional_params: Dict with 'system_message' and 'chat_history' keys
        """
        results = []
        for item in data:
            messages = item.get("messages", [])
            if not messages:
                continue

            # Extract system message (if present)
            system_message = None
            chat_history = []
            last_user_message = None

            # Process messages in order
            for msg in messages:
                role = msg.get("role", "").lower()
                content = msg.get("content", "")

                if role == "system":
                    system_message = content
                elif role == "user":
                    # If we already found a user message, add previous messages to history
                    if last_user_message is not None:
                        chat_history.append({"role": "user", "content": last_user_message})
                    last_user_message = content
                elif role == "assistant":
                    # Add assistant response to history
                    if last_user_message is not None:
                        chat_history.append({"role": "user", "content": last_user_message})
                        chat_history.append({"role": "assistant", "content": content})
                        last_user_message = None

            # If we have a last user message, use it as the prompt
            if last_user_message:
                additional_params = {}
                if system_message:
                    additional_params["system_message"] = system_message
                if chat_history:
                    additional_params["chat_history"] = chat_history

                results.append((last_user_message, additional_params))

        return results
