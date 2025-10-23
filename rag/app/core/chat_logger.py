import csv
import aiofiles
from datetime import datetime
import os
import asyncio
from typing import Dict, Any

# Define the path for the log file
LOG_FILE = "chat_history.csv"
LOG_COLUMNS = [
    "timestamp",
    "query",
    "model_name",
    "llm_response", 
    "retrieved_context",
    "response_time"
]

class ChatLogger:
    """A singleton-like class to manage asynchronous writing to the chat history CSV."""
    
    def __init__(self):
        if not os.path.exists(LOG_FILE):
            print(f"Creating new log file: {LOG_FILE}")
            with open(LOG_FILE, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(LOG_COLUMNS)

    async def log_chat_entry(self, entry: Dict[str, Any]):
        """
        Asynchronously writes a single chat entry to the CSV file.
        Uses asyncio to prevent blocking the main FastAPI thread.
        """
        entry['timestamp'] = datetime.now().isoformat()
        
        row_data = [entry.get(col, "") for col in LOG_COLUMNS]

        try:
            async with aiofiles.open(LOG_FILE, mode='a', newline='', encoding='utf-8') as f:
                await asyncio.to_thread(self._sync_write_row, row_data)
        except Exception as e:
            # Handle potential logging errors without breaking the main chat functionality
            print(f"Error writing to chat log file: {e}")

    def _sync_write_row(self, row_data: list):
        """Internal synchronous method to handle the file write."""
        with open(LOG_FILE, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(row_data)
