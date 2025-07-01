#!/usr/bin/env python3
"""
Output Manager: Handles large output streaming and prevents string overflow issues.

This module prevents the JavaScript string length overflow error by:
1. Limiting output size per message
2. Streaming large outputs to files
3. Chunking massive data structures
4. Providing summary views for large results
"""

import os
import json
import logging
from typing import Any, Dict, List, Optional, Union, TextIO
from pathlib import Path
from datetime import datetime
import tempfile
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OutputManager:
    """Manages output size and prevents CLI overflow"""
    
    def __init__(self, 
                 max_output_size: int = 50000,  # 50KB max per output
                 output_dir: str = "test_output",
                 enable_streaming: bool = True):
        self.max_output_size = max_output_size
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.enable_streaming = enable_streaming
        
        # Track outputs for session
        self.session_outputs = []
        self.file_counter = 0
    
    def format_output(self, content: Any, context: str = "") -> str:
        """Format output with size limits and streaming"""
        
        # Convert to string if needed
        if not isinstance(content, str):
            if hasattr(content, 'to_string'):
                content_str = str(content.to_string())
            else:
                content_str = str(content)
        else:
            content_str = content
        
        # Check size
        if len(content_str) <= self.max_output_size:
            return self._format_small_output(content_str, context)
        else:
            return self._handle_large_output(content_str, context)
    
    def _format_small_output(self, content: str, context: str) -> str:
        """Format small output directly"""
        if context:
            return f"=== {context} ===\n{content}\n"
        return content
    
    def _handle_large_output(self, content: str, context: str) -> str:
        """Handle large output by streaming to file"""
        
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        filename = f"output_{timestamp}_{content_hash}.txt"
        filepath = self.output_dir / filename
        
        # Write to file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        # Create summary
        summary = self._create_summary(content)
        
        # Track output
        self.session_outputs.append({
            'context': context,
            'filepath': str(filepath),
            'size': len(content),
            'timestamp': timestamp
        })
        
        return f"""=== {context} ===
Output too large for display ({len(content):,} characters).
Saved to: {filepath}

SUMMARY:
{summary}

To view full output:
- cat {filepath}
- Or open {filepath} in your editor
"""
    
    def _create_summary(self, content: str, max_lines: int = 20) -> str:
        """Create summary of large content"""
        lines = content.split('\n')
        
        if len(lines) <= max_lines:
            return content[:1000] + "..." if len(content) > 1000 else content
        
        # Show first and last lines
        first_lines = lines[:max_lines//2]
        last_lines = lines[-max_lines//2:]
        
        summary = '\n'.join(first_lines)
        summary += f"\n... [{len(lines) - max_lines} lines omitted] ...\n"
        summary += '\n'.join(last_lines)
        
        return summary
    
    def format_dataframe(self, df, context: str = "DataFrame") -> str:
        """Format pandas DataFrame with size awareness"""
        try:
            import pandas as pd
            
            if not isinstance(df, pd.DataFrame):
                return self.format_output(str(df), context)
            
            # Get basic info
            info = f"Shape: {df.shape}\n"
            info += f"Columns: {list(df.columns)}\n"
            info += f"Data types: {dict(df.dtypes)}\n"
            
            # Show sample
            if df.shape[0] > 10:
                sample = df.head(5).to_string() + "\n...\n" + df.tail(5).to_string()
            else:
                sample = df.to_string()
            
            formatted = f"{info}\nSample data:\n{sample}"
            
            return self.format_output(formatted, context)
            
        except ImportError:
            return self.format_output(str(df), context)
    
    def format_array(self, arr, context: str = "Array") -> str:
        """Format numpy array with size awareness"""
        try:
            import numpy as np
            
            if not isinstance(arr, np.ndarray):
                return self.format_output(str(arr), context)
            
            # Get basic info
            info = f"Shape: {arr.shape}\n"
            info += f"Data type: {arr.dtype}\n"
            info += f"Size: {arr.size} elements\n"
            
            # Show sample
            if arr.size > 100:
                flat = arr.flatten()
                sample = f"First 10: {flat[:10]}\nLast 10: {flat[-10:]}"
            else:
                sample = str(arr)
            
            formatted = f"{info}\nSample data:\n{sample}"
            
            return self.format_output(formatted, context)
            
        except ImportError:
            return self.format_output(str(arr), context)
    
    def format_plot_result(self, plot_info: Dict[str, Any]) -> str:
        """Format plot result information"""
        result = "=== Plot Generated ===\n"
        
        if 'filepath' in plot_info:
            result += f"Saved to: {plot_info['filepath']}\n"
        
        if 'title' in plot_info:
            result += f"Title: {plot_info['title']}\n"
        
        if 'description' in plot_info:
            result += f"Description: {plot_info['description']}\n"
        
        if 'stats' in plot_info:
            result += f"Statistics: {plot_info['stats']}\n"
        
        return result
    
    def get_session_summary(self) -> str:
        """Get summary of all outputs in session"""
        if not self.session_outputs:
            return "No large outputs generated in this session."
        
        summary = "=== Session Output Summary ===\n"
        for i, output in enumerate(self.session_outputs, 1):
            summary += f"{i}. {output['context']}\n"
            summary += f"   File: {output['filepath']}\n"
            summary += f"   Size: {output['size']:,} characters\n"
            summary += f"   Time: {output['timestamp']}\n\n"
        
        return summary
    
    def cleanup_old_outputs(self, max_age_hours: int = 24):
        """Clean up old output files"""
        try:
            from datetime import timedelta
            
            cutoff = datetime.now() - timedelta(hours=max_age_hours)
            
            for file_path in self.output_dir.glob("output_*.txt"):
                if file_path.stat().st_mtime < cutoff.timestamp():
                    file_path.unlink()
                    logger.info(f"Cleaned up old output: {file_path}")
                    
        except Exception as e:
            logger.warning(f"Error cleaning up outputs: {e}")


# Global instance
output_manager = OutputManager()

# Convenience functions
def safe_print(content: Any, context: str = "") -> str:
    """Safe print that prevents overflow"""
    return output_manager.format_output(content, context)

def safe_print_df(df, context: str = "DataFrame") -> str:
    """Safe DataFrame print"""
    return output_manager.format_dataframe(df, context)

def safe_print_array(arr, context: str = "Array") -> str:
    """Safe array print"""
    return output_manager.format_array(arr, context)

def format_plot(plot_info: Dict[str, Any]) -> str:
    """Format plot information"""
    return output_manager.format_plot_result(plot_info)

def get_session_summary() -> str:
    """Get session output summary"""
    return output_manager.get_session_summary()