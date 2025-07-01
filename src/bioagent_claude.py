#!/usr/bin/env python3
"""
BioinformaticsAgent Claude Integration: Claude API client for LLM-powered analysis.

This module provides:
- Claude API integration with proper authentication
- Conversation management and context handling
- Specialized prompts for bioinformatics tasks
- Error handling and retry logic
- Token usage tracking and optimization
"""

import asyncio
import json
import logging
import os
import time
from typing import Dict, List, Any, Optional, AsyncGenerator
from dataclasses import dataclass, field
from datetime import datetime
import anthropic
from anthropic import AsyncAnthropic

# Import base classes
from bioagent_architecture import DataMetadata, AnalysisTask
from bioagent_prompts import BioinformaticsPromptConstructor


@dataclass
class ConversationMessage:
    """Represents a message in the conversation"""
    role: str  # "user", "assistant", "system"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ClaudeResponse:
    """Structured response from Claude API"""
    content: str
    usage: Dict[str, int]
    model: str
    finish_reason: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class ClaudeAPIClient:
    """Claude API client with enhanced functionality"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize Claude API client"""
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Claude API key not provided. Set ANTHROPIC_API_KEY environment variable.")
        
        try:
            self.client = AsyncAnthropic(api_key=self.api_key)
        except Exception as e:
            # Handle version compatibility issues
            logging.warning(f"Anthropic client initialization issue: {e}")
            # Try with minimal parameters
            try:
                self.client = AsyncAnthropic(api_key=self.api_key)
            except:
                logging.error("Failed to initialize Anthropic client")
                self.client = None
        
        # Use Claude Sonnet 4 (latest model)
        self.model = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514")
        self.max_tokens = int(os.getenv("CLAUDE_MAX_TOKENS", "20000"))
        self.thinking_tokens = int(os.getenv("CLAUDE_THINKING_TOKENS", "16000"))
        self.temperature = float(os.getenv("CLAUDE_TEMPERATURE", "1"))
        
    async def generate_response(self, 
                              messages: List[Dict[str, Any]], 
                              tools: Optional[List[Dict[str, Any]]] = None,
                              system_prompt: Optional[str] = None) -> ClaudeResponse:
        """Generate response from Claude"""
        try:
            # Prepare the request for Claude Sonnet 4
            request_params = {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "messages": messages,
                "thinking": {
                    "type": "enabled",
                    "budget_tokens": self.thinking_tokens
                }
            }
            
            if system_prompt:
                request_params["system"] = system_prompt
            
            if tools:
                request_params["tools"] = tools
            
            # Make API call
            response = await self.client.messages.create(**request_params)
            
            # Parse response
            content = ""
            tool_calls = []
            thinking_content = ""
            
            for content_block in response.content:
                if content_block.type == "text":
                    content += content_block.text
                elif content_block.type == "tool_use":
                    tool_calls.append({
                        "name": content_block.name,
                        "parameters": content_block.input
                    })
                elif content_block.type == "thinking":
                    # Skip thinking content for now - we'll handle it later
                    pass
            
            return ClaudeResponse(
                content=content,
                usage={
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens
                },
                model=response.model,
                finish_reason=response.stop_reason,
                metadata={
                    "tool_calls": tool_calls,
                    "thinking": thinking_content
                }
            )
            
        except Exception as e:
            logging.error(f"Claude API error: {e}")
            raise


class ConversationManager:
    """Manages conversation history and context for Claude interactions"""
    
    def __init__(self, max_context_length: int = 150000):
        self.messages: List[ConversationMessage] = []
        self.max_context_length = max_context_length
        self.system_prompt: Optional[str] = None
        
    def add_message(self, role: str, content: str, metadata: Dict[str, Any] = None):
        """Add a message to the conversation"""
        message = ConversationMessage(
            role=role,
            content=content,
            metadata=metadata or {}
        )
        self.messages.append(message)
        
    def set_system_prompt(self, prompt: str):
        """Set the system prompt for the conversation"""
        self.system_prompt = prompt
        
    def get_messages_for_api(self) -> List[Dict[str, str]]:
        """Convert conversation to format expected by Claude API"""
        api_messages = []
        
        # Add conversation messages (excluding system messages)
        for msg in self.messages:
            if msg.role != "system":  # System prompt handled separately
                api_messages.append({
                    "role": msg.role,
                    "content": msg.content
                })
                
        return api_messages
    
    def trim_context(self):
        """Trim conversation to fit within context limits"""
        # Estimate token count (rough approximation: 1 token ≈ 4 characters)
        total_chars = sum(len(msg.content) for msg in self.messages)
        if self.system_prompt:
            total_chars += len(self.system_prompt)
            
        # If too long, remove oldest non-system messages
        while total_chars > self.max_context_length * 4 and len(self.messages) > 1:
            # Keep the most recent messages, remove older ones
            if self.messages[0].role != "system":
                removed = self.messages.pop(0)
                total_chars -= len(removed.content)
            else:
                # If first message is system, remove the second one
                if len(self.messages) > 1:
                    removed = self.messages.pop(1)
                    total_chars -= len(removed.content)
                else:
                    break
                    
    def start_conversation(self, system_prompt: str, session_id: str):
        """Start a new conversation with system prompt"""
        self.system_prompt = system_prompt
        self.messages = []
        
    def clear(self):
        """Clear conversation history"""
        self.messages.clear()


class ClaudeClient:
    """Claude API client for bioinformatics agent"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = None):
        # Load environment variables
        from dotenv import load_dotenv
        load_dotenv()
        
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Anthropic API key not provided. Please:\n"
                "1. Set ANTHROPIC_API_KEY environment variable, or\n"
                "2. Create a .env file with ANTHROPIC_API_KEY=your-key, or\n"
                "3. Pass api_key parameter to ClaudeClient()"
            )
            
        self.model = model or os.getenv("CLAUDE_MODEL", "claude-3-5-sonnet-20241022")
        self.client = AsyncAnthropic(api_key=self.api_key)
        self.conversation = ConversationManager()
        self.prompt_constructor = BioinformaticsPromptConstructor()
        
        # Usage tracking
        self.total_tokens_used = 0
        self.request_count = 0
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Log successful initialization (without exposing key)
        masked_key = self.api_key[:8] + "..." + self.api_key[-4:] if len(self.api_key) > 12 else "***"
        self.logger.info(f"Claude client initialized with key: {masked_key}, model: {self.model}")
        
    async def chat(self, message: str, context: Dict[str, Any] = None) -> ClaudeResponse:
        """Send a chat message to Claude and get response"""
        
        # Add user message to conversation
        self.conversation.add_message("user", message, metadata=context)
        
        # Trim context if necessary
        self.conversation.trim_context()
        
        try:
            # Prepare API call
            api_messages = self.conversation.get_messages_for_api()
            
            kwargs = {
                "model": self.model,
                "max_tokens": 4096,
                "messages": api_messages
            }
            
            # Add system prompt if available
            if self.conversation.system_prompt:
                kwargs["system"] = self.conversation.system_prompt
                
            # Make API call
            response = await self.client.messages.create(**kwargs)
            
            # Extract response content
            content = ""
            if hasattr(response, 'content') and response.content:
                content = response.content[0].text if response.content else ""
            
            # Create structured response
            claude_response = ClaudeResponse(
                content=content,
                usage=response.usage.model_dump() if hasattr(response, 'usage') else {},
                model=response.model if hasattr(response, 'model') else self.model,
                finish_reason=response.stop_reason if hasattr(response, 'stop_reason') else "complete"
            )
            
            # Add Claude's response to conversation
            self.conversation.add_message("assistant", content)
            
            # Update usage tracking
            if claude_response.usage:
                self.total_tokens_used += claude_response.usage.get('input_tokens', 0) + claude_response.usage.get('output_tokens', 0)
            self.request_count += 1
            
            self.logger.info(f"Claude API call successful. Tokens used: {claude_response.usage}")
            
            return claude_response
            
        except Exception as e:
            self.logger.error(f"Claude API call failed: {e}")
            raise
            
    async def generate_analysis_plan(self, task: AnalysisTask, available_tools: List[str]) -> Dict[str, Any]:
        """Generate an analysis plan for a bioinformatics task"""
        
        # Build planning prompt
        planning_prompt = self.prompt_constructor.create_planning_prompt(
            task=task,
            available_tools=available_tools
        )
        
        # Set system prompt for planning
        system_prompt = self.prompt_constructor.create_system_prompt(
            mode="planning",
            available_tools=available_tools
        )
        self.conversation.set_system_prompt(system_prompt)
        
        # Get plan from Claude
        response = await self.chat(planning_prompt)
        
        # Parse the plan (expecting JSON response)
        try:
            plan = json.loads(response.content)
            return {
                "success": True,
                "plan": plan,
                "raw_response": response.content,
                "usage": response.usage
            }
        except json.JSONDecodeError:
            # If not JSON, return as structured text
            return {
                "success": True,
                "plan": {"description": response.content, "steps": []},
                "raw_response": response.content,
                "usage": response.usage
            }
            
    async def generate_code(self, task: AnalysisTask, plan: Dict[str, Any], 
                          data_metadata: List[DataMetadata]) -> Dict[str, Any]:
        """Generate Python code for bioinformatics analysis"""
        
        # Build code generation prompt
        code_prompt = self.prompt_constructor.create_code_generation_prompt(
            task=task,
            plan=plan,
            data_metadata=data_metadata
        )
        
        # Set system prompt for code generation
        system_prompt = self.prompt_constructor.create_system_prompt(mode="code_generation")
        self.conversation.set_system_prompt(system_prompt)
        
        # Get code from Claude
        response = await self.chat(code_prompt)
        
        # Extract code from response (handle code blocks)
        code = self._extract_code_from_response(response.content)
        
        return {
            "success": True,
            "code": code,
            "explanation": response.content,
            "usage": response.usage
        }
        
    async def analyze_results(self, task: AnalysisTask, results: Dict[str, Any], 
                            code: str) -> Dict[str, Any]:
        """Analyze execution results and provide feedback"""
        
        # Build results analysis prompt
        analysis_prompt = self.prompt_constructor.create_results_analysis_prompt(
            task=task,
            results=results,
            code=code
        )
        
        # Set system prompt for analysis
        system_prompt = self.prompt_constructor.create_system_prompt(mode="results_analysis")
        self.conversation.set_system_prompt(system_prompt)
        
        # Get analysis from Claude
        response = await self.chat(analysis_prompt)
        
        return {
            "success": True,
            "analysis": response.content,
            "usage": response.usage
        }
        
    async def suggest_tool_modification(self, tool_name: str, current_implementation: str,
                                     requirements: str) -> Dict[str, Any]:
        """Suggest modifications to an existing tool"""
        
        modification_prompt = f"""
        I need to modify the bioinformatics tool '{tool_name}' to meet new requirements.
        
        Current Implementation:
        ```python
        {current_implementation}
        ```
        
        New Requirements:
        {requirements}
        
        Please provide the modified tool implementation that meets these requirements while maintaining compatibility with the existing tool interface.
        """
        
        # Set system prompt for tool modification
        system_prompt = self.prompt_constructor.create_system_prompt(mode="tool_modification")
        self.conversation.set_system_prompt(system_prompt)
        
        response = await self.chat(modification_prompt)
        
        # Extract modified code
        modified_code = self._extract_code_from_response(response.content)
        
        return {
            "success": True,
            "modified_code": modified_code,
            "explanation": response.content,
            "usage": response.usage
        }
        
    async def create_new_tool(self, tool_description: str, requirements: str,
                            example_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create a completely new bioinformatics tool"""
        
        creation_prompt = f"""
        I need to create a new bioinformatics tool with the following description:
        {tool_description}
        
        Requirements:
        {requirements}
        
        Please create a complete tool implementation including:
        1. Tool class that inherits from BioinformaticsTool
        2. Parameter schema definition
        3. Execute method implementation
        4. Error handling and validation
        5. Benchmark/test function
        
        {f"Example data for testing: {json.dumps(example_data, indent=2)}" if example_data else ""}
        """
        
        # Set system prompt for tool creation
        system_prompt = self.prompt_constructor.create_system_prompt(mode="tool_creation")
        self.conversation.set_system_prompt(system_prompt)
        
        response = await self.chat(creation_prompt)
        
        # Extract tool code and benchmark
        tool_code = self._extract_code_from_response(response.content)
        
        return {
            "success": True,
            "tool_code": tool_code,
            "explanation": response.content,
            "usage": response.usage
        }
        
    def _extract_code_from_response(self, content: str) -> str:
        """Extract Python code from Claude's response"""
        
        # Look for code blocks
        if "```python" in content:
            start = content.find("```python") + 9
            end = content.find("```", start)
            if end != -1:
                return content[start:end].strip()
                
        elif "```" in content:
            start = content.find("```") + 3
            end = content.find("```", start)
            if end != -1:
                return content[start:end].strip()
        
        # If no code blocks found, return the entire content
        return content.strip()
        
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get API usage statistics"""
        return {
            "total_tokens": self.total_tokens_used,
            "request_count": self.request_count,
            "average_tokens_per_request": self.total_tokens_used / max(self.request_count, 1)
        }
        
    def reset_conversation(self):
        """Reset the conversation history"""
        self.conversation.clear()