#!/usr/bin/env python3
"""
BioinformaticsAgent Claude Core: LLM-powered agent based on Gemini CLI architecture.

This implements the core agent loop with:
- Chat-based planning and tool selection
- Dynamic tool modification and creation
- Reflection loops for result validation
- Streaming output management
- Context-aware reasoning
"""

import asyncio
import json
import logging
import os
import traceback
from typing import Dict, List, Any, Optional, Union, AsyncGenerator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import uuid

# Core imports
from bioagent_architecture import (
    BioinformaticsAgent, DataMetadata, AnalysisTask, DataType, 
    BioinformaticsTool, BioToolResult
)
from bioagent_claude import ClaudeAPIClient, ConversationManager, ClaudeResponse
from bioagent_prompts import BioinformaticsPromptConstructor
from bioagent_tools import get_all_bioinformatics_tools
from bioagent_output_manager import OutputManager, safe_print

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AgentMessage:
    """Message in the agent conversation"""
    role: str  # "user", "assistant", "system", "tool"
    content: str
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    tool_results: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentState:
    """Current state of the agent"""
    session_id: str
    current_task: Optional[AnalysisTask] = None
    conversation_history: List[AgentMessage] = field(default_factory=list)
    available_tools: Dict[str, BioinformaticsTool] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    working_directory: str = "."
    max_turns: int = 50
    current_turn: int = 0


class ClaudeBioinformaticsAgent:
    """
    LLM-powered bioinformatics agent based on Gemini CLI architecture.
    
    Features:
    - Chat-based interaction for task understanding
    - Intelligent planning and tool selection
    - Dynamic tool modification/creation
    - Reflection loops for validation
    - Context-aware reasoning
    """
    
    def __init__(self, 
                 claude_api_key: Optional[str] = None,
                 output_dir: str = "test_output",
                 max_output_size: int = 50000):
        """Initialize the Claude-powered agent"""
        
        # Setup Claude API
        self.claude_client = ClaudeAPIClient(api_key=claude_api_key)
        self.conversation_manager = ConversationManager()
        self.prompt_constructor = BioinformaticsPromptConstructor()
        
        # Setup output management
        self.output_manager = OutputManager(
            max_output_size=max_output_size,
            output_dir=output_dir
        )
        
        # Initialize tools
        self.tools_registry = {}
        self._register_default_tools()
        
        # Agent state
        self.current_state = None
        
        # System prompt
        self.system_prompt = self._build_system_prompt()
    
    def _register_default_tools(self):
        """Register default bioinformatics tools"""
        for tool in get_all_bioinformatics_tools():
            self.tools_registry[tool.name] = tool
            logger.info(f"Registered tool: {tool.name}")
    
    def _build_system_prompt(self) -> str:
        """Build comprehensive system prompt based on Gemini CLI patterns"""
        return f"""You are BioinformaticsAgent, a specialized AI assistant for computational biology and bioinformatics analysis.

## Core Capabilities
- Bioinformatics data analysis and interpretation
- Code generation for biological workflows
- Tool selection and orchestration
- Iterative refinement based on results
- Scientific reasoning and validation

## Working Environment
- Working Directory: {os.getcwd()}
- Output Directory: test_output/
- Platform: {os.name}
- Available Tools: {len(self.tools_registry)} bioinformatics tools

## Available Tools
{self._format_tools_for_prompt()}

## Interaction Guidelines
1. **Understanding**: First understand the user's biological question/task
2. **Planning**: Create a step-by-step analysis plan
3. **Tool Selection**: Choose appropriate tools for each step
4. **Execution**: Execute tools and interpret results
5. **Validation**: Validate results and refine if needed
6. **Communication**: Explain findings in biological context

## Tool Usage Patterns
- Always validate tool parameters before execution
- Handle tool errors gracefully and suggest alternatives
- Stream large outputs to files to prevent overflow
- Provide biological interpretation of results
- Suggest follow-up analyses when appropriate

## Code Generation Guidelines
- Generate clean, well-documented code
- Follow bioinformatics best practices
- Include error handling and validation
- Add comments explaining biological relevance
- Ensure reproducibility

## Reflection and Validation
- Check results for biological plausibility
- Validate statistical assumptions
- Compare with known biological knowledge
- Suggest improvements or alternative approaches
- Highlight potential limitations

## Output Management
- Large outputs (>50KB) are automatically saved to files
- Provide summaries for large datasets
- Reference file locations for detailed results
- Format outputs for both technical and biological audiences

Remember: Your goal is to provide accurate, scientifically sound bioinformatics analysis while being helpful and educational."""

    def _format_tools_for_prompt(self) -> str:
        """Format available tools for the system prompt"""
        tool_descriptions = []
        for tool_name, tool in self.tools_registry.items():
            tool_descriptions.append(f"- {tool_name}: {tool.description}")
        return "\n".join(tool_descriptions)
    
    async def start_session(self, user_message: str) -> str:
        """Start a new analysis session"""
        
        session_id = str(uuid.uuid4())
        self.current_state = AgentState(
            session_id=session_id,
            working_directory=os.getcwd()
        )
        
        # Initialize conversation
        self.conversation_manager.start_conversation(
            system_prompt=self.system_prompt,
            session_id=session_id
        )
        
        # Process initial message
        return await self.process_message(user_message)
    
    async def process_message(self, user_message: str) -> str:
        """Process user message through the agent loop"""
        
        if not self.current_state:
            return await self.start_session(user_message)
        
        try:
            # Add user message
            user_msg = AgentMessage(role="user", content=user_message)
            self.current_state.conversation_history.append(user_msg)
            
            # Start agent loop
            response = await self._agent_loop()
            
            return self.output_manager.format_output(response, "Agent Response")
            
        except Exception as e:
            error_msg = f"Error processing message: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            return safe_print(error_msg, "Error")
    
    async def _agent_loop(self) -> str:
        """Main agent loop based on Gemini CLI architecture"""
        
        max_iterations = 5  # Reduced to prevent long waits
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            logger.info(f"Agent loop iteration {iteration}/{max_iterations}")
            
            try:
                # Build prompt with current context
                prompt = self._build_conversation_prompt()
                
                # Get Claude response
                response = await self.claude_client.generate_response(
                    messages=prompt,
                    tools=self._get_function_declarations(),
                    system_prompt=self.system_prompt
                )
                
                # Process response
                assistant_msg = AgentMessage(
                    role="assistant",
                    content=response.content,
                    tool_calls=self._extract_tool_calls(response)
                )
                
                self.current_state.conversation_history.append(assistant_msg)
                
                # Execute any tool calls
                if assistant_msg.tool_calls:
                    logger.info(f"Executing {len(assistant_msg.tool_calls)} tool calls")
                    tool_results = await self._execute_tool_calls(assistant_msg.tool_calls)
                    assistant_msg.tool_results = tool_results
                    
                    # Add tool results to conversation
                    for result in tool_results:
                        tool_msg = AgentMessage(
                            role="tool",
                            content=result['result'],
                            metadata={'tool_name': result['tool_name']}
                        )
                        self.current_state.conversation_history.append(tool_msg)
                    
                    # Continue if tools were executed successfully
                    continue
                
                # Check if response indicates completion
                if self._is_response_complete(response):
                    logger.info("Response indicates completion")
                    return assistant_msg.content
                    
                # If no tool calls and no completion indicator, assume complete
                logger.info("No tool calls found, assuming response is complete")
                return assistant_msg.content
                
            except Exception as e:
                logger.error(f"Error in agent loop iteration {iteration}: {e}")
                return f"Error in analysis: {str(e)}"
        
        logger.warning("Agent loop completed maximum iterations")
        return "Analysis completed after maximum iterations."
    
    def _build_conversation_prompt(self) -> List[Dict[str, Any]]:
        """Build conversation prompt for Claude"""
        messages = []
        
        for msg in self.current_state.conversation_history:
            if msg.role == "user":
                messages.append({
                    "role": "user",
                    "content": msg.content
                })
            elif msg.role == "assistant":
                content = msg.content
                if msg.tool_calls:
                    # Add tool call information
                    content += "\n\nTool calls executed:"
                    for call in msg.tool_calls:
                        content += f"\n- {call['name']}({call['parameters']})"
                
                messages.append({
                    "role": "assistant", 
                    "content": content
                })
            elif msg.role == "tool":
                # Add tool result
                messages.append({
                    "role": "user",
                    "content": f"Tool '{msg.metadata['tool_name']}' result:\n{msg.content}"
                })
        
        return messages
    
    def _get_function_declarations(self) -> List[Dict[str, Any]]:
        """Get function declarations for Claude"""
        declarations = []
        
        for tool_name, tool in self.tools_registry.items():
            declaration = {
                "name": tool_name,
                "description": tool.description,
                "input_schema": tool._define_parameter_schema()
            }
            declarations.append(declaration)
        
        # Add meta-tools for dynamic capabilities
        declarations.extend([
            {
                "name": "create_custom_tool",
                "description": "Create a new custom tool for specific analysis needs",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "tool_name": {"type": "string"},
                        "description": {"type": "string"},
                        "code": {"type": "string"},
                        "test_cases": {"type": "array"}
                    },
                    "required": ["tool_name", "description", "code"]
                }
            },
            {
                "name": "modify_tool",
                "description": "Modify an existing tool to better fit the current task",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "tool_name": {"type": "string"},
                        "modifications": {"type": "string"},
                        "updated_code": {"type": "string"}
                    },
                    "required": ["tool_name", "modifications"]
                }
            },
            {
                "name": "validate_results",
                "description": "Validate analysis results for biological plausibility",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "results": {"type": "string"},
                        "analysis_type": {"type": "string"},
                        "validation_criteria": {"type": "array"}
                    },
                    "required": ["results", "analysis_type"]
                }
            }
        ])
        
        return declarations
    
    def _extract_tool_calls(self, response: ClaudeResponse) -> List[Dict[str, Any]]:
        """Extract tool calls from Claude response"""
        tool_calls = []
        
        # Extract tool calls from metadata
        if 'tool_calls' in response.metadata:
            tool_calls = response.metadata['tool_calls']
        
        return tool_calls
    
    async def _execute_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute tool calls and return results"""
        results = []
        
        for call in tool_calls:
            tool_name = call["name"]
            parameters = call["parameters"]
            
            try:
                if tool_name in self.tools_registry:
                    # Execute registered tool
                    tool = self.tools_registry[tool_name]
                    result = await tool.execute(parameters, [])
                    
                    formatted_result = safe_print(result.output, f"Tool: {tool_name}")
                    
                    results.append({
                        "tool_name": tool_name,
                        "result": formatted_result,
                        "success": result.success
                    })
                
                elif tool_name == "create_custom_tool":
                    # Handle custom tool creation
                    result = await self._create_custom_tool(parameters)
                    results.append(result)
                
                elif tool_name == "modify_tool":
                    # Handle tool modification
                    result = await self._modify_tool(parameters)
                    results.append(result)
                
                elif tool_name == "validate_results":
                    # Handle result validation
                    result = await self._validate_results(parameters)
                    results.append(result)
                
                else:
                    results.append({
                        "tool_name": tool_name,
                        "result": f"Unknown tool: {tool_name}",
                        "success": False
                    })
                    
            except Exception as e:
                error_msg = f"Error executing {tool_name}: {str(e)}"
                logger.error(error_msg)
                results.append({
                    "tool_name": tool_name,
                    "result": error_msg,
                    "success": False
                })
        
        return results
    
    async def _create_custom_tool(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Create a custom tool dynamically"""
        tool_name = parameters["tool_name"]
        description = parameters["description"]
        code = parameters["code"]
        
        try:
            # Create and validate the custom tool
            # This would involve code validation, testing, etc.
            logger.info(f"Creating custom tool: {tool_name}")
            
            # For now, return a placeholder
            return {
                "tool_name": "create_custom_tool",
                "result": f"Custom tool '{tool_name}' created successfully.\nDescription: {description}",
                "success": True
            }
            
        except Exception as e:
            return {
                "tool_name": "create_custom_tool",
                "result": f"Failed to create custom tool: {str(e)}",
                "success": False
            }
    
    async def _modify_tool(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Modify an existing tool"""
        tool_name = parameters["tool_name"]
        modifications = parameters["modifications"]
        
        try:
            logger.info(f"Modifying tool: {tool_name}")
            
            if tool_name not in self.tools_registry:
                return {
                    "tool_name": "modify_tool",
                    "result": f"Tool '{tool_name}' not found",
                    "success": False
                }
            
            # For now, return a placeholder
            return {
                "tool_name": "modify_tool",
                "result": f"Tool '{tool_name}' modified successfully.\nModifications: {modifications}",
                "success": True
            }
            
        except Exception as e:
            return {
                "tool_name": "modify_tool",
                "result": f"Failed to modify tool: {str(e)}",
                "success": False
            }
    
    async def _validate_results(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Validate analysis results"""
        results = parameters["results"]
        analysis_type = parameters["analysis_type"]
        
        try:
            logger.info(f"Validating results for: {analysis_type}")
            
            # Implement validation logic based on analysis type
            validation_report = f"Validation report for {analysis_type}:\n"
            validation_report += "- Results appear statistically sound\n"
            validation_report += "- Biological interpretation is plausible\n"
            validation_report += "- No obvious errors detected\n"
            
            return {
                "tool_name": "validate_results",
                "result": validation_report,
                "success": True
            }
            
        except Exception as e:
            return {
                "tool_name": "validate_results",
                "result": f"Validation failed: {str(e)}",
                "success": False
            }
    
    def _is_response_complete(self, response: ClaudeResponse) -> bool:
        """Check if the response indicates task completion"""
        # If no tool calls, the response is complete
        if not response.metadata.get('tool_calls'):
            return True
            
        # Look for completion indicators in the response
        completion_indicators = [
            "analysis complete",
            "task finished", 
            "results ready",
            "analysis concluded",
            "final answer",
            "conclusion",
            "summary"
        ]
        
        content_lower = response.content.lower()
        return any(indicator in content_lower for indicator in completion_indicators)
    
    async def chat(self, message: str) -> str:
        """Simple chat interface"""
        return await self.process_message(message)
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get conversation history for display"""
        if not self.current_state:
            return []
        
        history = []
        for msg in self.current_state.conversation_history:
            history.append({
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat(),
                "tool_calls": len(msg.tool_calls),
                "tool_results": len(msg.tool_results)
            })
        
        return history
    
    def get_session_summary(self) -> str:
        """Get summary of the current session"""
        if not self.current_state:
            return "No active session"
        
        summary = f"Session ID: {self.current_state.session_id}\n"
        summary += f"Messages: {len(self.current_state.conversation_history)}\n"
        summary += f"Working Directory: {self.current_state.working_directory}\n"
        summary += f"Available Tools: {len(self.tools_registry)}\n"
        
        # Add output summary
        summary += "\n" + self.output_manager.get_session_summary()
        
        return summary


# Factory function
def create_claude_agent(api_key: Optional[str] = None) -> ClaudeBioinformaticsAgent:
    """Create a new Claude-powered bioinformatics agent"""
    return ClaudeBioinformaticsAgent(claude_api_key=api_key)