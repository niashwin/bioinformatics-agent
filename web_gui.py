#!/usr/bin/env python3
"""
Web GUI for Claude Sonnet 4 BioinformaticsAgent

A modern web interface that provides:
- Real-time chat with the agent
- File upload for bioinformatics data
- Output visualization and download
- Thinking process display
- Session management
"""

import os
import sys
import asyncio
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# Flask and web dependencies
from flask import Flask, render_template, request, jsonify, send_file, session
from flask_socketio import SocketIO, emit
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Load environment
try:
    from dotenv import load_dotenv
    if Path(".env.local").exists():
        load_dotenv(".env.local")
        print("✅ Environment loaded from .env.local")
except ImportError:
    print("⚠️  python-dotenv not available")

# Import agent components
try:
    from bioagent_claude_core import create_claude_agent
    from bioagent_output_manager import OutputManager
    print("✅ BioinformaticsAgent components loaded")
except ImportError as e:
    print(f"❌ Failed to import agent components: {e}")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flask app setup
app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

# SocketIO for real-time communication
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global agent instance and session management
agents: Dict[str, Any] = {}
output_managers: Dict[str, OutputManager] = {}


class WebGUIManager:
    """Manages the web GUI sessions and agent interactions"""
    
    def __init__(self):
        self.sessions = {}
        
    def create_session(self, session_id: str):
        """Create a new agent session"""
        try:
            # Create agent for this session
            agent = create_claude_agent()
            output_manager = OutputManager(output_dir=f"web_output/{session_id}")
            
            self.sessions[session_id] = {
                'agent': agent,
                'output_manager': output_manager,
                'created': datetime.now(),
                'messages': [],
                'thinking_enabled': False
            }
            
            logger.info(f"Created new session: {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create session {session_id}: {e}")
            return False
    
    def get_session(self, session_id: str):
        """Get existing session or create new one"""
        if session_id not in self.sessions:
            if not self.create_session(session_id):
                return None
        return self.sessions[session_id]
    
    def cleanup_old_sessions(self, max_age_hours: int = 24):
        """Clean up old sessions"""
        cutoff = datetime.now().timestamp() - (max_age_hours * 3600)
        
        to_remove = []
        for session_id, session_data in self.sessions.items():
            if session_data['created'].timestamp() < cutoff:
                to_remove.append(session_id)
        
        for session_id in to_remove:
            del self.sessions[session_id]
            logger.info(f"Cleaned up old session: {session_id}")


# Global GUI manager
gui_manager = WebGUIManager()


@app.route('/')
def index():
    """Serve the main web interface"""
    # Generate session ID if not exists
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    
    return render_template('index.html')


@app.route('/api/status')
def api_status():
    """Get system status"""
    try:
        # Check if Claude API is configured
        api_key = os.getenv("ANTHROPIC_API_KEY")
        model = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514")
        
        status = {
            'status': 'ready',
            'api_configured': bool(api_key),
            'model': model,
            'session_id': session.get('session_id'),
            'active_sessions': len(gui_manager.sessions),
            'timestamp': datetime.now().isoformat()
        }
        
        if api_key:
            status['api_key_preview'] = f"{'*' * 20}{api_key[-4:]}"
        
        return jsonify(status)
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/tools')
def api_tools():
    """Get available bioinformatics tools"""
    try:
        session_id = session.get('session_id')
        if not session_id:
            return jsonify({'error': 'No session'}), 400
        
        session_data = gui_manager.get_session(session_id)
        if not session_data:
            return jsonify({'error': 'Session creation failed'}), 500
        
        agent = session_data['agent']
        tools = []
        
        for tool_name, tool in agent.tools_registry.items():
            tools.append({
                'name': tool_name,
                'description': tool.description,
                'supported_data_types': [dt.value for dt in tool.supported_data_types]
            })
        
        return jsonify({'tools': tools})
        
    except Exception as e:
        logger.error(f"Error getting tools: {e}")
        return jsonify({'error': str(e)}), 500


@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    logger.info(f"Client connected: {request.sid}")
    emit('status', {'message': 'Connected to BioinformaticsAgent', 'type': 'success'})


@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    logger.info(f"Client disconnected: {request.sid}")


@socketio.on('chat_message')
def handle_chat_message(data):
    """Handle chat message from client"""
    try:
        session_id = data.get('session_id')
        message = data.get('message', '').strip()
        
        if not session_id or not message:
            emit('error', {'message': 'Invalid message data'})
            return
        
        # Get or create session
        session_data = gui_manager.get_session(session_id)
        if not session_data:
            emit('error', {'message': 'Failed to create agent session'})
            return
        
        agent = session_data['agent']
        
        # Add user message to session history
        user_msg = {
            'role': 'user',
            'content': message,
            'timestamp': datetime.now().isoformat()
        }
        session_data['messages'].append(user_msg)
        
        # Emit user message confirmation
        emit('message_received', user_msg)
        
        # Show thinking indicator
        emit('agent_thinking', {'thinking': True})
        
        # CRITICAL FIX: Capture request.sid before background task
        client_sid = request.sid
        logger.info(f"Processing message for client: {client_sid}")
        
        # Process message asynchronously using proper asyncio handling
        async def process_message_async():
            try:
                logger.info(f"Starting to process message: {message[:50]}...")
                
                # Call agent directly without creating new event loop
                logger.info("Calling agent.process_message...")
                response = await agent.process_message(message)
                logger.info(f"Agent response received: {len(response)} characters")
                
                # Format response using output manager
                output_manager = session_data['output_manager']
                formatted_response = output_manager.format_output(response, "Agent Response")
                logger.info(f"Response formatted: {len(formatted_response)} characters")
                
                # Create agent message
                agent_msg = {
                    'role': 'assistant',
                    'content': formatted_response,
                    'timestamp': datetime.now().isoformat()
                }
                session_data['messages'].append(agent_msg)
                
                # Emit agent response using captured client_sid
                logger.info(f"Emitting agent response to client: {client_sid}")
                socketio.emit('agent_response', agent_msg, room=client_sid)
                socketio.emit('agent_thinking', {'thinking': False}, room=client_sid)
                logger.info("Response emitted successfully")
                
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                
                error_msg = {
                    'role': 'system',
                    'content': f"Error: {str(e)}",
                    'timestamp': datetime.now().isoformat()
                }
                socketio.emit('agent_response', error_msg, room=client_sid)
                socketio.emit('agent_thinking', {'thinking': False}, room=client_sid)
        
        # Wrapper to run async function in background thread
        def run_async_task():
            try:
                # Run the async function
                asyncio.run(process_message_async())
            except Exception as e:
                logger.error(f"Background task error: {e}")
                error_msg = {
                    'role': 'system',
                    'content': f"Processing error: {str(e)}",
                    'timestamp': datetime.now().isoformat()
                }
                socketio.emit('agent_response', error_msg, room=client_sid)
                socketio.emit('agent_thinking', {'thinking': False}, room=client_sid)
        
        # Start processing in background thread
        socketio.start_background_task(run_async_task)
        
    except Exception as e:
        logger.error(f"Error handling chat message: {e}")
        emit('error', {'message': str(e)})


@socketio.on('toggle_thinking')
def handle_toggle_thinking(data):
    """Toggle thinking display for session"""
    try:
        session_id = data.get('session_id')
        if not session_id:
            emit('error', {'message': 'No session ID'})
            return
        
        session_data = gui_manager.get_session(session_id)
        if session_data:
            session_data['thinking_enabled'] = not session_data['thinking_enabled']
            emit('thinking_toggled', {'enabled': session_data['thinking_enabled']})
        
    except Exception as e:
        emit('error', {'message': str(e)})


@socketio.on('get_history')
def handle_get_history(data):
    """Get conversation history"""
    try:
        session_id = data.get('session_id')
        if not session_id:
            emit('error', {'message': 'No session ID'})
            return
        
        session_data = gui_manager.get_session(session_id)
        if session_data:
            emit('conversation_history', {'messages': session_data['messages']})
        
    except Exception as e:
        emit('error', {'message': str(e)})


@socketio.on('clear_session')
def handle_clear_session(data):
    """Clear current session"""
    try:
        session_id = data.get('session_id')
        if not session_id:
            emit('error', {'message': 'No session ID'})
            return
        
        # Remove session to force recreation
        if session_id in gui_manager.sessions:
            del gui_manager.sessions[session_id]
        
        emit('session_cleared', {'message': 'Session cleared successfully'})
        
    except Exception as e:
        emit('error', {'message': str(e)})


def create_templates_dir():
    """Create templates directory and HTML file"""
    templates_dir = Path("templates")
    templates_dir.mkdir(exist_ok=True)
    
    # We'll create the HTML template next
    return templates_dir


def check_dependencies():
    """Check if required dependencies are available"""
    missing = []
    
    try:
        import flask
    except ImportError:
        missing.append("flask")
    
    try:
        import flask_socketio
    except ImportError:
        missing.append("flask-socketio")
    
    if missing:
        print(f"❌ Missing dependencies: {', '.join(missing)}")
        print("Install with: pip install flask flask-socketio")
        return False
    
    return True


def main():
    """Main function to run the web GUI"""
    
    print("🌐 BioinformaticsAgent Web GUI")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        return False
    
    # Check API configuration
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("❌ No ANTHROPIC_API_KEY found!")
        print("Please set your API key in .env.local")
        return False
    
    print(f"✅ Claude API configured: {'*' * 20}{api_key[-4:]}")
    print(f"✅ Model: {os.getenv('CLAUDE_MODEL', 'claude-sonnet-4-20250514')}")
    
    # Create templates directory
    create_templates_dir()
    
    # Create output directories
    Path("web_output").mkdir(exist_ok=True)
    Path("static").mkdir(exist_ok=True)
    
    print("✅ Web GUI setup complete!")
    print()
    print("🚀 Starting web server...")
    print("📱 Open your browser to: http://localhost:8080")
    print()
    print("Features available:")
    print("- Real-time chat with Claude Sonnet 4")
    print("- File upload for bioinformatics data") 
    print("- Output visualization and download")
    print("- Session management and history")
    print("- Thinking process display")
    print()
    
    # Run the Flask app
    try:
        socketio.run(app, host='0.0.0.0', port=8080, debug=False, allow_unsafe_werkzeug=True)
    except KeyboardInterrupt:
        print("\n👋 Web GUI stopped")
    except Exception as e:
        print(f"❌ Error running web server: {e}")
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)