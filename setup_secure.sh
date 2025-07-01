#!/bin/bash
# Secure setup script for BioinformaticsAgent

echo "🔒 Setting up secure BioinformaticsAgent configuration..."

# 1. Create .env from template if it doesn't exist
if [ ! -f .env ]; then
    if [ -f .env.example ]; then
        cp .env.example .env
        echo "✅ Created .env file from template"
        echo "⚠️  Please edit .env and add your ANTHROPIC_API_KEY"
    else
        echo "❌ .env.example not found. Creating basic .env..."
        cat > .env << 'EOF'
# Anthropic Claude API Key
# Get your API key from: https://console.anthropic.com/
ANTHROPIC_API_KEY=your-api-key-here

# Optional: Custom model configuration
CLAUDE_MODEL=claude-3-5-sonnet-20241022

# Optional: Execution settings
BIOAGENT_WORK_DIR=./workspace
BIOAGENT_OUTPUT_DIR=output
BIOAGENT_MAX_MEMORY_MB=2048
BIOAGENT_TIMEOUT_SECONDS=300
EOF
        echo "✅ Created basic .env file"
        echo "⚠️  Please edit .env and add your ANTHROPIC_API_KEY"
    fi
else
    echo "✅ .env file already exists"
fi

# 2. Set secure permissions on .env
chmod 600 .env
echo "✅ Set secure permissions on .env (600)"

# 3. Check and update .gitignore
if [ -f .gitignore ]; then
    if grep -q "^\.env$" .gitignore; then
        echo "✅ .env is already in .gitignore"
    else
        echo ".env" >> .gitignore
        echo "✅ Added .env to .gitignore"
    fi
else
    echo ".env" > .gitignore
    echo "✅ Created .gitignore with .env"
fi

# 4. Install pre-commit hook to prevent accidental API key commits
if [ -d .git ]; then
    if [ ! -f .git/hooks/pre-commit ]; then
        cat > .git/hooks/pre-commit << 'EOF'
#!/bin/bash
# Pre-commit hook to prevent API key commits
if git diff --cached --name-only | xargs grep -l "sk-ant-api03-\|ANTHROPIC_API_KEY.*sk-" 2>/dev/null; then
    echo "🚨 ERROR: Potential API key found in staged files!"
    echo "Please remove API keys before committing."
    echo "Check files for patterns like 'sk-ant-api03-' or hardcoded keys."
    exit 1
fi
EOF
        chmod +x .git/hooks/pre-commit
        echo "✅ Installed pre-commit hook to prevent API key leaks"
    else
        echo "✅ Pre-commit hook already exists"
    fi
else
    echo "⚠️  Not a git repository - skipping pre-commit hook setup"
fi

# 5. Create workspace directory
mkdir -p workspace
echo "✅ Created workspace directory"

# 6. Verify .env is not tracked by git
if [ -d .git ]; then
    if git ls-files | grep -q "^\.env$"; then
        echo "🚨 WARNING: .env is tracked by git!"
        echo "Run: git rm --cached .env"
        echo "Then commit the removal"
    else
        echo "✅ .env is not tracked by git"
    fi
fi

echo ""
echo "🎯 Setup complete! Next steps:"
echo "1. Edit .env and add your Anthropic API key:"
echo "   nano .env"
echo ""
echo "2. Your API key should look like:"
echo "   ANTHROPIC_API_KEY=sk-ant-api03-xxxxxxxxxxxx"
echo ""
echo "3. Test the setup:"
echo "   python test_claude_integration.py"
echo ""
echo "4. Start the interactive agent:"
echo "   python interactive_agent.py"
echo ""
echo "🔒 Security reminders:"
echo "- NEVER commit .env files to git"
echo "- NEVER share API keys in messages or code"
echo "- Rotate your API keys regularly"
echo "- Check the SECURITY.md file for more details"