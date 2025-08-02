# SentinelGem MCP Server Usage Guide

## Quick Commands

The MCP server (`mcp_server.py`) provides automated deployment tools for SentinelGem.

### Available Commands

```bash
# Activate virtual environment first
source venv/bin/activate

# Check Kaggle authentication
python mcp_server.py check-auth

# Validate project structure
python mcp_server.py validate

# Prepare files for Kaggle deployment
python mcp_server.py prepare

# Upload to Kaggle (if no conflicts)
python mcp_server.py upload

# Commit changes to GitHub
python mcp_server.py commit --message "Your commit message"

# Full automated deployment
python mcp_server.py deploy-all --message "Production release"
```

### MCP Server Functions

1. **Project Validation**: Ensures all required files and directories exist
2. **Kaggle Preparation**: Creates deployment-ready package in `kaggle_deploy/`
3. **GitHub Integration**: Automated commit and push with proper .gitignore handling
4. **Deployment Automation**: One-command deployment to both platforms

### Manual Kaggle Upload

If automated upload fails, use the prepared files in `kaggle_deploy/`:

1. Copy all files from `kaggle_deploy/` directory
2. Upload to https://www.kaggle.com/code/muzansano/sentinelgem/edit
3. Ensure proper metadata configuration

## Project is Production Ready! 🚀

All components are validated and ready for the Google Gemma 3n Impact Challenge 2025.
