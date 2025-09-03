#!/usr/bin/env python3
"""
Hugging Face Hub CLI tool for Brandlytics.
Manage models, upload, download, and configure HF integration.
"""

import argparse
import sys
import os
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from app.services.huggingface_service import get_hf_model_service
from app.config.model_config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class HuggingFaceCLI:
    """CLI interface for Hugging Face Hub operations."""
    
    def __init__(self):
        self.hf_service = get_hf_model_service()
    
    def list_models(self):
        """List all models in the organization."""
        try:
            models = self.hf_service.list_organization_models()
            
            if not models:
                print("No models found in the organization.")
                return
            
            print(f"\n🤗 Models in {settings.HF_ORG} organization:")
            print("=" * 60)
            
            for model in models:
                print(f"📦 {model['id']}")
                print(f"   Downloads: {model.get('downloads', 0)}")
                print(f"   Likes: {model.get('likes', 0)}")
                print(f"   Tags: {', '.join(model.get('tags', []))}")
                if model.get('last_modified'):
                    print(f"   Last modified: {model['last_modified']}")
                print()
                
        except Exception as e:
            print(f"❌ Error listing models: {e}")
    
    def model_info(self, repo_id: str):
        """Get detailed information about a model."""
        try:
            info = self.hf_service.get_model_info(repo_id)
            
            if not info:
                print(f"❌ Model {repo_id} not found")
                return
            
            print(f"\n📊 Model Information: {repo_id}")
            print("=" * 60)
            print(f"ID: {info.get('id', 'N/A')}")
            print(f"Downloads: {info.get('downloads', 0)}")
            print(f"Likes: {info.get('likes', 0)}")
            print(f"Pipeline Tag: {info.get('pipeline_tag', 'N/A')}")
            print(f"Tags: {', '.join(info.get('tags', []))}")
            print(f"Created: {info.get('created_at', 'N/A')}")
            print(f"Last Modified: {info.get('last_modified', 'N/A')}")
            
            siblings = info.get('siblings', [])
            if siblings:
                print(f"Files ({len(siblings)}):")
                for file in siblings[:10]:  # Show first 10 files
                    print(f"  - {file}")
                if len(siblings) > 10:
                    print(f"  ... and {len(siblings) - 10} more files")
            
        except Exception as e:
            print(f"❌ Error getting model info: {e}")
    
    def download_model(self, repo_id: str, filename: str = None):
        """Download a model from Hugging Face Hub."""
        try:
            print(f"📥 Downloading model {repo_id}...")
            if filename:
                print(f"   File: {filename}")
            
            model_path = self.hf_service.download_model(repo_id, filename)
            print(f"✅ Model downloaded to: {model_path}")
            
        except Exception as e:
            print(f"❌ Error downloading model: {e}")
    
    def upload_model(self, local_path: str, repo_id: str, commit_message: str = None):
        """Upload a local model to Hugging Face Hub."""
        try:
            if not os.path.exists(local_path):
                print(f"❌ Local path does not exist: {local_path}")
                return
            
            print(f"📤 Uploading {local_path} to {repo_id}...")
            
            success = self.hf_service.upload_model(local_path, repo_id, commit_message)
            
            if success:
                print(f"✅ Model uploaded successfully to {repo_id}")
            else:
                print(f"❌ Upload failed")
                
        except Exception as e:
            print(f"❌ Error uploading model: {e}")
    
    def test_connection(self):
        """Test connection to Hugging Face Hub."""
        try:
            print("🔧 Testing Hugging Face Hub connection...")
            
            if not settings.HF_TOKEN:
                print("⚠️  Warning: HF_TOKEN not set")
            else:
                print("✅ HF_TOKEN is configured")
            
            models = self.hf_service.list_organization_models()
            print(f"✅ Successfully connected to {settings.HF_ORG}")
            print(f"   Found {len(models)} models in organization")
            
        except Exception as e:
            print(f"❌ Connection test failed: {e}")
    
    def show_config(self):
        """Show current Hugging Face configuration."""
        print("\n⚙️  Hugging Face Configuration:")
        print("=" * 40)
        print(f"Organization: {settings.HF_ORG}")
        print(f"Default Model Repo: {settings.HF_MODEL_REPO}")
        print(f"Use HF Model: {settings.USE_HF_MODEL}")
        print(f"Cache Directory: {settings.HF_CACHE_DIR}")
        print(f"Token Configured: {'Yes' if settings.HF_TOKEN else 'No'}")
        
        if settings.USE_HF_MODEL:
            print(f"\n🎯 Current model source: Hugging Face Hub ({settings.HF_MODEL_REPO})")
        else:
            print(f"\n🎯 Current model source: Local file ({settings.MODEL_PATH})")
    
    def clear_cache(self):
        """Clear the model cache."""
        try:
            print("🧹 Clearing Hugging Face model cache...")
            success = self.hf_service.clear_cache()
            
            if success:
                print("✅ Cache cleared successfully")
            else:
                print("❌ Failed to clear cache")
                
        except Exception as e:
            print(f"❌ Error clearing cache: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Hugging Face Hub CLI for Brandlytics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python hf_manager.py list                                    # List all models
  python hf_manager.py info CV-Brandlytics/ModelM             # Get model info
  python hf_manager.py download CV-Brandlytics/ModelM         # Download model
  python hf_manager.py upload models/best.pt CV-Brandlytics/NewModel  # Upload model
  python hf_manager.py config                                  # Show configuration
  python hf_manager.py test                                    # Test connection
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # List models
    subparsers.add_parser("list", help="List all models in organization")
    
    # Model info
    info_parser = subparsers.add_parser("info", help="Get model information")
    info_parser.add_argument("repo_id", help="Model repository ID")
    
    # Download model
    download_parser = subparsers.add_parser("download", help="Download model")
    download_parser.add_argument("repo_id", help="Model repository ID")
    download_parser.add_argument("--file", help="Specific file to download")
    
    # Upload model
    upload_parser = subparsers.add_parser("upload", help="Upload model")
    upload_parser.add_argument("local_path", help="Local model path")
    upload_parser.add_argument("repo_id", help="Target repository ID")
    upload_parser.add_argument("--message", help="Commit message")
    
    # Configuration
    subparsers.add_parser("config", help="Show configuration")
    
    # Test connection
    subparsers.add_parser("test", help="Test Hugging Face connection")
    
    # Clear cache
    subparsers.add_parser("clear-cache", help="Clear model cache")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    cli = HuggingFaceCLI()
    
    try:
        if args.command == "list":
            cli.list_models()
        elif args.command == "info":
            cli.model_info(args.repo_id)
        elif args.command == "download":
            cli.download_model(args.repo_id, args.file)
        elif args.command == "upload":
            cli.upload_model(args.local_path, args.repo_id, args.message)
        elif args.command == "config":
            cli.show_config()
        elif args.command == "test":
            cli.test_connection()
        elif args.command == "clear-cache":
            cli.clear_cache()
        else:
            parser.print_help()
            
    except KeyboardInterrupt:
        print("\n\n⏹️  Operation cancelled by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
