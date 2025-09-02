"""
Professional Video Analysis Configuration
Centralized configuration management for Brandlytics video processing.
"""

from pathlib import Path
from typing import Dict, List, Tuple
import yaml


class VideoAnalysisConfig:
    """Configuration class for video analysis parameters."""
    
    # Default video processing settings
    DEFAULT_CONFIDENCE_THRESHOLD = 0.5
    DEFAULT_MAX_FRAMES = None  # Process all frames with smart sampling
    DEFAULT_SAMPLING_STRATEGIES = {
        'short_video': {'max_duration': 60, 'sampling_interval': 5},
        'medium_video': {'max_duration': 300, 'sampling_interval': 10},
        'long_video': {'max_duration': float('inf'), 'sampling_interval': 15}
    }
    
    # Database settings
    DEFAULT_BATCH_SIZE = 50
    DEFAULT_SAVE_TO_DATABASE = True
    
    # Supported video formats
    SUPPORTED_VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm')
    
    # Model settings
    DEFAULT_MODEL_PATH = "models/best2.pt"
    
    # Output settings
    DEFAULT_OUTPUT_FORMAT = 'json'
    SUPPORTED_OUTPUT_FORMATS = ['json', 'csv', 'yaml']
    
    def __init__(self, config_file: str = None):
        """
        Initialize configuration.
        
        Args:
            config_file: Optional path to YAML configuration file
        """
        self.config = self._load_default_config()
        
        if config_file and Path(config_file).exists():
            self._load_config_file(config_file)
    
    def _load_default_config(self) -> Dict:
        """Load default configuration."""
        return {
            'processing': {
                'confidence_threshold': self.DEFAULT_CONFIDENCE_THRESHOLD,
                'max_frames': self.DEFAULT_MAX_FRAMES,
                'sampling_strategies': self.DEFAULT_SAMPLING_STRATEGIES,
                'batch_size': self.DEFAULT_BATCH_SIZE
            },
            'database': {
                'save_to_database': self.DEFAULT_SAVE_TO_DATABASE,
                'batch_size': self.DEFAULT_BATCH_SIZE
            },
            'model': {
                'model_path': self.DEFAULT_MODEL_PATH
            },
            'video': {
                'supported_extensions': self.SUPPORTED_VIDEO_EXTENSIONS
            },
            'output': {
                'format': self.DEFAULT_OUTPUT_FORMAT,
                'include_sample_detections': True,
                'max_sample_detections': 10
            }
        }
    
    def _load_config_file(self, config_file: str) -> None:
        """Load configuration from YAML file."""
        try:
            with open(config_file, 'r') as f:
                file_config = yaml.safe_load(f)
            
            # Merge with default config
            self._deep_merge(self.config, file_config)
            
        except Exception as e:
            print(f"Warning: Failed to load config file {config_file}: {e}")
    
    def _deep_merge(self, base_dict: Dict, update_dict: Dict) -> None:
        """Deep merge two dictionaries."""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_merge(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def get_processing_config(self) -> Dict:
        """Get processing configuration."""
        return self.config['processing']
    
    def get_database_config(self) -> Dict:
        """Get database configuration."""
        return self.config['database']
    
    def get_model_config(self) -> Dict:
        """Get model configuration."""
        return self.config['model']
    
    def get_video_config(self) -> Dict:
        """Get video configuration."""
        return self.config['video']
    
    def get_output_config(self) -> Dict:
        """Get output configuration."""
        return self.config['output']
    
    def get_sampling_strategy(self, duration_seconds: float) -> Dict:
        """Get appropriate sampling strategy based on video duration."""
        strategies = self.config['processing']['sampling_strategies']
        
        for strategy_name, strategy_config in strategies.items():
            if duration_seconds <= strategy_config['max_duration']:
                return strategy_config
        
        # Default to long video strategy
        return strategies['long_video']
    
    def save_config(self, output_path: str) -> None:
        """Save current configuration to YAML file."""
        with open(output_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2)


# Default analysis presets
ANALYSIS_PRESETS = {
    'fast': {
        'processing': {
            'confidence_threshold': 0.7,
            'max_frames': 100,
            'sampling_strategies': {
                'short_video': {'max_duration': 60, 'sampling_interval': 10},
                'medium_video': {'max_duration': 300, 'sampling_interval': 20},
                'long_video': {'max_duration': float('inf'), 'sampling_interval': 30}
            }
        }
    },
    'balanced': {
        'processing': {
            'confidence_threshold': 0.5,
            'max_frames': None,
            'sampling_strategies': {
                'short_video': {'max_duration': 60, 'sampling_interval': 5},
                'medium_video': {'max_duration': 300, 'sampling_interval': 10},
                'long_video': {'max_duration': float('inf'), 'sampling_interval': 15}
            }
        }
    },
    'thorough': {
        'processing': {
            'confidence_threshold': 0.3,
            'max_frames': None,
            'sampling_strategies': {
                'short_video': {'max_duration': 60, 'sampling_interval': 2},
                'medium_video': {'max_duration': 300, 'sampling_interval': 5},
                'long_video': {'max_duration': float('inf'), 'sampling_interval': 8}
            }
        }
    }
}


def create_config_template(output_path: str = "video_analysis_config.yaml") -> None:
    """Create a configuration template file."""
    config = VideoAnalysisConfig()
    
    # Add comments to the template
    template_content = """# Brandlytics Video Analysis Configuration
# Professional configuration for video brand detection

# Processing settings
processing:
  # Minimum confidence threshold for detections (0.0 - 1.0)
  confidence_threshold: 0.5
  
  # Maximum number of frames to process (null for auto-sampling)
  max_frames: null
  
  # Sampling strategies based on video duration
  sampling_strategies:
    short_video:
      max_duration: 60      # seconds
      sampling_interval: 5  # process every Nth frame
    medium_video:
      max_duration: 300
      sampling_interval: 10
    long_video:
      max_duration: .inf
      sampling_interval: 15
  
  # Database batch size for saving detections
  batch_size: 50

# Database settings
database:
  # Whether to save detections to Supabase
  save_to_database: true
  
  # Batch size for database operations
  batch_size: 50

# Model settings
model:
  # Path to YOLO model file
  model_path: "models/best2.pt"

# Video settings
video:
  # Supported video file extensions
  supported_extensions:
    - ".mp4"
    - ".avi"
    - ".mov"
    - ".mkv"
    - ".wmv"
    - ".flv"
    - ".webm"

# Output settings
output:
  # Output format for results (json, csv, yaml)
  format: "json"
  
  # Include sample detections in output
  include_sample_detections: true
  
  # Maximum number of sample detections to include
  max_sample_detections: 10

# Analysis presets (uncomment to use)
# Use these presets by copying the settings above

# Fast analysis (lower accuracy, faster processing)
# fast_preset:
#   confidence_threshold: 0.7
#   max_frames: 100
#   sampling_intervals: [10, 20, 30]

# Thorough analysis (higher accuracy, slower processing)
# thorough_preset:
#   confidence_threshold: 0.3
#   sampling_intervals: [2, 5, 8]
"""
    
    with open(output_path, 'w') as f:
        f.write(template_content)
    
    print(f"Configuration template created: {output_path}")


def load_preset_config(preset_name: str) -> VideoAnalysisConfig:
    """
    Load a predefined configuration preset.
    
    Args:
        preset_name: Name of preset ('fast', 'balanced', 'thorough')
    
    Returns:
        VideoAnalysisConfig with preset applied
    """
    if preset_name not in ANALYSIS_PRESETS:
        available = ', '.join(ANALYSIS_PRESETS.keys())
        raise ValueError(f"Unknown preset '{preset_name}'. Available: {available}")
    
    config = VideoAnalysisConfig()
    preset = ANALYSIS_PRESETS[preset_name]
    config._deep_merge(config.config, preset)
    
    return config


# Convenience function
def get_default_config() -> VideoAnalysisConfig:
    """Get default configuration."""
    return VideoAnalysisConfig()
