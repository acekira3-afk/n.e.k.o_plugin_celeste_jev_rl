"""Pytest configuration for standalone testing (outside N.E.K.O).

Mocks the plugin.sdk.plugin module so core modules can be tested
without the full N.E.K.O runtime.
"""

import sys
import types
from unittest.mock import MagicMock

# Create a mock plugin.sdk.plugin package
_mock_plugin = types.ModuleType("plugin")
_mock_sdk = types.ModuleType("plugin.sdk")
_mock_plugin_sdk = types.ModuleType("plugin.sdk.plugin")

# Mock all commonly used imports
_mock_names = [
    "NekoPluginBase", "neko_plugin", "plugin_entry", "lifecycle",
    "timer_interval", "llm_tool", "message", "on_event", "custom_event",
    "hook", "before_entry", "after_entry", "around_entry", "replace_entry",
    "plugin", "Ok", "Err", "Result", "unwrap", "unwrap_or",
    "SdkError", "TransportError", "PluginMeta", "EntryKind",
    "LlmToolMeta", "OsActivitySnapshot", "get_os_activity_snapshot",
    "PluginI18n", "tr", "PluginSettings", "SettingsField",
    "Plugins", "PluginRouter", "PluginConfig", "PluginStore",
    "SystemInfo", "get_plugin_logger",
]

for name in _mock_names:
    setattr(_mock_plugin_sdk, name, MagicMock())

sys.modules["plugin"] = _mock_plugin
sys.modules["plugin.sdk"] = _mock_sdk
sys.modules["plugin.sdk.plugin"] = _mock_plugin_sdk
