# OpenLitterPI - Automated cat litterbox
# Copyright (C) 2025 Mark Nelson
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""
Basic tests for OpenLitterPI.
Run with: python -m pytest tests/
"""

import unittest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestImports(unittest.TestCase):
    """Test that modules can be imported."""

    def test_import_notifications(self):
        """Test notifications module imports."""
        import notifications
        self.assertTrue(hasattr(notifications, 'send_message'))

    def test_import_motor(self):
        """Test motor module imports (may fail without hardware)."""
        try:
            import motor
            self.assertTrue(hasattr(motor, 'cycle'))
            self.assertTrue(hasattr(motor, 'move'))
        except ImportError as e:
            self.skipTest(f"Motor module requires Raspberry PI hardware: {e}")

    def test_import_utils(self):
        """Test utils module imports."""
        try:
            import utils
            self.assertTrue(hasattr(utils, 'visualize'))
        except ImportError as e:
            self.skipTest(f"Utils module has unmet dependencies: {e}")


class TestNotifications(unittest.TestCase):
    """Test notification functionality."""

    def test_missing_credentials_no_crash(self):
        """Verify send_message handles missing credentials gracefully."""
        import notifications
        # Should not raise, just print warning
        notifications.send_message("Test message", photo=None, include_image=False)


class TestConfig(unittest.TestCase):
    """Test configuration."""

    def test_env_variables_defined(self):
        """Check that environment variable names are correct."""
        import notifications
        self.assertEqual(notifications.EMAIL_SUBJECT, "OpenLitterPI")


if __name__ == '__main__':
    unittest.main()
