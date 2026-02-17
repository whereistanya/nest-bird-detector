"""
Tests for filename sanitization security
Ensures species names cannot cause path traversal vulnerabilities
"""
import pytest
from notifications import sanitize_filename


class TestFilenameSanitization:
    """Test suite for sanitize_filename security function"""

    def test_normal_species_name(self):
        """Test normal species names work correctly"""
        assert sanitize_filename("American Robin") == "american_robin"
        assert sanitize_filename("Blue Jay") == "blue_jay"
        assert sanitize_filename("Northern Cardinal") == "northern_cardinal"

    def test_path_traversal_attacks(self):
        """Test that path traversal sequences are blocked"""
        # Basic path traversal
        assert ".." not in sanitize_filename("../../../etc/passwd")
        assert "/" not in sanitize_filename("../../secret")
        assert "\\" not in sanitize_filename("..\\..\\windows\\system32")

        # More complex attempts
        assert sanitize_filename("../../../etc/passwd") == "etc_passwd"
        assert sanitize_filename("../../secret") == "secret"
        assert sanitize_filename("..\\..\\windows") == "windows"

    def test_null_bytes(self):
        """Test that null bytes are removed"""
        result = sanitize_filename("evil\x00.jpg")
        assert "\x00" not in result
        assert result == "evil.jpg"

    def test_special_characters(self):
        """Test that filesystem special characters are handled"""
        # Forward slash
        assert "/" not in sanitize_filename("species/name")

        # Backslash
        assert "\\" not in sanitize_filename("species\\name")

        # Colons (Windows drive letters)
        assert ":" not in sanitize_filename("C:\\windows")

        # Asterisks, question marks
        assert "*" not in sanitize_filename("wild*card")
        assert "?" not in sanitize_filename("question?")

        # Pipe characters
        assert "|" not in sanitize_filename("pipe|char")

        # Quotes
        assert '"' not in sanitize_filename('quote"name')
        assert "'" not in sanitize_filename("quote'name")

    def test_reserved_windows_filenames(self):
        """Test that reserved Windows filenames are prefixed"""
        reserved = ["CON", "PRN", "AUX", "NUL", "COM1", "COM9", "LPT1", "LPT9"]

        for name in reserved:
            result = sanitize_filename(name)
            # Should be prefixed with "bird_" to avoid conflict
            assert result.startswith("bird_") or result != name.lower()

    def test_length_limiting(self):
        """Test that very long names are truncated"""
        long_name = "a" * 200
        result = sanitize_filename(long_name)
        assert len(result) <= 50  # Default max_length

        # Test with custom max_length
        result = sanitize_filename(long_name, max_length=20)
        assert len(result) <= 20

    def test_leading_trailing_dots(self):
        """Test that leading/trailing dots are removed"""
        assert not sanitize_filename("...hidden").startswith(".")
        assert not sanitize_filename("file...").endswith(".")
        assert not sanitize_filename(".....").startswith(".")

    def test_multiple_underscores_collapsed(self):
        """Test that multiple underscores are collapsed"""
        assert "___" not in sanitize_filename("bird___with___spaces")
        assert sanitize_filename("a   b   c") == "a_b_c"

    def test_empty_or_invalid_input(self):
        """Test handling of empty or invalid input"""
        assert sanitize_filename("") == "unknown"
        assert sanitize_filename(None) == "unknown"
        assert sanitize_filename(".") == "unknown"
        assert sanitize_filename("..") == "unknown"
        assert sanitize_filename("...") == "unknown"

    def test_real_world_species_names(self):
        """Test with real species names that might have special chars"""
        # Species with apostrophes (apostrophes become underscores for safety)
        assert sanitize_filename("Anna's Hummingbird") == "anna_s_hummingbird"

        # Species with dashes (dashes are preserved as they're safe)
        assert sanitize_filename("Black-capped Chickadee") == "black-capped_chickadee"

        # Species with periods (periods are preserved but not at start/end)
        assert sanitize_filename("St. Lawrence Sparrow") == "st._lawrence_sparrow"

        # Species with parentheses (parentheses become underscores)
        assert sanitize_filename("Robin (American)") == "robin_american"

    def test_unicode_characters(self):
        """Test handling of unicode characters"""
        # Non-ASCII characters should be replaced
        result = sanitize_filename("Pájaro Azul")
        # Should only contain ASCII alphanumeric and underscores
        assert all(c.isalnum() or c in ['_', '-', '.'] for c in result)

    def test_xss_attempts_in_filenames(self):
        """Test that HTML/script injection attempts are neutralized"""
        result = sanitize_filename("<script>alert('xss')</script>")
        assert "<" not in result
        assert ">" not in result
        assert "script" in result  # Text remains but tags removed

    def test_absolute_paths(self):
        """Test that absolute paths cannot be created"""
        # Unix absolute path
        result = sanitize_filename("/etc/passwd")
        assert not result.startswith("/")

        # Windows absolute path
        result = sanitize_filename("C:\\Windows\\System32")
        assert ":" not in result
        assert "\\" not in result

    def test_case_sensitivity(self):
        """Test that names are normalized to lowercase"""
        assert sanitize_filename("ROBIN") == "robin"
        assert sanitize_filename("RoBiN") == "robin"
        assert sanitize_filename("American ROBIN") == "american_robin"
