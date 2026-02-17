"""
eBird API Client for Regional Species Lists

Fetches and caches regional bird species lists from eBird API.
Used to filter species identifications to only birds that occur in the user's region.
"""
import requests
import json
import csv
import os
from typing import List, Set
from pathlib import Path
from io import StringIO


class EBirdClient:
    """
    Client for eBird API v2.
    Fetches regional species lists and caches them locally.
    """

    BASE_URL = "https://api.ebird.org/v2"
    CACHE_DIR = "data/ebird_cache"

    def __init__(self, api_key: str, region_code: str):
        """
        Initialize eBird client.

        Args:
            api_key: eBird API key (get from https://ebird.org/api/keygen)
            region_code: eBird region code (e.g., 'US-NM' for New Mexico, 'US-CA' for California)
        """
        self.api_key = api_key
        self.region_code = region_code
        self.headers = {'x-ebirdapitoken': api_key}

        # Create cache directory
        Path(self.CACHE_DIR).mkdir(parents=True, exist_ok=True)

        # Load or fetch species data
        self.species_codes = self._get_regional_species()
        self.taxonomy = self._get_taxonomy()
        self.common_names = self._build_common_names_set()

        print(f"✓ eBird client initialized for region {region_code}")
        print(f"  Species in region: {len(self.common_names)}")

    def _get_cache_path(self, cache_name: str) -> str:
        """Get path to cache file."""
        return os.path.join(self.CACHE_DIR, f"{cache_name}_{self.region_code}.json")

    def _get_regional_species(self) -> List[str]:
        """
        Fetch list of species codes observed in the region.
        Uses cache if available.
        """
        cache_path = self._get_cache_path("species_codes")

        # Try to load from cache
        if os.path.exists(cache_path):
            print(f"📂 Loading regional species list from cache...")
            with open(cache_path, 'r') as f:
                return json.load(f)

        # Fetch from API
        print(f"🌐 Fetching regional species list for {self.region_code} from eBird...")
        url = f"{self.BASE_URL}/product/spplist/{self.region_code}"

        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            species_codes = response.json()

            # Cache for future use
            with open(cache_path, 'w') as f:
                json.dump(species_codes, f, indent=2)

            print(f"  ✓ Found {len(species_codes)} species in {self.region_code}")
            return species_codes

        except requests.exceptions.RequestException as e:
            print(f"  ⚠️  Failed to fetch species list: {e}")
            print(f"  Using empty species list (no filtering)")
            return []

    def _get_taxonomy(self) -> dict:
        """
        Fetch eBird taxonomy (maps species codes to common names).
        Uses cache if available.

        Note: eBird taxonomy endpoint returns CSV, not JSON.
        """
        cache_path = self._get_cache_path("taxonomy")

        # Try to load from cache
        if os.path.exists(cache_path):
            print(f"📂 Loading eBird taxonomy from cache...")
            with open(cache_path, 'r') as f:
                return json.load(f)

        # Fetch from API (returns CSV)
        print(f"🌐 Fetching eBird taxonomy (CSV format)...")
        url = f"{self.BASE_URL}/ref/taxonomy/ebird"

        try:
            response = requests.get(url, headers=self.headers, timeout=60)
            response.raise_for_status()

            # Parse CSV response
            csv_text = response.text
            csv_reader = csv.DictReader(StringIO(csv_text))

            # Convert to dict for faster lookup: speciesCode -> entry
            taxonomy_dict = {}
            for row in csv_reader:
                species_code = row.get('SPECIES_CODE')
                if species_code:
                    taxonomy_dict[species_code] = {
                        'speciesCode': species_code,
                        'comName': row.get('COMMON_NAME', ''),
                        'sciName': row.get('SCIENTIFIC_NAME', ''),
                        'category': row.get('CATEGORY', ''),
                        'order': row.get('ORDER', ''),
                        'familyComName': row.get('FAMILY_COM_NAME', ''),
                    }

            # Cache for future use
            with open(cache_path, 'w') as f:
                json.dump(taxonomy_dict, f, indent=2)

            print(f"  ✓ Loaded {len(taxonomy_dict)} species from eBird taxonomy")
            return taxonomy_dict

        except requests.exceptions.RequestException as e:
            print(f"  ⚠️  Failed to fetch taxonomy: {e}")
            print(f"  Using empty taxonomy (no filtering)")
            return {}
        except Exception as e:
            print(f"  ⚠️  Failed to parse taxonomy CSV: {e}")
            print(f"  Using empty taxonomy (no filtering)")
            return {}

    def _build_common_names_set(self) -> Set[str]:
        """
        Build a set of common names for species in the region.
        Used for fast lookup when filtering predictions.
        """
        common_names = set()

        for species_code in self.species_codes:
            if species_code in self.taxonomy:
                entry = self.taxonomy[species_code]
                common_name = entry.get('comName', '').upper()
                if common_name:
                    common_names.add(common_name)

        return common_names

    def is_species_in_region(self, common_name: str) -> bool:
        """
        Check if a species occurs in the configured region.

        Args:
            common_name: Common name of bird (e.g., "Northern Cardinal")

        Returns:
            True if species has been observed in region, False otherwise
        """
        # Normalize to uppercase for case-insensitive comparison
        normalized_name = common_name.upper().strip()
        return normalized_name in self.common_names



