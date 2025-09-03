#!/usr/bin/env python3
"""
Simple API test script to test the updated single-model configuration
"""

import requests
import json

def test_api():
    try:
        print("Testing API endpoints...")
        
        # Test basic endpoint
        response = requests.get("http://localhost:8000/")
        print(f"Root endpoint: {response.status_code}")
        print(f"Response: {response.json()}")
        
        # Test brands endpoint
        response = requests.get("http://localhost:8000/brands")
        print(f"\nBrands endpoint: {response.status_code}")
        brands_data = response.json()
        print(f"Response: {json.dumps(brands_data, indent=2)}")
        
        # Verify the expected brands
        expected_brands = ["coca cola", "nike", "starbucks"]
        actual_brands = brands_data.get("brands", [])
        
        print(f"\nExpected brands: {expected_brands}")
        print(f"Actual brands: {actual_brands}")
        print(f"Brands match: {set(expected_brands) == set(actual_brands)}")
        print(f"Model count: {brands_data.get('models_count', 0)}")
        print(f"Model source: {brands_data.get('model_source', 'Unknown')}")
        
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API. Make sure the server is running on http://localhost:8000")
    except Exception as e:
        print(f"❌ Error testing API: {e}")

if __name__ == "__main__":
    test_api()
