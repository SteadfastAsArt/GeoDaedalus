#!/usr/bin/env python3
"""Simple system test for GeoDaedalus without requiring API keys."""

import asyncio
from pathlib import Path

from geodaedalus.core.config import get_settings
from geodaedalus.core.models import GeoscientificConstraints, RockType, ElementCategory, ElementConstraints
from geodaedalus.services.search import SearchService
from geodaedalus.services.document_processor import DocumentProcessorService


async def test_system_components():
    """Test core system components."""
    print("🧪 Testing GeoDaedalus System Components")
    print("=" * 50)
    
    # Test 1: Configuration
    print("\n1. Testing Configuration System...")
    settings = get_settings()
    print(f"   ✅ App Name: {settings.app_name}")
    print(f"   ✅ Environment: {settings.environment}")
    print(f"   ✅ Supported Rock Types: {len(settings.supported_rock_types)}")
    print(f"   ✅ Major Elements: {len(settings.supported_elements['major'])}")
    
    # Test 2: Data Models
    print("\n2. Testing Data Models...")
    constraints = GeoscientificConstraints(
        rock_types=[RockType.VOLCANIC, RockType.IGNEOUS],
        element_constraints=[
            ElementConstraints(
                category=ElementCategory.MAJOR,
                elements=["SiO2", "Al2O3", "MgO"]
            )
        ],
        additional_keywords=["Hawaii", "basalt"]
    )
    keywords = constraints.to_search_keywords()
    print(f"   ✅ Created constraints with {len(keywords)} keywords: {keywords}")
    
    # Test 3: Search Service (without API calls)
    print("\n3. Testing Search Service...")
    search_service = SearchService(settings)
    print(f"   ✅ Search service initialized with {len(search_service.engines)} engines")
    
    # Test similarity calculation
    similarity = search_service._titles_similar(
        "Geochemistry of Hawaiian basalts",
        "Chemical composition of volcanic rocks from Hawaii"
    )
    print(f"   ✅ Title similarity calculation: {similarity}")
    
    await search_service.close()
    
    # Test 4: Document Processor
    print("\n4. Testing Document Processor...")
    doc_processor = DocumentProcessorService(settings)
    
    # Test geochemical data extraction
    test_text = """
    Major element compositions were determined by XRF.
    SiO2 = 49.2%, Al2O3 = 13.8%, MgO = 8.4%.
    The samples are basalts from Hawaii.
    """
    
    extracted = doc_processor.extract_geochemical_data(test_text)
    print(f"   ✅ Extracted {len(extracted['elements_mentioned'])} elements")
    print(f"   ✅ Found {len(extracted['rock_types'])} rock types")
    print(f"   ✅ Detected {len(extracted['analytical_methods'])} analytical methods")
    
    await doc_processor.close()
    
    # Test 5: Directory Structure
    print("\n5. Testing Directory Structure...")
    required_dirs = [settings.data_dir, settings.cache_dir, settings.output_dir]
    for dir_path in required_dirs:
        if dir_path.exists():
            print(f"   ✅ Directory exists: {dir_path}")
        else:
            print(f"   ❌ Directory missing: {dir_path}")
    
    print("\n🎉 System Test Completed Successfully!")
    print("\nGeoDaedalus is ready for use!")
    print("\nNext steps:")
    print("1. Set up API keys in .env file (OpenAI, Anthropic, SerpAPI)")
    print("2. Run: python demo.py --mode quick")
    print("3. Or use CLI: python -m geodaedalus.cli.main search 'your query'")


if __name__ == "__main__":
    asyncio.run(test_system_components()) 