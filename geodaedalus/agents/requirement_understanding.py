"""Agent 1: Requirement Understanding - Converts natural language to geoscientific constraints."""

import json
import re
from typing import Any, Dict, List, Optional

from geodaedalus.agents.base import BaseAgent
from geodaedalus.core.models import (
    GeoscientificConstraints,
    GeospatialLocation,
    TemporalConstraints,
    ElementConstraints,
    ElementCategory,
    RockType,
)
from geodaedalus.services.llm import LLMService


class RequirementUnderstandingAgent(BaseAgent[str, GeoscientificConstraints]):
    """Agent for understanding and extracting geoscientific requirements from natural language."""
    
    def __init__(self, **kwargs):
        """Initialize requirement understanding agent."""
        super().__init__("requirement_understanding", **kwargs)
        self.llm_service = LLMService(self.settings)
        
        # Domain knowledge for validation
        self.valid_rock_types = [rt.value for rt in RockType]
        self.valid_elements = self.settings.supported_elements
        
        # Geological knowledge patterns
        self.geological_periods = [
            "precambrian", "cambrian", "ordovician", "silurian", "devonian", 
            "carboniferous", "permian", "triassic", "jurassic", "cretaceous",
            "paleogene", "neogene", "quaternary", "archean", "proterozoic"
        ]
        
        self.location_patterns = [
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',  # Proper nouns
            r'\b(Mount|Mt\.?|Lake|River|Bay|Peninsula|Island|Mountains|Hills)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'
        ]
    
    async def process(self, user_query: str, **kwargs: Any) -> GeoscientificConstraints:
        """Process natural language query into structured constraints."""
        if not self.validate_input(user_query):
            raise ValueError("Invalid input: user_query must be a non-empty string")
        
        self.logger.info("Processing user query", query_length=len(user_query))
        
        # Extract constraints using LLM
        constraints_dict = await self._extract_constraints_with_llm(user_query)
        
        # Validate and structure the constraints
        constraints = await self._structure_constraints(constraints_dict, user_query)
        
        # Enhance with domain knowledge
        enhanced_constraints = await self._enhance_constraints(constraints, user_query)
        
        self.logger.info(
            "Constraints extracted successfully",
            rock_types_count=len(enhanced_constraints.rock_types),
            element_constraints_count=len(enhanced_constraints.element_constraints),
            has_spatial=enhanced_constraints.spatial is not None,
            has_temporal=enhanced_constraints.temporal is not None
        )
        
        return enhanced_constraints
    
    async def _extract_constraints_with_llm(self, user_query: str) -> Dict[str, Any]:
        """Extract geoscientific constraints using LLM."""
        prompt = f"""
        Extract geoscientific research constraints from this natural language query:
        "{user_query}"
        
        Extract the following information and return as JSON:
        
        {{
            "spatial": {{
                "location_name": "specific location mentioned (e.g., 'Deccan Traps', 'Hawaii')",
                "country": "country if mentioned",
                "region": "region if mentioned",
                "coordinates": {{"latitude": null, "longitude": null}}
            }},
            "temporal": {{
                "geological_period": "geological time period if mentioned",
                "age_min": null,
                "age_max": null,
                "stratigraphic_unit": "specific formation or unit if mentioned"
            }},
            "rock_types": ["list of rock types mentioned"],
            "element_constraints": [
                {{
                    "category": "major|minor|isotopes|whole_rock",
                    "elements": ["list of elements or oxides"],
                    "required_elements": ["must-have elements"],
                    "excluded_elements": ["elements to avoid"]
                }}
            ],
            "additional_keywords": ["other relevant geological/geochemical terms"]
        }}
        
        Guidelines:
        - Rock types: igneous, metamorphic, sedimentary, volcanic, plutonic, clastic, carbonate
        - Element categories: major (oxides like SiO2, Al2O3), minor (trace elements), isotopes (ratios)
        - Extract only what is explicitly mentioned or strongly implied
        - Use null for missing information
        """
        
        response = await self.llm_service.generate_response(
            prompt=prompt,
            max_tokens=1000,
            temperature=0.1,
            agent_name=self.agent_name
        )
        
        try:
            constraints_dict = json.loads(response)
            return constraints_dict
        except json.JSONDecodeError as e:
            self.logger.error("Failed to parse LLM response as JSON", error=str(e))
            # Fallback to pattern-based extraction
            return await self._fallback_pattern_extraction(user_query)
    
    async def _fallback_pattern_extraction(self, user_query: str) -> Dict[str, Any]:
        """Fallback pattern-based extraction if LLM fails."""
        self.logger.info("Using fallback pattern extraction")
        
        query_lower = user_query.lower()
        
        # Extract rock types
        rock_types = []
        for rock_type in self.valid_rock_types:
            if rock_type in query_lower:
                rock_types.append(rock_type)
        
        # Extract geological periods
        geological_period = None
        for period in self.geological_periods:
            if period in query_lower:
                geological_period = period
                break
        
        # Extract element information
        element_constraints = []
        
        # Major elements (oxides)
        major_elements = []
        for element in self.valid_elements["major"]:
            if element.lower() in query_lower:
                major_elements.append(element)
        
        if major_elements or "major element" in query_lower:
            element_constraints.append({
                "category": "major",
                "elements": major_elements or self.valid_elements["major"][:5],  # Default subset
                "required_elements": None,
                "excluded_elements": None
            })
        
        # Minor/trace elements
        minor_elements = []
        for element in self.valid_elements["minor"]:
            if element.lower() in query_lower:
                minor_elements.append(element)
        
        if minor_elements or any(term in query_lower for term in ["trace element", "minor element"]):
            element_constraints.append({
                "category": "minor",
                "elements": minor_elements or self.valid_elements["minor"][:5],
                "required_elements": None,
                "excluded_elements": None
            })
        
        # Extract location using patterns
        location_name = None
        for pattern in self.location_patterns:
            matches = re.findall(pattern, user_query, re.IGNORECASE)
            if matches:
                location_name = matches[0] if isinstance(matches[0], str) else " ".join(matches[0])
                break
        
        return {
            "spatial": {
                "location_name": location_name,
                "country": None,
                "region": None,
                "coordinates": {"latitude": None, "longitude": None}
            },
            "temporal": {
                "geological_period": geological_period,
                "age_min": None,
                "age_max": None,
                "stratigraphic_unit": None
            },
            "rock_types": rock_types,
            "element_constraints": element_constraints,
            "additional_keywords": self._extract_keywords(user_query)
        }
    
    def _extract_keywords(self, user_query: str) -> List[str]:
        """Extract additional geological keywords."""
        keywords = []
        geo_terms = [
            "geochemistry", "petrology", "mineralogy", "volcanism", "magma",
            "basalt", "granite", "andesite", "rhyolite", "gabbro", "diorite",
            "metamorphism", "sedimentary", "igneous", "volcanic", "plutonic",
            "mantle", "crust", "lithosphere", "asthenosphere", "subduction",
            "rifting", "hotspot", "island arc", "continental", "oceanic"
        ]
        
        query_lower = user_query.lower()
        for term in geo_terms:
            if term in query_lower:
                keywords.append(term)
        
        return keywords
    
    async def _structure_constraints(
        self, 
        constraints_dict: Dict[str, Any], 
        user_query: str
    ) -> GeoscientificConstraints:
        """Structure the constraints into Pydantic models."""
        
        # Spatial constraints
        spatial = None
        spatial_data = constraints_dict.get("spatial", {})
        if any(spatial_data.values()):
            coords = spatial_data.get("coordinates", {})
            spatial = GeospatialLocation(
                location_name=spatial_data.get("location_name"),
                country=spatial_data.get("country"),
                region=spatial_data.get("region"),
                latitude=coords.get("latitude") if coords else None,
                longitude=coords.get("longitude") if coords else None
            )
        
        # Temporal constraints
        temporal = None
        temporal_data = constraints_dict.get("temporal", {})
        if any(temporal_data.values()):
            temporal = TemporalConstraints(
                geological_period=temporal_data.get("geological_period"),
                age_min=temporal_data.get("age_min"),
                age_max=temporal_data.get("age_max"),
                stratigraphic_unit=temporal_data.get("stratigraphic_unit")
            )
        
        # Rock types
        rock_types = []
        for rock_type_str in constraints_dict.get("rock_types", []):
            try:
                rock_types.append(RockType(rock_type_str))
            except ValueError:
                self.logger.warning(f"Invalid rock type: {rock_type_str}")
        
        # Element constraints
        element_constraints = []
        for ec_data in constraints_dict.get("element_constraints", []):
            try:
                category = ElementCategory(ec_data.get("category", "major"))
                elements = ec_data.get("elements", [])
                
                if elements:  # Only add if elements are specified
                    element_constraints.append(ElementConstraints(
                        category=category,
                        elements=elements,
                        required_elements=ec_data.get("required_elements"),
                        excluded_elements=ec_data.get("excluded_elements")
                    ))
            except (ValueError, TypeError) as e:
                self.logger.warning(f"Invalid element constraint: {e}")
        
        return GeoscientificConstraints(
            spatial=spatial,
            temporal=temporal,
            rock_types=rock_types,
            element_constraints=element_constraints,
            additional_keywords=constraints_dict.get("additional_keywords", [])
        )
    
    async def _enhance_constraints(
        self, 
        constraints: GeoscientificConstraints, 
        user_query: str
    ) -> GeoscientificConstraints:
        """Enhance constraints with domain knowledge and geocoding."""
        
        # Enhance spatial information with geocoding
        if constraints.spatial and constraints.spatial.location_name:
            enhanced_spatial = await self._geocode_location(constraints.spatial)
            if enhanced_spatial:
                constraints.spatial = enhanced_spatial
        
        # Enhance element constraints with defaults if missing
        if not constraints.element_constraints and constraints.rock_types:
            constraints.element_constraints = await self._infer_element_constraints(constraints.rock_types)
        
        # Enhance temporal information
        if constraints.temporal and constraints.temporal.geological_period:
            enhanced_temporal = await self._enhance_temporal_constraints(constraints.temporal)
            if enhanced_temporal:
                constraints.temporal = enhanced_temporal
        
        return constraints
    
    async def _geocode_location(self, spatial: GeospatialLocation) -> Optional[GeospatialLocation]:
        """Enhance spatial information with geocoding (simplified version)."""
        # This would normally use a geocoding service
        # For now, use a simple lookup for common geological locations
        known_locations = {
            "deccan traps": {"latitude": 19.0, "longitude": 73.0, "country": "India"},
            "hawaii": {"latitude": 19.7, "longitude": -155.1, "country": "USA"},
            "iceland": {"latitude": 64.1, "longitude": -21.9, "country": "Iceland"},
            "yellowstone": {"latitude": 44.6, "longitude": -110.5, "country": "USA"},
            "andes": {"latitude": -15.0, "longitude": -70.0, "country": "South America"},
            "alps": {"latitude": 46.5, "longitude": 9.5, "country": "Europe"},
            "himalayas": {"latitude": 28.0, "longitude": 84.0, "country": "Asia"},
        }
        
        location_lower = spatial.location_name.lower()
        for known_name, coords in known_locations.items():
            if known_name in location_lower:
                return GeospatialLocation(
                    location_name=spatial.location_name,
                    country=coords["country"],
                    region=spatial.region,
                    latitude=coords["latitude"],
                    longitude=coords["longitude"]
                )
        
        return spatial
    
    async def _infer_element_constraints(self, rock_types: List[RockType]) -> List[ElementConstraints]:
        """Infer appropriate element constraints based on rock types."""
        element_constraints = []
        
        # Default to major elements for most rock types
        if any(rt in [RockType.IGNEOUS, RockType.VOLCANIC, RockType.PLUTONIC] for rt in rock_types):
            element_constraints.append(ElementConstraints(
                category=ElementCategory.MAJOR,
                elements=self.valid_elements["major"][:8],  # Common major elements
                required_elements=["SiO2", "Al2O3", "MgO", "CaO"],
                excluded_elements=None
            ))
            
            # Add trace elements for igneous rocks
            element_constraints.append(ElementConstraints(
                category=ElementCategory.MINOR,
                elements=self.valid_elements["minor"][:10],
                required_elements=None,
                excluded_elements=None
            ))
        
        elif any(rt in [RockType.SEDIMENTARY, RockType.CLASTIC, RockType.CARBONATE] for rt in rock_types):
            element_constraints.append(ElementConstraints(
                category=ElementCategory.MAJOR,
                elements=["SiO2", "Al2O3", "CaO", "MgO", "Fe2O3", "K2O", "Na2O"],
                required_elements=["SiO2", "CaO"],
                excluded_elements=None
            ))
        
        elif RockType.METAMORPHIC in rock_types:
            element_constraints.append(ElementConstraints(
                category=ElementCategory.MAJOR,
                elements=self.valid_elements["major"],
                required_elements=["SiO2", "Al2O3"],
                excluded_elements=None
            ))
        
        return element_constraints
    
    async def _enhance_temporal_constraints(self, temporal: TemporalConstraints) -> Optional[TemporalConstraints]:
        """Enhance temporal constraints with age ranges."""
        # Simplified geological time scale (Ma = million years ago)
        geological_ages = {
            "archean": {"min": 2500, "max": 4000},
            "proterozoic": {"min": 541, "max": 2500},
            "cambrian": {"min": 485, "max": 541},
            "ordovician": {"min": 444, "max": 485},
            "silurian": {"min": 419, "max": 444},
            "devonian": {"min": 359, "max": 419},
            "carboniferous": {"min": 299, "max": 359},
            "permian": {"min": 252, "max": 299},
            "triassic": {"min": 201, "max": 252},
            "jurassic": {"min": 145, "max": 201},
            "cretaceous": {"min": 66, "max": 145},
            "paleogene": {"min": 23, "max": 66},
            "neogene": {"min": 2.6, "max": 23},
            "quaternary": {"min": 0, "max": 2.6},
        }
        
        if temporal.geological_period:
            period_lower = temporal.geological_period.lower()
            if period_lower in geological_ages:
                ages = geological_ages[period_lower]
                return TemporalConstraints(
                    geological_period=temporal.geological_period,
                    age_min=ages["min"],
                    age_max=ages["max"],
                    stratigraphic_unit=temporal.stratigraphic_unit
                )
        
        return temporal
    
    def validate_input(self, input_data: str) -> bool:
        """Validate input query."""
        return isinstance(input_data, str) and len(input_data.strip()) > 0 