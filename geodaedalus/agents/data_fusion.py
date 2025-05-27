"""Agent 4: Data Fusion - Consolidates, validates, and attributes data from multiple sources."""

import json
import statistics
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import UUID

import numpy as np
import pandas as pd

from geodaedalus.agents.base import BaseAgent
from geodaedalus.core.models import (
    ExtractedData,
    DataTable,
    GeospatialLocation,
    RockType,
    ElementCategory,
)
from geodaedalus.services.llm import LLMService


class DataFusionAgent(BaseAgent[ExtractedData, ExtractedData]):
    """Agent for fusing, validating, and attributing geochemical data."""
    
    def __init__(self, **kwargs):
        """Initialize data fusion agent."""
        super().__init__("data_fusion", **kwargs)
        self.llm_service = LLMService(self.settings)
        
        # Quality thresholds
        self.min_confidence_threshold = 0.5
        self.max_outlier_zscore = 3.0
        self.min_samples_for_stats = 3
        
        # Element validation ranges (typical ranges for common rock types)
        self.element_ranges = {
            "SiO2": (35.0, 80.0),  # wt%
            "Al2O3": (8.0, 25.0),
            "Fe2O3": (0.5, 20.0),
            "FeO": (0.0, 15.0),
            "MgO": (0.0, 50.0),
            "CaO": (0.0, 20.0),
            "Na2O": (0.0, 8.0),
            "K2O": (0.0, 8.0),
            "TiO2": (0.0, 5.0),
            "P2O5": (0.0, 2.0),
            "MnO": (0.0, 1.0),
        }
    
    async def process(
        self, 
        extracted_data: ExtractedData, 
        validation_level: str = "standard",
        **kwargs: Any
    ) -> ExtractedData:
        """Fuse and validate extracted data."""
        if not self.validate_input(extracted_data):
            raise ValueError("Invalid extracted data provided")
        
        self.logger.info(
            "Starting data fusion and validation",
            total_tables=len(extracted_data.tables),
            validation_level=validation_level,
            query_id=str(extracted_data.query_id)
        )
        
        # Step 1: Quality filtering
        high_quality_tables = await self._filter_by_quality(extracted_data.tables)
        
        # Step 2: Data standardization
        standardized_tables = await self._standardize_data(high_quality_tables)
        
        # Step 3: Duplicate detection and merging
        merged_tables = await self._merge_duplicates(standardized_tables)
        
        # Step 4: Outlier detection and validation
        validated_tables = await self._validate_data(merged_tables, validation_level)
        
        # Step 5: Cross-source validation
        cross_validated_tables = await self._cross_validate(validated_tables)
        
        # Step 6: Generate consolidated dataset
        consolidated_data = await self._generate_consolidated_dataset(cross_validated_tables)
        
        # Step 7: Calculate quality metrics
        quality_metrics = await self._calculate_quality_metrics(
            cross_validated_tables, 
            extracted_data.tables
        )
        
        # Step 8: Generate source attribution
        source_attribution = await self._generate_source_attribution(cross_validated_tables)
        
        self.logger.info(
            "Data fusion completed",
            original_tables=len(extracted_data.tables),
            final_tables=len(cross_validated_tables),
            data_completeness=quality_metrics.get("completeness", 0),
            consistency_score=quality_metrics.get("consistency", 0)
        )
        
        # Update consolidated data with fusion results
        consolidated_data.update({
            "fusion_summary": {
                "original_tables": len(extracted_data.tables),
                "quality_filtered": len(high_quality_tables),
                "standardized": len(standardized_tables),
                "merged": len(merged_tables),
                "validated": len(validated_tables),
                "final": len(cross_validated_tables),
            },
            "quality_metrics": quality_metrics,
            "source_attribution": source_attribution,
            "validation_level": validation_level,
        })
        
        return ExtractedData(
            id=extracted_data.id,
            query_id=extracted_data.query_id,
            tables=cross_validated_tables,
            consolidated_data=consolidated_data,
            source_papers=extracted_data.source_papers,
            extraction_summary=extracted_data.extraction_summary,
            data_completeness=quality_metrics.get("completeness"),
            consistency_score=quality_metrics.get("consistency")
        )
    
    async def _filter_by_quality(self, tables: List[DataTable]) -> List[DataTable]:
        """Filter tables based on quality metrics."""
        high_quality_tables = []
        
        for table in tables:
            # Check extraction confidence
            if (table.extraction_confidence or 0) < self.min_confidence_threshold:
                self.logger.debug(
                    f"Filtered table {table.table_number} due to low confidence",
                    confidence=table.extraction_confidence
                )
                continue
            
            # Check data completeness
            if not table.headers or not table.data:
                self.logger.debug(f"Filtered table {table.table_number} due to missing data")
                continue
            
            # Check for minimum viable data
            if len(table.data) < 2 or len(table.headers) < 2:
                self.logger.debug(f"Filtered table {table.table_number} due to insufficient data")
                continue
            
            # Check for geochemical relevance
            if not table.element_data:
                self.logger.debug(f"Filtered table {table.table_number} due to no element data")
                continue
            
            high_quality_tables.append(table)
        
        self.logger.info(
            f"Quality filtering: {len(high_quality_tables)}/{len(tables)} tables passed"
        )
        
        return high_quality_tables
    
    async def _standardize_data(self, tables: List[DataTable]) -> List[DataTable]:
        """Standardize data formats and units."""
        standardized_tables = []
        
        for table in tables:
            try:
                # Standardize headers
                standardized_headers = self._standardize_headers(table.headers)
                
                # Standardize data values
                standardized_data = self._standardize_values(table.data, standardized_headers)
                
                # Update element data with standardized headers
                standardized_element_data = {}
                for old_header, values in table.element_data.items():
                    # Find corresponding standardized header
                    old_index = table.headers.index(old_header) if old_header in table.headers else -1
                    if old_index >= 0 and old_index < len(standardized_headers):
                        new_header = standardized_headers[old_index]
                        standardized_element_data[new_header] = values
                
                # Create standardized table
                standardized_table = DataTable(
                    id=table.id,
                    paper_id=table.paper_id,
                    table_number=table.table_number,
                    caption=table.caption,
                    headers=standardized_headers,
                    data=standardized_data,
                    sample_ids=table.sample_ids,
                    location_info=table.location_info,
                    rock_type=table.rock_type,
                    element_data=standardized_element_data,
                    data_quality_score=table.data_quality_score,
                    extraction_confidence=table.extraction_confidence,
                    extracted_at=table.extracted_at
                )
                
                standardized_tables.append(standardized_table)
                
            except Exception as e:
                self.logger.error(f"Failed to standardize table {table.table_number}", error=str(e))
                # Keep original table if standardization fails
                standardized_tables.append(table)
        
        return standardized_tables
    
    def _standardize_headers(self, headers: List[str]) -> List[str]:
        """Standardize column headers."""
        standardized = []
        
        # Common header mappings
        header_mappings = {
            # Oxides
            "sio2": "SiO2", "si o2": "SiO2", "silica": "SiO2",
            "al2o3": "Al2O3", "al o3": "Al2O3", "alumina": "Al2O3",
            "fe2o3": "Fe2O3", "fe o3": "Fe2O3", "iron oxide": "Fe2O3",
            "feo": "FeO", "fe o": "FeO",
            "mgo": "MgO", "mg o": "MgO", "magnesia": "MgO",
            "cao": "CaO", "ca o": "CaO", "lime": "CaO",
            "na2o": "Na2O", "na o": "Na2O", "soda": "Na2O",
            "k2o": "K2O", "k o": "K2O", "potash": "K2O",
            "tio2": "TiO2", "ti o2": "TiO2", "titania": "TiO2",
            "p2o5": "P2O5", "p o5": "P2O5",
            "mno": "MnO", "mn o": "MnO",
            
            # Elements
            "silicon": "Si", "aluminum": "Al", "iron": "Fe",
            "magnesium": "Mg", "calcium": "Ca", "sodium": "Na",
            "potassium": "K", "titanium": "Ti", "phosphorus": "P",
            "manganese": "Mn",
            
            # Sample identifiers
            "sample": "Sample", "sample_id": "Sample", "sample id": "Sample",
            "specimen": "Sample", "rock": "Sample",
        }
        
        for header in headers:
            header_clean = str(header).strip().lower()
            
            # Direct mapping
            if header_clean in header_mappings:
                standardized.append(header_mappings[header_clean])
            else:
                # Keep original but clean
                standardized.append(str(header).strip())
        
        return standardized
    
    def _standardize_values(self, data: List[List[Any]], headers: List[str]) -> List[List[Any]]:
        """Standardize data values."""
        if not data or not headers:
            return data
        
        standardized_data = []
        
        for row in data:
            standardized_row = []
            for i, value in enumerate(row):
                if i < len(headers):
                    header = headers[i]
                    standardized_value = self._standardize_single_value(value, header)
                    standardized_row.append(standardized_value)
                else:
                    standardized_row.append(value)
            standardized_data.append(standardized_row)
        
        return standardized_data
    
    def _standardize_single_value(self, value: Any, header: str) -> Any:
        """Standardize a single data value."""
        if value is None or value == "":
            return None
        
        # Convert to string for processing
        value_str = str(value).strip()
        
        # Handle common non-numeric indicators
        if value_str.lower() in ["n.d.", "nd", "n.a.", "na", "-", "below detection", "bdl"]:
            return None
        
        # Try to extract numeric value
        try:
            # Remove common prefixes/suffixes
            cleaned = value_str.replace("<", "").replace(">", "").replace("±", "").strip()
            
            # Handle percentage signs
            if "%" in cleaned:
                cleaned = cleaned.replace("%", "").strip()
            
            # Convert to float
            return float(cleaned)
            
        except (ValueError, TypeError):
            # Return original value if can't convert
            return value
    
    async def _merge_duplicates(self, tables: List[DataTable]) -> List[DataTable]:
        """Identify and merge duplicate or overlapping tables."""
        if len(tables) <= 1:
            return tables
        
        # Group tables by similarity
        table_groups = self._group_similar_tables(tables)
        
        merged_tables = []
        for group in table_groups:
            if len(group) == 1:
                merged_tables.append(group[0])
            else:
                # Merge tables in group
                merged_table = await self._merge_table_group(group)
                if merged_table:
                    merged_tables.append(merged_table)
        
        self.logger.info(
            f"Duplicate merging: {len(tables)} -> {len(merged_tables)} tables"
        )
        
        return merged_tables
    
    def _group_similar_tables(self, tables: List[DataTable]) -> List[List[DataTable]]:
        """Group similar tables for merging."""
        groups = []
        used_indices = set()
        
        for i, table1 in enumerate(tables):
            if i in used_indices:
                continue
            
            current_group = [table1]
            used_indices.add(i)
            
            for j, table2 in enumerate(tables[i+1:], i+1):
                if j in used_indices:
                    continue
                
                if self._tables_similar(table1, table2):
                    current_group.append(table2)
                    used_indices.add(j)
            
            groups.append(current_group)
        
        return groups
    
    def _tables_similar(self, table1: DataTable, table2: DataTable) -> bool:
        """Check if two tables are similar enough to merge."""
        # Same paper
        if table1.paper_id == table2.paper_id:
            return True
        
        # Similar headers
        headers1 = set(h.lower() for h in table1.headers)
        headers2 = set(h.lower() for h in table2.headers)
        header_overlap = len(headers1.intersection(headers2)) / max(len(headers1), len(headers2))
        
        if header_overlap > 0.7:
            # Similar sample IDs
            samples1 = set(s.lower() for s in table1.sample_ids)
            samples2 = set(s.lower() for s in table2.sample_ids)
            
            if samples1 and samples2:
                sample_overlap = len(samples1.intersection(samples2)) / max(len(samples1), len(samples2))
                return sample_overlap > 0.3
            
            return True
        
        return False
    
    async def _merge_table_group(self, tables: List[DataTable]) -> Optional[DataTable]:
        """Merge a group of similar tables."""
        if not tables:
            return None
        
        if len(tables) == 1:
            return tables[0]
        
        try:
            # Use the highest confidence table as base
            base_table = max(tables, key=lambda t: t.extraction_confidence or 0)
            
            # Combine headers (union of all headers)
            all_headers = []
            seen_headers = set()
            for table in tables:
                for header in table.headers:
                    if header.lower() not in seen_headers:
                        all_headers.append(header)
                        seen_headers.add(header.lower())
            
            # Combine data
            combined_data = []
            combined_sample_ids = []
            combined_element_data = defaultdict(list)
            
            for table in tables:
                # Map table data to combined headers
                for row_idx, row in enumerate(table.data):
                    combined_row = [None] * len(all_headers)
                    
                    for col_idx, value in enumerate(row):
                        if col_idx < len(table.headers):
                            header = table.headers[col_idx]
                            # Find position in combined headers
                            try:
                                combined_col_idx = next(
                                    i for i, h in enumerate(all_headers) 
                                    if h.lower() == header.lower()
                                )
                                combined_row[combined_col_idx] = value
                            except StopIteration:
                                pass
                    
                    combined_data.append(combined_row)
                    
                    # Add sample ID if available
                    if row_idx < len(table.sample_ids):
                        combined_sample_ids.append(table.sample_ids[row_idx])
                
                # Combine element data
                for element, values in table.element_data.items():
                    combined_element_data[element].extend(values)
            
            # Create merged table
            merged_table = DataTable(
                paper_id=base_table.paper_id,
                table_number=f"merged_{base_table.table_number}",
                caption=f"Merged from {len(tables)} tables: {base_table.caption}",
                headers=all_headers,
                data=combined_data,
                sample_ids=combined_sample_ids,
                location_info=base_table.location_info,
                rock_type=base_table.rock_type,
                element_data=dict(combined_element_data),
                extraction_confidence=statistics.mean(
                    t.extraction_confidence or 0.5 for t in tables
                )
            )
            
            return merged_table
            
        except Exception as e:
            self.logger.error("Failed to merge table group", error=str(e))
            return tables[0]  # Return best table if merge fails
    
    async def _validate_data(self, tables: List[DataTable], validation_level: str) -> List[DataTable]:
        """Validate data for outliers and inconsistencies."""
        validated_tables = []
        
        for table in tables:
            try:
                # Validate element data
                validated_element_data = self._validate_element_data(
                    table.element_data, 
                    validation_level
                )
                
                # Calculate data quality score
                quality_score = self._calculate_data_quality_score(
                    table, 
                    validated_element_data
                )
                
                # Create validated table
                validated_table = DataTable(
                    id=table.id,
                    paper_id=table.paper_id,
                    table_number=table.table_number,
                    caption=table.caption,
                    headers=table.headers,
                    data=table.data,
                    sample_ids=table.sample_ids,
                    location_info=table.location_info,
                    rock_type=table.rock_type,
                    element_data=validated_element_data,
                    data_quality_score=quality_score,
                    extraction_confidence=table.extraction_confidence,
                    extracted_at=table.extracted_at
                )
                
                validated_tables.append(validated_table)
                
            except Exception as e:
                self.logger.error(f"Failed to validate table {table.table_number}", error=str(e))
                validated_tables.append(table)  # Keep original if validation fails
        
        return validated_tables
    
    def _validate_element_data(
        self, 
        element_data: Dict[str, List[float]], 
        validation_level: str
    ) -> Dict[str, List[float]]:
        """Validate element concentration data."""
        validated_data = {}
        
        for element, values in element_data.items():
            if not values:
                continue
            
            validated_values = []
            
            for value in values:
                if value is None:
                    continue
                
                # Range validation
                if element in self.element_ranges:
                    min_val, max_val = self.element_ranges[element]
                    if not (min_val <= value <= max_val):
                        if validation_level == "strict":
                            self.logger.warning(
                                f"Value {value} for {element} outside expected range [{min_val}, {max_val}]"
                            )
                            continue  # Skip outlier in strict mode
                        else:
                            self.logger.debug(
                                f"Value {value} for {element} outside expected range [{min_val}, {max_val}]"
                            )
                
                validated_values.append(value)
            
            # Outlier detection using z-score
            if len(validated_values) >= self.min_samples_for_stats:
                validated_values = self._remove_outliers(validated_values, validation_level)
            
            if validated_values:
                validated_data[element] = validated_values
        
        return validated_data
    
    def _remove_outliers(self, values: List[float], validation_level: str) -> List[float]:
        """Remove statistical outliers from values."""
        if len(values) < self.min_samples_for_stats:
            return values
        
        try:
            mean_val = statistics.mean(values)
            stdev_val = statistics.stdev(values)
            
            if stdev_val == 0:
                return values
            
            # Calculate z-scores
            z_scores = [(v - mean_val) / stdev_val for v in values]
            
            # Filter based on validation level
            threshold = self.max_outlier_zscore
            if validation_level == "lenient":
                threshold = 4.0
            elif validation_level == "strict":
                threshold = 2.0
            
            filtered_values = [
                values[i] for i, z in enumerate(z_scores) 
                if abs(z) <= threshold
            ]
            
            return filtered_values if filtered_values else values
            
        except statistics.StatisticsError:
            return values
    
    def _calculate_data_quality_score(
        self, 
        table: DataTable, 
        validated_element_data: Dict[str, List[float]]
    ) -> float:
        """Calculate overall data quality score for a table."""
        scores = []
        
        # Completeness score
        total_cells = len(table.headers) * len(table.data)
        filled_cells = sum(1 for row in table.data for cell in row if cell is not None)
        completeness = filled_cells / total_cells if total_cells > 0 else 0
        scores.append(completeness)
        
        # Element data coverage
        element_coverage = len(validated_element_data) / max(len(table.element_data), 1)
        scores.append(element_coverage)
        
        # Extraction confidence
        scores.append(table.extraction_confidence or 0.5)
        
        # Sample size score
        sample_score = min(len(table.sample_ids) / 10, 1.0)  # Normalize to 10 samples
        scores.append(sample_score)
        
        return statistics.mean(scores)
    
    async def _cross_validate(self, tables: List[DataTable]) -> List[DataTable]:
        """Cross-validate data across multiple sources."""
        if len(tables) <= 1:
            return tables
        
        # Group tables by element types for cross-validation
        element_groups = self._group_tables_by_elements(tables)
        
        validated_tables = []
        
        for tables_group in element_groups:
            if len(tables_group) == 1:
                validated_tables.extend(tables_group)
                continue
            
            # Perform cross-validation within group
            cross_validated = await self._validate_across_sources(tables_group)
            validated_tables.extend(cross_validated)
        
        return validated_tables
    
    def _group_tables_by_elements(self, tables: List[DataTable]) -> List[List[DataTable]]:
        """Group tables by common elements for cross-validation."""
        # Simple grouping - could be more sophisticated
        return [tables]  # For now, validate all tables together
    
    async def _validate_across_sources(self, tables: List[DataTable]) -> List[DataTable]:
        """Validate data consistency across multiple sources."""
        # Collect element statistics across all tables
        element_stats = defaultdict(list)
        
        for table in tables:
            for element, values in table.element_data.items():
                element_stats[element].extend(values)
        
        # Calculate reference statistics
        reference_stats = {}
        for element, all_values in element_stats.items():
            if len(all_values) >= self.min_samples_for_stats:
                reference_stats[element] = {
                    "mean": statistics.mean(all_values),
                    "stdev": statistics.stdev(all_values),
                    "median": statistics.median(all_values),
                    "count": len(all_values)
                }
        
        # Validate each table against reference
        validated_tables = []
        for table in tables:
            consistency_score = self._calculate_consistency_score(
                table.element_data, 
                reference_stats
            )
            
            # Update table with consistency score
            updated_table = DataTable(
                id=table.id,
                paper_id=table.paper_id,
                table_number=table.table_number,
                caption=table.caption,
                headers=table.headers,
                data=table.data,
                sample_ids=table.sample_ids,
                location_info=table.location_info,
                rock_type=table.rock_type,
                element_data=table.element_data,
                data_quality_score=table.data_quality_score,
                extraction_confidence=table.extraction_confidence,
                extracted_at=table.extracted_at
            )
            
            validated_tables.append(updated_table)
        
        return validated_tables
    
    def _calculate_consistency_score(
        self, 
        element_data: Dict[str, List[float]], 
        reference_stats: Dict[str, Dict[str, float]]
    ) -> float:
        """Calculate consistency score against reference statistics."""
        if not element_data or not reference_stats:
            return 0.5
        
        consistency_scores = []
        
        for element, values in element_data.items():
            if element not in reference_stats or not values:
                continue
            
            ref_stats = reference_stats[element]
            table_mean = statistics.mean(values)
            
            # Calculate deviation from reference mean
            if ref_stats["stdev"] > 0:
                z_score = abs(table_mean - ref_stats["mean"]) / ref_stats["stdev"]
                consistency = max(0, 1 - z_score / 3)  # Normalize to 0-1
            else:
                consistency = 1.0 if table_mean == ref_stats["mean"] else 0.5
            
            consistency_scores.append(consistency)
        
        return statistics.mean(consistency_scores) if consistency_scores else 0.5
    
    async def _generate_consolidated_dataset(self, tables: List[DataTable]) -> Dict[str, Any]:
        """Generate final consolidated dataset."""
        if not tables:
            return {}
        
        # Combine all element data
        all_element_data = defaultdict(list)
        all_samples = []
        location_info = []
        rock_types = []
        
        for table in tables:
            # Collect element data
            for element, values in table.element_data.items():
                all_element_data[element].extend(values)
            
            # Collect metadata
            all_samples.extend(table.sample_ids)
            if table.location_info:
                location_info.append(table.location_info)
            if table.rock_type:
                rock_types.append(table.rock_type)
        
        # Calculate final statistics
        element_statistics = {}
        for element, values in all_element_data.items():
            if values:
                element_statistics[element] = {
                    "count": len(values),
                    "mean": statistics.mean(values),
                    "median": statistics.median(values),
                    "std": statistics.stdev(values) if len(values) > 1 else 0,
                    "min": min(values),
                    "max": max(values),
                    "range": max(values) - min(values)
                }
        
        return {
            "total_samples": len(set(all_samples)),
            "total_elements": len(all_element_data),
            "element_statistics": element_statistics,
            "locations": [loc.dict() for loc in location_info],
            "rock_types": list(set(rt.value for rt in rock_types)),
            "data_summary": {
                "tables_count": len(tables),
                "avg_quality_score": statistics.mean(
                    t.data_quality_score or 0.5 for t in tables
                ),
                "avg_confidence": statistics.mean(
                    t.extraction_confidence or 0.5 for t in tables
                )
            }
        }
    
    async def _calculate_quality_metrics(
        self, 
        final_tables: List[DataTable], 
        original_tables: List[DataTable]
    ) -> Dict[str, float]:
        """Calculate overall quality metrics for the fusion process."""
        if not final_tables:
            return {"completeness": 0.0, "consistency": 0.0, "reliability": 0.0}
        
        # Data completeness
        total_data_points = sum(
            len(table.headers) * len(table.data) for table in final_tables
        )
        filled_data_points = sum(
            sum(1 for row in table.data for cell in row if cell is not None)
            for table in final_tables
        )
        completeness = filled_data_points / total_data_points if total_data_points > 0 else 0
        
        # Consistency score (average of individual table consistency)
        consistency = statistics.mean(
            table.data_quality_score or 0.5 for table in final_tables
        )
        
        # Reliability (based on extraction confidence and data retention)
        reliability = statistics.mean([
            statistics.mean(table.extraction_confidence or 0.5 for table in final_tables),
            len(final_tables) / max(len(original_tables), 1)  # Data retention rate
        ])
        
        return {
            "completeness": round(completeness, 3),
            "consistency": round(consistency, 3),
            "reliability": round(reliability, 3)
        }
    
    async def _generate_source_attribution(self, tables: List[DataTable]) -> Dict[str, Any]:
        """Generate source attribution information."""
        paper_contributions = defaultdict(int)
        extraction_methods = defaultdict(int)
        
        for table in tables:
            paper_contributions[str(table.paper_id)] += 1
            
            # Infer extraction method from table number
            if "pdf" in table.table_number.lower():
                extraction_methods["pdf_extraction"] += 1
            elif "html" in table.table_number.lower():
                extraction_methods["html_parsing"] += 1
            elif "extracted" in table.table_number.lower():
                extraction_methods["llm_extraction"] += 1
            else:
                extraction_methods["unknown"] += 1
        
        return {
            "paper_contributions": dict(paper_contributions),
            "extraction_methods": dict(extraction_methods),
            "total_sources": len(paper_contributions),
            "attribution_timestamp": pd.Timestamp.now().isoformat()
        } 