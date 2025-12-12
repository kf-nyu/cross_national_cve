#!/usr/bin/env python3
"""
Extract CVE features from JSON files (2021-2025) and convert to CSV.

This script processes CVE JSON files in CVE 5.1 format and extracts
all defined features into a CSV file for analysis.
"""

import json
import csv
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import sys


def safe_get(data: Dict, *keys, default=None):
    """Safely navigate nested dictionaries."""
    for key in keys:
        if isinstance(data, dict):
            data = data.get(key, {})
        else:
            return default
    return data if data else default


def extract_list_values(data_list: List, key: str, default: List = None) -> str:
    """Extract values from a list of dictionaries and join with pipe."""
    if not data_list or not isinstance(data_list, list):
        return ""
    values = []
    for item in data_list:
        if isinstance(item, dict):
            value = item.get(key, "")
            if value:
                values.append(str(value))
    return "|".join(values) if values else ""


def extract_cwe_ids(problem_types: List) -> tuple:
    """Extract CWE IDs and descriptions from problem types."""
    cwe_ids = []
    cwe_descriptions = []
    problem_type_texts = []
    
    if not problem_types:
        return ("", "", "")
    
    for pt in problem_types:
        descriptions = pt.get("descriptions", [])
        for desc in descriptions:
            desc_type = desc.get("type", "")
            if desc_type == "CWE":
                cwe_id = desc.get("cweId", "")
                cwe_desc = desc.get("description", "")
                if cwe_id:
                    cwe_ids.append(cwe_id)
                if cwe_desc:
                    cwe_descriptions.append(cwe_desc)
            elif desc_type == "text":
                text_desc = desc.get("description", "")
                if text_desc:
                    problem_type_texts.append(text_desc)
    
    return (
        "|".join(cwe_ids),
        "|".join(cwe_descriptions),
        "|".join(problem_type_texts)
    )


def extract_capec_ids(impacts: List) -> tuple:
    """Extract CAPEC IDs and descriptions from impacts."""
    capec_ids = []
    capec_descriptions = []
    
    if not impacts:
        return ("", "")
    
    for impact in impacts:
        capec_id = impact.get("capecId", "")
        if capec_id:
            capec_ids.append(capec_id)
        
        descriptions = impact.get("descriptions", [])
        for desc in descriptions:
            capec_desc = desc.get("value", "")
            if capec_desc:
                capec_descriptions.append(capec_desc)
    
    return ("|".join(capec_ids), "|".join(capec_descriptions))


def extract_cvss_metrics(metrics: List) -> Dict[str, Any]:
    """Extract CVSS metrics from metrics list."""
    result = {
        "cvss_version": "",
        "cvss_base_score": "",
        "cvss_base_severity": "",
        "cvss_vector_string": "",
        "cvss_attack_vector": "",
        "cvss_attack_complexity": "",
        "cvss_privileges_required": "",
        "cvss_user_interaction": "",
        "cvss_scope": "",
        "cvss_confidentiality_impact": "",
        "cvss_integrity_impact": "",
        "cvss_availability_impact": "",
    }
    
    if not metrics:
        return result
    
    # Find CVSS metrics
    for metric in metrics:
        if metric.get("format") == "CVSS":
            cvss_v3_1 = metric.get("cvssV3_1", {})
            cvss_v3_0 = metric.get("cvssV3_0", {})
            cvss_v2_0 = metric.get("cvssV2_0", {})
            
            cvss_data = cvss_v3_1 or cvss_v3_0 or cvss_v2_0
            
            if cvss_data:
                result["cvss_version"] = cvss_data.get("version", "")
                result["cvss_base_score"] = cvss_data.get("baseScore", "")
                result["cvss_base_severity"] = cvss_data.get("baseSeverity", "")
                result["cvss_vector_string"] = cvss_data.get("vectorString", "")
                result["cvss_attack_vector"] = cvss_data.get("attackVector", "")
                result["cvss_attack_complexity"] = cvss_data.get("attackComplexity", "")
                result["cvss_privileges_required"] = cvss_data.get("privilegesRequired", "")
                result["cvss_user_interaction"] = cvss_data.get("userInteraction", "")
                result["cvss_scope"] = cvss_data.get("scope", "")
                result["cvss_confidentiality_impact"] = cvss_data.get("confidentialityImpact", "")
                result["cvss_integrity_impact"] = cvss_data.get("integrityImpact", "")
                result["cvss_availability_impact"] = cvss_data.get("availabilityImpact", "")
                break
    
    return result


def extract_affected_products(affected: List) -> Dict[str, Any]:
    """Extract affected product information."""
    result = {
        "vendor": "",
        "product": "",
        "package_name": "",
        "affected_versions": "",
        "version_type": "",
        "collection_url": "",
    }
    
    if not affected or len(affected) == 0:
        return result
    
    # Get first affected product
    first_affected = affected[0]
    result["vendor"] = first_affected.get("vendor", "")
    result["product"] = first_affected.get("product", "")
    result["package_name"] = first_affected.get("packageName", "")
    result["collection_url"] = first_affected.get("collectionURL", "")
    
    # Extract versions
    versions = first_affected.get("versions", [])
    version_list = []
    version_types = []
    for v in versions:
        version_str = v.get("version", "")
        version_type = v.get("versionType", "")
        if version_str and version_str != "n/a":
            version_list.append(version_str)
        if version_type:
            version_types.append(version_type)
    
    result["affected_versions"] = "|".join(version_list)
    result["version_type"] = "|".join(set(version_types))
    
    return result


def extract_references(references: List) -> Dict[str, Any]:
    """Extract reference information."""
    result = {
        "reference_count": len(references) if references else 0,
        "reference_urls": "",
        "reference_tags": "",
    }
    
    if not references:
        return result
    
    urls = []
    all_tags = []
    
    for ref in references:
        url = ref.get("url", "")
        if url:
            urls.append(url)
        
        tags = ref.get("tags", [])
        for tag in tags:
            if tag not in all_tags:
                all_tags.append(tag)
    
    result["reference_urls"] = "|".join(urls)
    result["reference_tags"] = "|".join(all_tags)
    
    return result


def extract_ssvc_metrics(adp: List) -> Dict[str, Any]:
    """Extract SSVC metrics from ADP containers."""
    result = {
        "ssvc_exploitation": "",
        "ssvc_automatable": "",
        "ssvc_technical_impact": "",
        "ssvc_version": "",
    }
    
    if not adp:
        return result
    
    for adp_item in adp:
        metrics = adp_item.get("metrics", [])
        for metric in metrics:
            other = metric.get("other", {})
            if other.get("type") == "ssvc":
                content = other.get("content", {})
                options = content.get("options", [])
                
                result["ssvc_version"] = content.get("version", "")
                
                for option in options:
                    if "Exploitation" in option:
                        result["ssvc_exploitation"] = option["Exploitation"]
                    if "Automatable" in option:
                        result["ssvc_automatable"] = option["Automatable"]
                    if "Technical Impact" in option:
                        result["ssvc_technical_impact"] = option["Technical Impact"]
    
    return result


def extract_credits(credits: List) -> str:
    """Extract credits information."""
    if not credits:
        return ""
    
    credit_list = []
    for credit in credits:
        value = credit.get("value", "")
        if value:
            credit_list.append(value)
    
    return "|".join(credit_list)


def process_cve_file(json_path: Path) -> Optional[Dict[str, Any]]:
    """Process a single CVE JSON file and extract features."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading {json_path}: {e}", file=sys.stderr)
        return None
    
    # Extract CVE metadata
    cve_metadata = data.get("cveMetadata", {})
    cve_id = cve_metadata.get("cveId", "")
    
    if not cve_id:
        return None
    
    # Filter: only process PUBLISHED CVEs
    state = cve_metadata.get("state", "")
    if state != "PUBLISHED":
        return None
    
    # Extract year from CVE ID
    year = ""
    if cve_id.startswith("CVE-"):
        parts = cve_id.split("-")
        if len(parts) >= 2:
            year = parts[1]
    
    # Extract containers
    containers = data.get("containers", {})
    cna = containers.get("cna", {})
    
    # Extract descriptions
    descriptions = cna.get("descriptions", [])
    description = ""
    description_html = ""
    for desc in descriptions:
        if desc.get("lang") == "en":
            description = desc.get("value", "")
            # Check for HTML description
            supporting_media = desc.get("supportingMedia", [])
            for media in supporting_media:
                if media.get("type") == "text/html":
                    description_html = media.get("value", "")
                    break
            break
    
    # Extract problem types
    problem_types = cna.get("problemTypes", [])
    cwe_ids, cwe_descriptions, problem_type_texts = extract_cwe_ids(problem_types)
    
    # Extract impacts
    impacts = cna.get("impacts", [])
    capec_ids, capec_descriptions = extract_capec_ids(impacts)
    
    # Extract CVSS metrics
    metrics = cna.get("metrics", [])
    cvss_data = extract_cvss_metrics(metrics)
    
    # Extract affected products
    affected = cna.get("affected", [])
    affected_data = extract_affected_products(affected)
    
    # Extract references
    references = cna.get("references", [])
    ref_data = extract_references(references)
    
    # Extract provider metadata
    provider_metadata = cna.get("providerMetadata", {})
    
    # Extract credits
    credits = cna.get("credits", [])
    credits_str = extract_credits(credits)
    
    # Extract source
    source = cna.get("source", {})
    
    # Extract SSVC metrics from ADP
    adp = containers.get("adp", [])
    ssvc_data = extract_ssvc_metrics(adp)
    
    # Build feature dictionary
    features = {
        "cve_id": cve_id,
        "year": year,
        "state": cve_metadata.get("state", ""),
        "assigner_org_id": cve_metadata.get("assignerOrgId", ""),
        "assigner_short_name": cve_metadata.get("assignerShortName", ""),
        "date_reserved": cve_metadata.get("dateReserved", ""),
        "date_published": cve_metadata.get("datePublished", ""),
        "date_updated": cve_metadata.get("dateUpdated", ""),
        "description": description,
        "description_html": description_html,
        "cwe_ids": cwe_ids,
        "cwe_descriptions": cwe_descriptions,
        "problem_type_text": problem_type_texts,
        "capec_ids": capec_ids,
        "capec_descriptions": capec_descriptions,
        **cvss_data,
        **affected_data,
        **ref_data,
        "provider_org_id": provider_metadata.get("orgId", ""),
        "provider_short_name": provider_metadata.get("shortName", ""),
        "provider_date_updated": provider_metadata.get("dateUpdated", ""),
        "title": cna.get("title", ""),
        "credits": credits_str,
        "source_discovery": source.get("discovery", ""),
        "has_metrics": 1 if metrics else 0,
        "has_affected": 1 if affected else 0,
        "has_references": 1 if references else 0,
        **ssvc_data,
    }
    
    return features


def get_cve_files(data_dir: Path, years: List[int]) -> List[Path]:
    """Get all CVE JSON files for specified years."""
    cve_files = []
    
    for year in years:
        year_dir = data_dir / str(year)
        if year_dir.exists():
            # Find all JSON files recursively
            json_files = list(year_dir.rglob("*.json"))
            cve_files.extend(json_files)
    
    return sorted(cve_files)


def main():
    """Main function to process CVE files and create CSV."""
    # Get project root directory (parent of scripts directory)
    project_root = Path(__file__).parent.parent
    
    # Configuration - using relative paths from project root
    data_dir = project_root / "data" / "cvelistV5-main" / "cves"
    output_csv = project_root / "results" / "cve_features_2021_2025.csv"
    years = [2021, 2022, 2023, 2024, 2025]
    
    # Create output directory if it doesn't exist
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    
    # Get all CVE files
    print(f"Scanning for CVE files in years {years}...")
    cve_files = get_cve_files(data_dir, years)
    print(f"Found {len(cve_files)} CVE files")
    
    # Define CSV columns (order matters)
    fieldnames = [
        "cve_id", "year", "state", "assigner_org_id", "assigner_short_name",
        "date_reserved", "date_published", "date_updated",
        "description", "description_html",
        "cwe_ids", "cwe_descriptions", "problem_type_text",
        "capec_ids", "capec_descriptions",
        "cvss_version", "cvss_base_score", "cvss_base_severity", "cvss_vector_string",
        "cvss_attack_vector", "cvss_attack_complexity", "cvss_privileges_required",
        "cvss_user_interaction", "cvss_scope",
        "cvss_confidentiality_impact", "cvss_integrity_impact", "cvss_availability_impact",
        "vendor", "product", "package_name", "affected_versions", "version_type", "collection_url",
        "reference_count", "reference_urls", "reference_tags",
        "provider_org_id", "provider_short_name", "provider_date_updated",
        "title", "credits", "source_discovery",
        "has_metrics", "has_affected", "has_references",
        "ssvc_exploitation", "ssvc_automatable", "ssvc_technical_impact", "ssvc_version",
    ]
    
    # Process files and write to CSV
    print(f"Processing CVE files (filtering for state='PUBLISHED') and writing to {output_csv}...")
    processed = 0
    filtered = 0
    errors = 0
    
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for i, cve_file in enumerate(cve_files):
            if (i + 1) % 1000 == 0:
                print(f"Processed {i + 1}/{len(cve_files)} files... (Published: {processed}, Filtered: {filtered}, Errors: {errors})")
            
            features = process_cve_file(cve_file)
            if features:
                writer.writerow(features)
                processed += 1
            elif features is None:
                # None can mean either filtered (non-PUBLISHED) or error
                # We'll count as filtered since errors print to stderr
                filtered += 1
            else:
                errors += 1
    
    print(f"\nProcessing complete!")
    print(f"  Total files: {len(cve_files)}")
    print(f"  Successfully processed (PUBLISHED): {processed}")
    print(f"  Filtered out (non-PUBLISHED): {filtered}")
    print(f"  Errors: {errors}")
    print(f"  Output CSV: {output_csv}")


if __name__ == "__main__":
    main()

