#!/usr/bin/env python3
"""
Extract CVE features from ALL available JSON files (1999-2025) and convert to CSV.
"""

import json
import csv
from pathlib import Path
from typing import Dict, List, Any, Optional
import sys
import os

# Increase CSV field size limit for large fields
csv.field_size_limit(10000000)

def safe_get(data: Dict, *keys, default=None):
    for key in keys:
        if isinstance(data, dict):
            data = data.get(key, {})
        else:
            return default
    return data if data else default

def extract_list_values(data_list: List, key: str, default: List = None) -> str:
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
    
    first_affected = affected[0]
    result["vendor"] = first_affected.get("vendor", "")
    result["product"] = first_affected.get("product", "")
    result["package_name"] = first_affected.get("packageName", "")
    result["collection_url"] = first_affected.get("collectionURL", "")
    
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
    if not credits:
        return ""
    
    credit_list = []
    for credit in credits:
        value = credit.get("value", "")
        if value:
            credit_list.append(value)
    
    return "|".join(credit_list)

def process_cve_file(json_path: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading {json_path}: {e}", file=sys.stderr)
        return None
    
    cve_metadata = data.get("cveMetadata", {})
    cve_id = cve_metadata.get("cveId", "")
    
    if not cve_id:
        return None
    
    state = cve_metadata.get("state", "")
    if state != "PUBLISHED":
        return None
    
    year = ""
    if cve_id.startswith("CVE-"):
        parts = cve_id.split("-")
        if len(parts) >= 2:
            year = parts[1]
    
    containers = data.get("containers", {})
    cna = containers.get("cna", {})
    
    descriptions = cna.get("descriptions", [])
    description = ""
    description_html = ""
    for desc in descriptions:
        if desc.get("lang") == "en":
            description = desc.get("value", "")
            supporting_media = desc.get("supportingMedia", [])
            for media in supporting_media:
                if media.get("type") == "text/html":
                    description_html = media.get("value", "")
                    break
            break
    
    problem_types = cna.get("problemTypes", [])
    cwe_ids, cwe_descriptions, problem_type_texts = extract_cwe_ids(problem_types)
    
    impacts = cna.get("impacts", [])
    capec_ids, capec_descriptions = extract_capec_ids(impacts)
    
    metrics = cna.get("metrics", [])
    cvss_data = extract_cvss_metrics(metrics)
    
    affected = cna.get("affected", [])
    affected_data = extract_affected_products(affected)
    
    references = cna.get("references", [])
    ref_data = extract_references(references)
    
    provider_metadata = cna.get("providerMetadata", {})
    
    credits = cna.get("credits", [])
    credits_str = extract_credits(credits)
    
    source = cna.get("source", {})
    
    adp = containers.get("adp", [])
    ssvc_data = extract_ssvc_metrics(adp)
    
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

def main():
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data" / "cvelistV5-main" / "cves"
    output_csv = project_root / "results" / "cve_features_all.csv"
    
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    
    # Dynamically find all years
    print(f"Scanning for ALL CVE directories in {data_dir}...")
    all_year_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.isdigit()], key=lambda x: int(x.name))
    years = [int(d.name) for d in all_year_dirs]
    print(f"Found years: {years}")
    
    cve_files = []
    for year_dir in all_year_dirs:
        files = list(year_dir.rglob("*.json"))
        cve_files.extend(files)
    
    print(f"Found {len(cve_files)} total CVE files across all years.")
    
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
    
    print(f"Processing and writing to {output_csv}...")
    processed = 0
    filtered = 0
    errors = 0
    
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for i, cve_file in enumerate(cve_files):
            if (i + 1) % 5000 == 0:
                print(f"Processed {i + 1}/{len(cve_files)} files... (Published: {processed}, Filtered: {filtered}, Errors: {errors})")
            
            features = process_cve_file(cve_file)
            if features:
                writer.writerow(features)
                processed += 1
            elif features is None:
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
