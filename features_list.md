# CVE Features List

This document defines all features extracted from CVE JSON files (CVE 5.1 format) for the cross-national CVE analysis project.

## Core Metadata Features

1. **cve_id** - CVE identifier (e.g., CVE-2021-28091)
2. **year** - Year from CVE ID
3. **state** - Publication state (PUBLISHED, REJECTED, etc.)
4. **assigner_org_id** - Organization ID that assigned the CVE
5. **assigner_short_name** - Short name of assigning organization
6. **date_reserved** - Date when CVE was reserved
7. **date_published** - Date when CVE was published
8. **date_updated** - Date when CVE was last updated

## Description Features

9. **description** - Primary description text (English)
10. **description_html** - HTML formatted description (if available)

## Problem Type Features

11. **cwe_ids** - List of CWE IDs (Common Weakness Enumeration)
12. **cwe_descriptions** - List of CWE descriptions
13. **problem_type_text** - Text-based problem type descriptions

## Impact Features

14. **capec_ids** - List of CAPEC IDs (Common Attack Pattern Enumeration)
15. **capec_descriptions** - List of CAPEC descriptions

## CVSS Metrics Features

16. **cvss_version** - CVSS version (e.g., 3.1, 3.0, 2.0)
17. **cvss_base_score** - Base CVSS score
18. **cvss_base_severity** - Base severity (CRITICAL, HIGH, MEDIUM, LOW, NONE)
19. **cvss_vector_string** - CVSS vector string
20. **cvss_attack_vector** - Attack vector (NETWORK, ADJACENT_NETWORK, LOCAL, PHYSICAL)
21. **cvss_attack_complexity** - Attack complexity (LOW, HIGH)
22. **cvss_privileges_required** - Privileges required (NONE, LOW, HIGH)
23. **cvss_user_interaction** - User interaction required (NONE, REQUIRED)
24. **cvss_scope** - Scope (UNCHANGED, CHANGED)
25. **cvss_confidentiality_impact** - Confidentiality impact (NONE, LOW, HIGH)
26. **cvss_integrity_impact** - Integrity impact (NONE, LOW, HIGH)
27. **cvss_availability_impact** - Availability impact (NONE, LOW, HIGH)

## Affected Products Features

28. **vendor** - Primary vendor name
29. **product** - Primary product name
30. **package_name** - Package name (if applicable)
31. **affected_versions** - List of affected versions
32. **version_type** - Version type (semver, custom, etc.)
33. **collection_url** - Collection URL for the product

## Reference Features

34. **reference_count** - Number of references
35. **reference_urls** - List of reference URLs
36. **reference_tags** - List of reference tags (vendor-advisory, exploit, etc.)

## Provider Features

37. **provider_org_id** - Provider organization ID
38. **provider_short_name** - Provider short name
39. **provider_date_updated** - Provider date updated

## Additional Features

40. **title** - CVE title
41. **credits** - List of credits/acknowledgments
42. **source_discovery** - Source of discovery (EXTERNAL, INTERNAL, etc.)
43. **has_metrics** - Boolean indicating if CVSS metrics exist
44. **has_affected** - Boolean indicating if affected products are listed
45. **has_references** - Boolean indicating if references exist

## SSVC Features (if available)

46. **ssvc_exploitation** - SSVC exploitation status
47. **ssvc_automatable** - SSVC automatable status
48. **ssvc_technical_impact** - SSVC technical impact
49. **ssvc_version** - SSVC version

## Notes

- Some features may be missing (null/empty) for certain CVEs
- Lists are stored as pipe-separated strings in CSV format
- Dates are stored in ISO 8601 format
- Boolean features are stored as 1/0 or True/False

