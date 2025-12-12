#!/usr/bin/env python3
"""
Analyze Japan-related vs Non-Japan-related CVEs (2021-2025)
Three categories: exploited_kev, has_indicators, not_exploited

Adapted to work with the extracted CVE CSV data.
"""

import csv
import json
from pathlib import Path
from collections import defaultdict
import sys

# Increase CSV field size limit for large fields
csv.field_size_limit(10000000)


def check_japan_related(text):
    """Check if text mentions JVN, JPCERT, or Japan-related terms (case-insensitive)."""
    if not text or not isinstance(text, str):
        return False
    text_lower = text.lower()
    japan_keywords = ['jvn', 'jpcert', 'jvnv', 'jvndb']
    return any(keyword in text_lower for keyword in japan_keywords)


def load_kev_data(kev_file):
    """Load KEV data and return a set of exploited CVE IDs."""
    exploited_cves = set()
    
    if not Path(kev_file).exists():
        print(f"⚠️  KEV file not found: {kev_file}")
        return exploited_cves
    
    try:
        with open(kev_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                cve_id = row.get('cve_id', '').strip()
                # Load all CVE IDs from KEV file (matching original script behavior)
                if cve_id:
                    exploited_cves.add(cve_id.upper())
        
        print(f"✅ Loaded {len(exploited_cves):,} exploited CVEs from KEV data")
    except Exception as e:
        print(f"⚠️  Error loading KEV data: {e}")
    
    return exploited_cves


def analyze_cve_row(row):
    """Analyze a single CVE row from CSV for Japan-related content and exploitation indicators."""
    cve_id = row.get('cve_id', '').strip()
    if not cve_id:
        return None
    
    has_japan = False
    exploitation_indicators = []
    
    # Check description for Japan-related content
    description = row.get('description', '')
    if check_japan_related(description):
        has_japan = True
    
    # Check description_html for Japan-related content
    if not has_japan:
        description_html = row.get('description_html', '')
        if check_japan_related(description_html):
            has_japan = True
    
    # Check reference URLs for Japan-related content
    if not has_japan:
        reference_urls = row.get('reference_urls', '')
        if reference_urls:
            for url in reference_urls.split('|'):
                if check_japan_related(url):
                    has_japan = True
                    break
    
    # Check for exploitation indicators in description
    description_lower = description.lower()
    if any(keyword in description_lower for keyword in ['exploit', 'exploited', 'active exploitation', 'in the wild']):
        exploitation_indicators.append('description_keyword')
    
    # Check CVSS score for high severity (>= 9.0)
    cvss_base_score_str = row.get('cvss_base_score', '')
    if cvss_base_score_str:
        try:
            cvss_score = float(cvss_base_score_str)
            if cvss_score >= 9.0:
                exploitation_indicators.append('high_cvss')
        except (ValueError, TypeError):
            pass
    
    return {
        'cve_id': cve_id,
        'has_japan': has_japan,
        'exploitation_indicators': exploitation_indicators
    }


def main():
    # Get project root directory (parent of scripts directory)
    project_root = Path(__file__).parent.parent
    
    # Configuration - using relative paths from project root
    cve_csv_file = project_root / 'results' / 'cve_features_2021_2025.csv'
    kev_file = project_root / 'data' / 'unified_vulnerability_dataset_real_kev.csv'
    
    if not cve_csv_file.exists():
        print(f"Error: CVE CSV file not found: {cve_csv_file}")
        return
    
    print("=" * 80)
    print("JAPAN vs NON-JAPAN CVEs ANALYSIS (2021-2025)")
    print("Three Categories: exploited_kev, has_indicators, not_exploited")
    print("=" * 80)
    
    # Load KEV data
    print("\n[1] Loading KEV (Known Exploited Vulnerabilities) data...")
    exploited_cves = load_kev_data(kev_file)
    
    # Statistics
    stats = {
        'japan_exploited_kev': 0,
        'japan_has_indicators': 0,
        'japan_not_exploited': 0,
        'non_japan_exploited_kev': 0,
        'non_japan_has_indicators': 0,
        'non_japan_not_exploited': 0,
        'by_year': {}
    }
    
    all_cves = []
    
    print(f"\n[2] Reading CVE data from CSV...")
    
    # Process CSV file
    try:
        with open(cve_csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            total_rows = 0
            
            for row in reader:
                total_rows += 1
                if total_rows % 10000 == 0:
                    print(f"  Processed {total_rows:,} CVEs...", end='\r', flush=True)
                
                result = analyze_cve_row(row)
                if result is None:
                    continue
                
                cve_id = result['cve_id']
                has_japan = result['has_japan']
                year = row.get('year', '')
                
                is_exploited_kev = cve_id and cve_id.upper() in exploited_cves
                has_indicators = len(result['exploitation_indicators']) > 0
                
                # Categorize into three groups
                if is_exploited_kev:
                    status = 'exploited_kev'
                elif has_indicators:
                    status = 'has_indicators'
                else:
                    status = 'not_exploited'
                
                # Initialize year stats if needed
                if year not in stats['by_year']:
                    stats['by_year'][year] = {
                        'japan_exploited_kev': 0,
                        'japan_has_indicators': 0,
                        'japan_not_exploited': 0,
                        'non_japan_exploited_kev': 0,
                        'non_japan_has_indicators': 0,
                        'non_japan_not_exploited': 0,
                        'total': 0
                    }
                
                year_stats = stats['by_year'][year]
                year_stats['total'] += 1
                
                # Update statistics
                if has_japan:
                    if status == 'exploited_kev':
                        year_stats['japan_exploited_kev'] += 1
                        stats['japan_exploited_kev'] += 1
                    elif status == 'has_indicators':
                        year_stats['japan_has_indicators'] += 1
                        stats['japan_has_indicators'] += 1
                    else:
                        year_stats['japan_not_exploited'] += 1
                        stats['japan_not_exploited'] += 1
                else:
                    if status == 'exploited_kev':
                        year_stats['non_japan_exploited_kev'] += 1
                        stats['non_japan_exploited_kev'] += 1
                    elif status == 'has_indicators':
                        year_stats['non_japan_has_indicators'] += 1
                        stats['non_japan_has_indicators'] += 1
                    else:
                        year_stats['non_japan_not_exploited'] += 1
                        stats['non_japan_not_exploited'] += 1
                
                all_cves.append({
                    'cve_id': cve_id,
                    'year': year,
                    'has_japan': has_japan,
                    'status': status,
                    'in_kev': is_exploited_kev,
                    'has_indicators': has_indicators
                })
            
            print(f"\n  Processed {total_rows:,} CVEs total")
    
    except Exception as e:
        print(f"\nError reading CSV file: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Calculate totals
    japan_total = stats['japan_exploited_kev'] + stats['japan_has_indicators'] + stats['japan_not_exploited']
    non_japan_total = stats['non_japan_exploited_kev'] + stats['non_japan_has_indicators'] + stats['non_japan_not_exploited']
    total_cves = japan_total + non_japan_total
    
    # Print summary
    print("\n" + "=" * 80)
    print("OVERALL SUMMARY (2021-2025)")
    print("=" * 80)
    
    print(f"\nTotal CVEs analyzed: {total_cves:,}")
    print(f"  - Japan-related: {japan_total:,} ({japan_total/total_cves*100:.2f}%)")
    print(f"  - Non-Japan-related: {non_japan_total:,} ({non_japan_total/total_cves*100:.2f}%)")
    
    print(f"\n" + "=" * 80)
    print("JAPAN-RELATED CVEs BREAKDOWN")
    print("=" * 80)
    
    jp_kev_pct = (stats['japan_exploited_kev'] / japan_total * 100) if japan_total > 0 else 0
    jp_indicators_pct = (stats['japan_has_indicators'] / japan_total * 100) if japan_total > 0 else 0
    jp_not_exploited_pct = (stats['japan_not_exploited'] / japan_total * 100) if japan_total > 0 else 0
    
    print(f"\n🇯🇵 JAPAN-RELATED: {japan_total:,}")
    print(f"  - exploited_kev: {stats['japan_exploited_kev']:,} ({jp_kev_pct:.2f}%)")
    print(f"  - has_indicators: {stats['japan_has_indicators']:,} ({jp_indicators_pct:.2f}%)")
    print(f"  - not_exploited: {stats['japan_not_exploited']:,} ({jp_not_exploited_pct:.2f}%)")
    
    print(f"\n" + "=" * 80)
    print("NON-JAPAN-RELATED CVEs BREAKDOWN")
    print("=" * 80)
    
    non_jp_kev_pct = (stats['non_japan_exploited_kev'] / non_japan_total * 100) if non_japan_total > 0 else 0
    non_jp_indicators_pct = (stats['non_japan_has_indicators'] / non_japan_total * 100) if non_japan_total > 0 else 0
    non_jp_not_exploited_pct = (stats['non_japan_not_exploited'] / non_japan_total * 100) if non_japan_total > 0 else 0
    
    print(f"\n🌍 NON-JAPAN-RELATED: {non_japan_total:,}")
    print(f"  - exploited_kev: {stats['non_japan_exploited_kev']:,} ({non_jp_kev_pct:.2f}%)")
    print(f"  - has_indicators: {stats['non_japan_has_indicators']:,} ({non_jp_indicators_pct:.2f}%)")
    print(f"  - not_exploited: {stats['non_japan_not_exploited']:,} ({non_jp_not_exploited_pct:.2f}%)")
    
    # Comparison table
    print(f"\n" + "=" * 80)
    print("COMPARISON TABLE")
    print("=" * 80)
    print(f"\n{'Category':<25} {'exploited_kev':<20} {'has_indicators':<20} {'not_exploited':<20}")
    print(f"{'':<25} {'Count':<10} {'%':<10} {'Count':<10} {'%':<10} {'Count':<10} {'%':<10}")
    print("-" * 80)
    
    print(f"{'Japan-related':<25} "
          f"{stats['japan_exploited_kev']:>8,} {jp_kev_pct:>8.2f}% "
          f"{stats['japan_has_indicators']:>8,} {jp_indicators_pct:>8.2f}% "
          f"{stats['japan_not_exploited']:>8,} {jp_not_exploited_pct:>8.2f}%")
    
    print(f"{'Non-Japan-related':<25} "
          f"{stats['non_japan_exploited_kev']:>8,} {non_jp_kev_pct:>8.2f}% "
          f"{stats['non_japan_has_indicators']:>8,} {non_jp_indicators_pct:>8.2f}% "
          f"{stats['non_japan_not_exploited']:>8,} {non_jp_not_exploited_pct:>8.2f}%")
    
    print("-" * 80)
    total_kev = stats['japan_exploited_kev'] + stats['non_japan_exploited_kev']
    total_indicators = stats['japan_has_indicators'] + stats['non_japan_has_indicators']
    total_not_exploited = stats['japan_not_exploited'] + stats['non_japan_not_exploited']
    total_kev_pct = (total_kev / total_cves * 100) if total_cves > 0 else 0
    total_indicators_pct = (total_indicators / total_cves * 100) if total_cves > 0 else 0
    total_not_exploited_pct = (total_not_exploited / total_cves * 100) if total_cves > 0 else 0
    
    print(f"{'TOTAL':<25} "
          f"{total_kev:>8,} {total_kev_pct:>8.2f}% "
          f"{total_indicators:>8,} {total_indicators_pct:>8.2f}% "
          f"{total_not_exploited:>8,} {total_not_exploited_pct:>8.2f}%")
    
    # By year breakdown
    print(f"\n" + "=" * 80)
    print("BY YEAR BREAKDOWN")
    print("=" * 80)
    print(f"\n{'Year':<8} {'Category':<20} {'exploited_kev':<15} {'has_indicators':<15} {'not_exploited':<15}")
    print("-" * 80)
    
    for year in sorted(stats['by_year'].keys()):
        year_stat = stats['by_year'][year]
        jp_total = year_stat['japan_exploited_kev'] + year_stat['japan_has_indicators'] + year_stat['japan_not_exploited']
        non_jp_total = year_stat['non_japan_exploited_kev'] + year_stat['non_japan_has_indicators'] + year_stat['non_japan_not_exploited']
        
        if jp_total > 0:
            jp_kev_pct = (year_stat['japan_exploited_kev'] / jp_total * 100)
            jp_ind_pct = (year_stat['japan_has_indicators'] / jp_total * 100)
            jp_not_pct = (year_stat['japan_not_exploited'] / jp_total * 100)
        else:
            jp_kev_pct = jp_ind_pct = jp_not_pct = 0
        
        if non_jp_total > 0:
            non_jp_kev_pct = (year_stat['non_japan_exploited_kev'] / non_jp_total * 100)
            non_jp_ind_pct = (year_stat['non_japan_has_indicators'] / non_jp_total * 100)
            non_jp_not_pct = (year_stat['non_japan_not_exploited'] / non_jp_total * 100)
        else:
            non_jp_kev_pct = non_jp_ind_pct = non_jp_not_pct = 0
        
        print(f"{year:<8} {'Japan-related':<20} "
              f"{year_stat['japan_exploited_kev']:>4} ({jp_kev_pct:>5.2f}%) "
              f"{year_stat['japan_has_indicators']:>4} ({jp_ind_pct:>5.2f}%) "
              f"{year_stat['japan_not_exploited']:>4} ({jp_not_pct:>5.2f}%)")
        
        print(f"{'':<8} {'Non-Japan-related':<20} "
              f"{year_stat['non_japan_exploited_kev']:>4} ({non_jp_kev_pct:>5.2f}%) "
              f"{year_stat['non_japan_has_indicators']:>4} ({non_jp_ind_pct:>5.2f}%) "
              f"{year_stat['non_japan_not_exploited']:>4} ({non_jp_not_pct:>5.2f}%)")
        print()
    
    # Save results
    results_dir = project_root / 'results'
    results_dir.mkdir(exist_ok=True)
    
    output_file = results_dir / 'jp_vs_nonjp_2021_2025_three_categories.json'
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'summary': {
                'total_cves': total_cves,
                'japan_total': japan_total,
                'japan_exploited_kev': stats['japan_exploited_kev'],
                'japan_exploited_kev_pct': jp_kev_pct,
                'japan_has_indicators': stats['japan_has_indicators'],
                'japan_has_indicators_pct': jp_indicators_pct,
                'japan_not_exploited': stats['japan_not_exploited'],
                'japan_not_exploited_pct': jp_not_exploited_pct,
                'non_japan_total': non_japan_total,
                'non_japan_exploited_kev': stats['non_japan_exploited_kev'],
                'non_japan_exploited_kev_pct': non_jp_kev_pct,
                'non_japan_has_indicators': stats['non_japan_has_indicators'],
                'non_japan_has_indicators_pct': non_jp_indicators_pct,
                'non_japan_not_exploited': stats['non_japan_not_exploited'],
                'non_japan_not_exploited_pct': non_jp_not_exploited_pct
            },
            'by_year': stats['by_year']
        }, f, indent=2, ensure_ascii=False)
    
    # Save CSV
    csv_file = results_dir / 'jp_vs_nonjp_2021_2025_three_categories.csv'
    with open(csv_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['cve_id', 'year', 'has_japan', 'status', 'in_kev', 'has_indicators'])
        writer.writeheader()
        for cve in all_cves:
            writer.writerow(cve)
    
    print(f"\n✅ Detailed results saved to:")
    print(f"   - {output_file}")
    print(f"   - {csv_file}")
    print("=" * 80)


if __name__ == '__main__':
    main()

