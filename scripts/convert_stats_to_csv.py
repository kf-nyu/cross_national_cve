import json
import csv
from pathlib import Path

def main():
    project_root = Path(__file__).parent.parent
    json_file = project_root / 'results' / 'jp_vs_nonjp_all_years.json'
    output_csv = project_root / 'results' / 'jp_vs_nonjp_all_years_stats.csv'
    
    if not json_file.exists():
        print(f"Error: JSON file not found: {json_file}")
        return

    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        by_year = data.get('by_year', {})
        
        # Define columns
        fieldnames = [
            'year',
            'japan_exploited_kev', 'japan_has_indicators', 'japan_not_exploited', 'japan_total',
            'non_japan_exploited_kev', 'non_japan_has_indicators', 'non_japan_not_exploited', 'non_japan_total',
            'grand_total'
        ]
        
        rows = []
        for year, stats in sorted(by_year.items()):
            # Calculate derived totals if not present (script had them, but safer to recalc or fetch)
            jp_total = (stats.get('japan_exploited_kev', 0) + 
                       stats.get('japan_has_indicators', 0) + 
                       stats.get('japan_not_exploited', 0))
                       
            non_jp_total = (stats.get('non_japan_exploited_kev', 0) + 
                           stats.get('non_japan_has_indicators', 0) + 
                           stats.get('non_japan_not_exploited', 0))
            
            row = {
                'year': year,
                'japan_exploited_kev': stats.get('japan_exploited_kev', 0),
                'japan_has_indicators': stats.get('japan_has_indicators', 0),
                'japan_not_exploited': stats.get('japan_not_exploited', 0),
                'japan_total': jp_total,
                'non_japan_exploited_kev': stats.get('non_japan_exploited_kev', 0),
                'non_japan_has_indicators': stats.get('non_japan_has_indicators', 0),
                'non_japan_not_exploited': stats.get('non_japan_not_exploited', 0),
                'non_japan_total': non_jp_total,
                'grand_total': jp_total + non_jp_total
            }
            rows.append(row)
            
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            
        print(f"✅ Converted stats to CSV: {output_csv}")
        
    except Exception as e:
        print(f"Error converting JSON to CSV: {e}")

if __name__ == "__main__":
    main()
