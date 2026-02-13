import os
import requests
from dotenv import load_dotenv
import pandas as pd
import time

# Load environment variables
load_dotenv()

class CollegeScorecardCollector:
    def __init__(self):
        self.api_key = os.getenv('COLLEGE_SCORECARD_API_KEY')
        self.base_url = "https://api.data.gov/ed/collegescorecard/v1/schools"
        
    def collect_institution_data(self, max_schools=1000):
        """Collect comprehensive institution-level data with correct field names"""
        
        # Corrected field names based on API documentation
        fields = [
            'id', 
            'school.name', 
            'school.state', 
            'school.city',
            'school.ownership',  # 1=Public, 2=Private nonprofit, 3=Private for-profit
            'latest.student.size',
            'latest.admissions.admission_rate.overall',
            'latest.admissions.sat_scores.average.overall',
            'latest.admissions.act_scores.midpoint.cumulative',
            'latest.cost.tuition.in_state',
            'latest.cost.tuition.out_of_state', 
            'latest.cost.attendance.academic_year',
            'latest.completion.completion_rate_4yr_100nt',
            'latest.earnings.10_yrs_after_entry.median',
            'latest.aid.median_debt.graduates.overall'
        ]
        
        all_schools = []
        page = 0
        per_page = 100
        
        while len(all_schools) < max_schools:
            params = {
                'api_key': self.api_key,
                'fields': ','.join(fields),
                'per_page': per_page,
                'page': page,
                # Better filtering - only get schools with actual data
                'school.degrees_awarded.predominant': '3',  # Bachelor's degree
                'school.operating': '1'  # Currently operating
            }
            
            print(f"Fetching page {page+1}... (collected {len(all_schools)} schools)")
            
            try:
                response = requests.get(self.base_url, params=params)
                response.raise_for_status()
                
                data = response.json()
                schools = data.get('results', [])
                
                if not schools:  # No more data
                    break
                    
                all_schools.extend(schools)
                page += 1
                
                # Rate limiting - be nice to the API
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Error on page {page}: {e}")
                break
                
        print(f"Collected {len(all_schools)} schools total")
        return self.process_school_data(all_schools)
    
    def process_school_data(self, schools):
        """Convert API data to clean DataFrame with proper null handling"""
        processed_data = []
        
        for school in schools:
            try:
                # Since API returns dotted field names, access them directly
                row = {
                    'school_id': school.get('id'),
                    'name': school.get('school.name'),
                    'state': school.get('school.state'),
                    'city': school.get('school.city'),
                    'ownership': school.get('school.ownership'),
                    'student_size': school.get('latest.student.size'),
                    'admission_rate': school.get('latest.admissions.admission_rate.overall'),
                    'sat_average': school.get('latest.admissions.sat_scores.average.overall'),
                    'act_midpoint': school.get('latest.admissions.act_scores.midpoint.cumulative'),
                    'tuition_in_state': school.get('latest.cost.tuition.in_state'),
                    'tuition_out_state': school.get('latest.cost.tuition.out_of_state'),
                    'total_cost': school.get('latest.cost.attendance.academic_year'),
                    'completion_rate_4yr': school.get('latest.completion.completion_rate_4yr_100nt'),
                    'median_earnings_10yr': school.get('latest.earnings.10_yrs_after_entry.median'),
                    'median_debt': school.get('latest.aid.median_debt.graduates.overall')
                }
                
                processed_data.append(row)
                
            except Exception as e:
                print(f"Error processing school: {e}")
                continue
        
        df = pd.DataFrame(processed_data)
        
        # Basic data quality filtering
        initial_count = len(df)
        
        # Remove rows where critical fields are missing
        df = df.dropna(subset=['name', 'state'])
        
        # Convert numeric fields properly, handling None values
        numeric_cols = ['student_size', 'admission_rate', 'sat_average', 'act_midpoint', 
                       'tuition_in_state', 'tuition_out_state', 'total_cost', 
                       'completion_rate_4yr', 'median_earnings_10yr', 'median_debt']
        
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Show some data quality info
        print(f"Processed {len(df)} schools into DataFrame (removed {initial_count - len(df)} with missing name/state)")
        print(f"Schools with earnings data: {df['median_earnings_10yr'].notna().sum()}")
        print(f"Schools with admission rate: {df['admission_rate'].notna().sum()}")
        
        return df
    
    def save_data(self, df, filename='college_data_raw.csv'):
        """Save data to file"""
        filepath = f'data/raw/{filename}'
        df.to_csv(filepath, index=False)
        print(f"Data saved to {filepath}")
        return filepath

def main():
    collector = CollegeScorecardCollector()
    
    print("Starting College Scorecard data collection...")
    df = collector.collect_institution_data(max_schools=1000)
    
    print(f"\nData summary:")
    print(f"Shape: {df.shape}")
    print(f"\nSample of collected data:")
    print(df.head())
    
    print(f"\nMissing data by field:")
    missing_data = df.isnull().sum()
    for col, missing in missing_data.items():
        if missing > 0:
            print(f"  {col}: {missing} missing ({missing/len(df)*100:.1f}%)")
    
    # Save raw data
    collector.save_data(df)
    
    print("\nData collection complete!")

if __name__ == "__main__":
    main()
