# College ROI Predictor

Machine learning analysis of college return on investment and earnings using Random Forest regression on U.S. Department of Education data.

## Project Overview

This project investigates which factors most significantly influence both the earnings potential and return on investment of higher education decisions. Using Random Forest modeling on comprehensive federal education data, I analyze patterns across 1000+ U.S. universities to understand what drives successful educational investments.

## Research Questions

- Which school characteristics (cost, selectivity, size, location) best predict graduate earnings?
- How does major choice interact with institutional factors to influence ROI?
- Can we identify undervalued educational opportunities in the market?
- What trade-offs exist between educational cost and long-term financial outcomes?

## Technical Implementation

**Machine Learning:** Random Forest regression (scikit-learn)  
**Data Processing:** Python, Pandas, NumPy  
**Visualization:** Matplotlib, Seaborn  
**Data Source:** U.S. Department of Education College Scorecard API

## Project Structure
```
college-roi-predictor/
├── data/
│   ├── raw/                    # Raw data from API
│   └── processed/              # Cleaned data and models
├── Code/
│   ├── data_collection.py     # Collect data from College Scorecard API
│   ├── preprocessing.py       # Clean and prepare data
│   ├── model.py              # Earnings prediction model
│   └── roi_model.py          # ROI prediction model
├── requirements.txt
└── README.md
```

## Key Results

### Model Performance
**Earnings Prediction Model:**
- R-squared: 0.458 (45.8% accuracy)
- Mean Absolute Error: $8,491
- Overfitting ratio: 1.50x (acceptable range)

**ROI Prediction Model:**
- R-squared: 0.875 (87.5% accuracy) 
- Mean Absolute Error: 191.3 percentage points
- Demonstrates significantly better predictive power

### Feature Importance Analysis

**For Earnings Prediction:**
1. Completion rate (19.4%)
2. Out-of-state tuition (14.8%) 
3. SAT scores (13.8%)
4. Student size (8.9%)

**For ROI Prediction:**
1. Total cost (27.1%)
2. In-state tuition (17.6%)
3. School ownership type (14.5%)
4. Tuition ratio (11.8%)

## Business Insights

**Cost vs. Prestige:** Analysis reveals that educational costs have significantly more predictive power for ROI than traditional prestige metrics, suggesting affordable institutions in high-salary markets offer superior value propositions.

**Specialized Programs:** The models consistently underestimate earnings for specialized professional programs (maritime academies, pharmacy schools, health sciences), indicating these fields provide unique value not captured by standard institutional metrics.

**Geographic Factors:** Schools in expensive metropolitan areas with lower tuition costs demonstrate exceptional ROI performance, highlighting the importance of geographic arbitrage in educational investment decisions.

**Program Focus:** Institutions with focused career outcomes in technical fields consistently outperform predictions, suggesting program specialization matters more than broad institutional rankings.

## Model Limitations

- Social science predictions inherently limited by unmeasured factors (student motivation, networking, career choices)
- Economic conditions change over time, affecting long-term predictions
- Some institutional advantages (alumni networks, location benefits) difficult to quantify
- Sample limited to schools reporting complete federal data

## Installation and Usage

### Prerequisites
- Python 3.9+
- College Scorecard API key (free from api.data.gov)

### Setup
```bash
git clone https://github.com/yourusername/college-roi-predictor.git
cd college-roi-predictor
pip install -r requirements.txt
echo "COLLEGE_SCORECARD_API_KEY=your_key_here" > .env
```

### Execution
```bash
python Code/data_collection.py    # Collect data from 1000+ schools
python Code/preprocessing.py      # Clean and prepare data
python Code/model.py             # Build earnings prediction model  
python Code/roi_model.py         # Build ROI prediction model
```

## Data Sources

U.S. Department of Education College Scorecard API providing institution-level data on costs, admissions, completion rates, and 10-year post-graduation median earnings derived from IRS records.

## License

MIT License - see LICENSE file for details.

## Author

Liam Dondisch  
Data Science Student, Northeastern University

---

This project demonstrates machine learning model development, statistical analysis, and business insight generation for educational investment decision-making.
