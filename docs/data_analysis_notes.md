# Data Analysis Notes

## Initial Data Exploration

### Training Data Structure
- **Total Records**: 16,078 rows (including header)
- **Columns**: 14 features
- **Date Range**: Single day - 2025-10-20
- **Time Range**: 06:00:45 to 16:30+ (approximately 10+ hours)

### Key Observations

#### 1. Video Segments
- Each row represents ~1 minute of video
- Video naming format: `{camera}/{camera}_YYYY-MM-DD-HH-MM-SS.mp4`
- Example: `normanniles1/normanniles1_2025-10-20-06-00-45.mp4`

#### 2. Camera Views
- 4 camera positions: Norman Niles #1, #2, #3, #4
- Each camera captures entrance and exit of roundabout

#### 3. Congestion Patterns (from sample)
- Most early morning segments show "free flowing"
- Congestion increases during peak hours
- Both entrance and exit ratings can differ

#### 4. Turn Signal Usage
- Values: none, low, medium, high
- Most segments show "none" - aligns with challenge description
- This could be a key feature for congestion prediction

#### 5. Time Segments
- Sequential IDs starting from 0
- Continuous time series data
- Important for temporal modeling

### Test Data Structure
- Contains `cycle_phase` column with values like "test_input_15"
- Indicates 15-minute input windows
- Same structure as training data but without target labels

## Data Quality Checks Needed
- [ ] Check for missing values
- [ ] Verify video file availability
- [ ] Analyze class distribution
- [ ] Check for temporal gaps
- [ ] Validate timestamp consistency

## Feature Engineering Ideas

### Temporal Features
- Hour of day
- Day of week
- Rush hour indicator
- Time since last congestion
- Rolling averages

### Video-Based Features
- Vehicle count (entry/exit)
- Vehicle speed estimation
- Queue length
- Vehicle types
- Turn signal compliance rate

### Camera-Specific Features
- Camera location encoding
- Historical congestion at location
- Interaction between cameras

### Contextual Features
- Weather (if available)
- Special events
- Seasonal patterns

## Next Steps
1. Load full dataset and perform EDA
2. Analyze class distribution
3. Explore temporal patterns
4. Design feature extraction pipeline
5. Create baseline model
