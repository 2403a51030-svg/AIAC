import pandas as pd
import re

# Social media data cleaning pipeline:
# - Load CSV
# - Handle missing values for likes/shares
# - Clean post text (HTML/URLs/punct/stopwords)
# - Extract time features (hour/weekday)
# - Detect and remove spam posts
# - Remove duplicate posts
# - Save cleaned dataset

# Read CSV
path = r"C:\Users\savin\OneDrive\Desktop\AI Ass\LAB 17\social_media.csv"
df = pd.read_csv(path)

# Handle missing values in likes and shares:
# - Coerce to numeric (non-numeric -> NaN)
# - Fill NaN with column median
# - Clip negatives to 0
# - Cast to integers
print("Missing before:", df[['likes','shares']].isna().sum().to_dict())
df['likes'] = pd.to_numeric(df['likes'], errors='coerce')
df['shares'] = pd.to_numeric(df['shares'], errors='coerce')
df['likes'] = df['likes'].fillna(df['likes'].median()).clip(lower=0).astype(int)
df['shares'] = df['shares'].fillna(df['shares'].median()).clip(lower=0).astype(int)
print("Missing after:", df[['likes','shares']].isna().sum().to_dict())

# Minimal text cleaning: remove HTML/URLs/punctuation; lowercase; drop stopwords
STOPWORDS = {
    'the','is','at','which','on','and','a','an','to','in','for','of','with','by','from',
    'it','this','that','are','was','were','be','been','have','has','had','do','does','did',
    'will','would','could','should','may','might','must','can','shall','i','you','he','she',
    'we','they','me','him','her','us','them','my','your','his','our','their'
}

def clean_text(text: str) -> str:
    # Returns a normalized, token-filtered version of post_text
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'http[s]?://\S+', '', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = [w for w in text.split() if w not in STOPWORDS and len(w) > 1]
    return ' '.join(words)

df['post_text_clean'] = df['post_text'].apply(clean_text)

# Convert timestamp to datetime and extract time features
df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
df['hour'] = df['timestamp'].dt.hour
df['weekday'] = df['timestamp'].dt.day_name()

# Detect spam/duplicate posts:
# - Spam: very short, too repetitive, too long, or excessive symbols
# - Duplicates: identical cleaned text; keep first occurrence
def is_spam(text: str) -> bool:
    if pd.isna(text) or len(str(text)) < 5:
        return True
    text_l = str(text).lower()
    words = text_l.split()
    if len(words) and (len(set(words)) / len(words)) < 0.3:
        return True
    if len(text_l) > 500:
        return True
    if re.search(r'[^a-z0-9\s]{10,}', text_l):
        return True
    return False

df['is_spam'] = df['post_text_clean'].apply(is_spam)
spam_count = int(df['is_spam'].sum())
df = df[~df['is_spam']].drop(columns=['is_spam']).reset_index(drop=True)

# Remove duplicate posts based on cleaned text (keep first occurrence)
duplicate_mask = df.duplicated(subset=['post_text_clean'], keep='first')
duplicate_count = int(duplicate_mask.sum())
df = df[~duplicate_mask].reset_index(drop=True)

# Preview dataset after removing spam/duplicates
print({'spam_removed': spam_count, 'duplicates_removed': duplicate_count, 'rows_remaining': len(df)})
print(df[['post_text', 'post_text_clean', 'timestamp', 'hour', 'weekday']].head())

# Save cleaned dataset to CSV in current folder
output_file = 'social_media_cleaned.csv'
df.to_csv(output_file, index=False)
print({'csv_created': output_file, 'rows': len(df), 'cols': len(df.columns)})