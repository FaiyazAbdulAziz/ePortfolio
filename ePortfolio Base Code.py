import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Simulated Courses Data
courses = pd.DataFrame({
    'course_id': range(1, 11),
    'course_name': ['Intro to Programming', 'Advanced Machine Learning', 'Business Analytics', 'Intro to Physics', 'Advanced Physics', 'Market Analysis', 'Software Development', 'Entrepreneurship', 'Environmental Science', 'Data Structures'],
    'major': ['Computer Science', 'Computer Science', 'Business', 'Physics', 'Physics', 'Business', 'Computer Science', 'Business', 'Environmental Science', 'Computer Science'],
    'year': [1, 4, 2, 1, 4, 3, 2, 4, 1, 2],
    'tags': ['programming, coding', 'machine learning, AI, deep learning', 'analytics, business', 'physics, basics', 'physics, advanced', 'market, business', 'coding, software development', 'business, startup', 'environment, science', 'data structures, coding']
})

# Simulated Student Profile
student_profile = {
    'major': 'Computer Science',
    'year_of_study': 2,
    'interests': 'coding, software development, AI',
    'aspirations': 'software developer'
}

# Simulated Market Trends
market_trends = ['AI', 'machine learning', 'software development', 'data analysis']


# Vectorize course tags
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(courses['tags'])

# Combine student's interests and aspirations to create a query vector
student_query = ', '.join([student_profile['interests'], student_profile['aspirations']])
student_query_vec = tfidf_vectorizer.transform([student_query])

# Calculate similarity scores
cosine_similarities = linear_kernel(student_query_vec, tfidf_matrix).flatten()

# Get top 5 course recommendations based on cosine similarity
top_course_indices = cosine_similarities.argsort()[-5:][::-1]
recommended_courses = courses.iloc[top_course_indices]

# Filter by major and year of study, ensuring courses align with market trends
final_recommendations = recommended_courses[
    (recommended_courses['major'] == student_profile['major']) &
    (recommended_courses['year'] <= student_profile['year_of_study']) &
    (recommended_courses['tags'].apply(lambda tags: any(trend in tags for trend in market_trends)))
]

print(final_recommendations[['course_id', 'course_name']])

# Re-import linear_kernel for cosine similarity calculation
from sklearn.metrics.pairwise import linear_kernel

# Recalculate similarity scores with the corrected setup
cosine_similarities_example = linear_kernel(student_query_vec_example, tfidf_matrix_courses).flatten()
top_course_indices_example = cosine_similarities_example.argsort()[-5:][::-1]
recommended_courses_example = course_data_relevant.iloc[top_course_indices_example]

recommended_courses_example[['course_id', 'course_name']]