import streamlit as st
import sqlite3
from sql_generator import SQLGenerator
import os

# 2. Define the database schema
schema = """
CREATE TABLE students (
  student_id INT PRIMARY KEY,
  name VARCHAR(100),
  major VARCHAR(50),
  gpa FLOAT
);

CREATE TABLE courses (
  course_id INT PRIMARY KEY,
  course_name VARCHAR(100),
  department VARCHAR(50)
);

CREATE TABLE enrollments (
  student_id INT,
  course_id INT,
  grade CHAR(1),
  FOREIGN KEY (student_id) REFERENCES students(student_id),
  FOREIGN KEY (course_id) REFERENCES courses(course_id)
);
"""

# 3. Set up the Streamlit app title and introductory text
st.title("NLU Text-to-SQL Chatbot")
st.write("Ask me questions about the database schema below, and I will generate and execute the SQL query.")

st.header("Database Schema")
st.code(schema, language="sql")

# 4. Load the SQLGenerator model
@st.cache_resource
def load_sql_generator():
    """Loads the SQLGenerator model."""
    # Assuming the model was saved in a directory named 'model'
    if not os.path.exists("model"):
         st.error("Model directory 'model' not found. Please ensure the model was saved correctly.")
         return None
    return SQLGenerator(save_dir="model")

sql_generator = load_sql_generator()

if sql_generator is None:
    st.stop() # Stop the app if the model didn't load

# 5. Set up an in-memory SQLite database and populate it
@st.cache_resource
def setup_database(schema_str):
    """Sets up and populates the in-memory SQLite database."""
    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()
    cursor.executescript(schema_str)
    cursor.executescript("""
    INSERT INTO students (student_id, name, major, gpa) VALUES
    (1, 'Alice', 'Computer Science', 3.8),
    (2, 'Bob', 'Physics', 3.5),
    (3, 'Charlie', 'Math', 3.9);

    INSERT INTO courses (course_id, course_name, department) VALUES
    (101, 'Intro to CS', 'Computer Science'),
    (102, 'Calculus I', 'Math'),
    (103, 'Quantum Physics', 'Physics');

    INSERT INTO enrollments (student_id, course_id, grade) VALUES
    (1, 101, 'A'),
    (1, 102, 'B'),
    (2, 103, 'A'),
    (3, 102, 'A');
    """)
    conn.commit()
    return conn

conn = setup_database(schema)

st.header("Ask a Question")

# 6. Create a Streamlit text input widget for the user question
user_question = st.text_input("Enter your question here:")

# 7. Create a Streamlit button to trigger the process
if st.button("Generate and Execute SQL"):
    if user_question:
        st.subheader("Generated SQL")
        # 8. Generate the SQL
        sql_query = sql_generator.generate_sql(user_question, schema)
        st.code(sql_query, language="sql")

        st.subheader("Query Results")
        # 9. Execute the SQL
        try:
            cursor = conn.cursor()
            cursor.execute(sql_query)
            results = cursor.fetchall()
            st.write(results)
        except sqlite3.Error as e:
            st.error(f"SQL Error: {e}")
    else:
        st.warning("Please enter a question.")
