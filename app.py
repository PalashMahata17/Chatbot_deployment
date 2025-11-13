import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import sqlite3

# 1. Define the database schema
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

# 2. Define create_prompt function
def create_prompt(question, schema_str):
    prompt = f"""
### Task
Generate a SQL query to answer [QUESTION]{question}[/QUESTION]

### Database Schema
The query will run on a database with the following schema:
{schema_str}

### Answer
Given the database schema, here is the SQL query that [QUESTION]{question}[/QUESTION]
[SQL]
"""
    return prompt

# 3. Define generate_sql function with caching for model and tokenizer
@st.cache_resource
def load_model_and_tokenizer():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    model_name = "defog/sqlcoder-7b-2"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    return tokenizer, model

def generate_sql(question, tokenizer, model):
    prompt = create_prompt(question, schema)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=False,
        num_beams=4,
        pad_token_id=tokenizer.eos_token_id # Set pad_token_id to eos_token_id
    )

    output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    try:
        sql_query = output.split("[SQL]")[1]
    except IndexError:
        return "Error: Could not parse the model's output. No [SQL] tag found."

    sql_query = sql_query.replace("```sql", "").replace("```", "").strip().strip(";") + ";"
    sql_query = sql_query.replace(" ilike ", " LIKE ")

    return sql_query

# 4. Set up SQLite in-memory database with caching
@st.cache_resource
def setup_database(schema_str):
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
    return conn, cursor

# Main Streamlit app logic
st.title("🤖 NLU Text-to-SQL Chatbot")
st.write("Ask me questions about the database. I'll generate and run SQL queries for you!")

# Load model and tokenizer once
with st.spinner('Loading model and tokenizer... This may take a few minutes.'):
    tokenizer, model = load_model_and_tokenizer()
st.success('Model and tokenizer loaded!')

# Setup database once
with st.spinner('Setting up in-memory database and populating data...'):
    conn, cursor = setup_database(schema)
st.success('Database ready!')

user_question = st.text_input("Your question:")

if user_question:
    st.markdown("### Generated SQL")
    try:
        sql_query = generate_sql(user_question, tokenizer, model)
        st.code(sql_query, language='sql')

        st.markdown("### Query Results")
        try:
            cursor.execute(sql_query)
            results = cursor.fetchall()
            if results:
                st.write(results)
            else:
                st.info("No results found.")
        except Exception as e:
            st.error(f"SQL Execution Error: {e}")

    except Exception as e:
        st.error(f"SQL Generation Error: {e}")
