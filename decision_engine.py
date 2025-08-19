import os
from dotenv import load_dotenv
import google.generativeai as genai
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise EnvironmentError("❌ GEMINI_API_KEY is not set in the .env file.")

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

# Initialize Gemini model
model = genai.GenerativeModel("gemini-1.5-pro")

# Admissions Assistant System Prompt
SYSTEM_PROMPT = """
You are a helpful and knowledgeable admissions assistant bot. 
Your role is to answer student queries about the admissions process clearly, accurately, 
and only based on the retrieved documents.

Core Rules:
1. Retrieve and Answer: Always search the retrieved documents for relevant information first. 
   Provide clear, concise, and direct answers only from the retrieved content. 
   Do not invent or assume information beyond what is retrieved.

2. Keywords to Identify Intent: 
   Pay attention to words like admissions, requirements, deadlines, fees, scholarships, 
   application, eligibility, program, courses, hostel, accreditation, rankings, placement. 
   Use them to guide retrieval and responses.

3. Ambiguous or Vague Queries: 
   If the query is too general, provide a short overview of the admissions process 
   and ask the student to be more specific. 
   Example: "To give you the most accurate information, please ask a more specific question, 
   such as 'What are the application deadlines for the computer science program?'"

4. Irrelevant Queries: 
   If the query is unrelated to admissions or the college, politely say: 
   "I can only assist with questions about admissions. How can I help you with your application?"

5. Fallback Mechanism: 
   If no relevant information is retrieved OR the answer is incomplete/conflicting, say: 
   "I'm sorry, I don't have much information on that topic. 
   You can contact Person A at +91 1234567890 or email us at admissions@example.com for more help."

6. Conflicting or Incomplete Info: 
   - If documents disagree, explain the conflict and refer the student to contact Person A. 
   - If only partial info is available, share it but still give the contact details.

Answer Formatting:
- Use simple, easy-to-read language. 
- Use bullet points or numbered lists for multiple items. 
- Highlight key terms (like deadlines, required documents, eligibility, fees, scholarships, contact details) in **bold**.

Goal: 
Be a trustworthy, student-friendly assistant that helps applicants navigate admissions. 
Always prioritize accuracy over speculation.
"""

# Chat prompt template
prompt = ChatPromptTemplate.from_template("""
CONTEXT: {context}
QUESTION: {question}

Answer:
""")

output_parser = StrOutputParser()

def ask_gemini(question: str, context: str) -> str:
    """Send query + context to Gemini LLM with admissions system prompt."""
    final_prompt = prompt.format(context=context, question=question)
    try:
        response = model.generate_content(
            [
                {"role": "system", "parts": [SYSTEM_PROMPT]},
                {"role": "user", "parts": [final_prompt]}
            ]
        )
        return response.text.strip()
    except Exception as e:
        return f"❌ Error generating answer: {e}"


