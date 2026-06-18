import streamlit as st
import pandas as pd
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI

# १. UI आणि स्क्रीनची रचना (Page Configuration)
st.set_page_config(page_title="Universal AI Assistant", layout="wide", initial_sidebar_state="expanded")

st.title("🤖 Universal AI Document & Data Assistant")
st.write("कोणतीही PDF (Resume), CSV किंवा Excel फाईल अपलोड करा आणि तिच्याबद्दल स्मार्ट प्रश्न विचारा!")
st.markdown("---")

# तुमची सुरक्षित API Key
# API Key - Streamlit secrets चा वापर करून सुरक्षित ठेवली आहे
api_key = st.secrets["GEMINI_API_KEY"]
if api_key:
    # डाव्या बाजूला फाईल अपलोड करण्यासाठी सिस्टीम (Sidebar)
    with st.sidebar:
        st.header("📁 Upload Section")
        uploaded_file = st.file_uploader("तुमची फाईल निवडा (PDF, CSV, Excel)", type=["pdf", "csv", "xlsx", "xls"])
        st.write("---")
        st.info("💡 **टीप:** जर तुम्ही **Resume** अपलोड केला असेल, तर खाली चॅटमध्ये 'Check ATS Score' किंवा 'Review my resume' असे विचारू शकता.")

    # २. फाईल अपलोड झाल्यावर तिची प्रोसेस सुरू करणे
    if uploaded_file is not None:
        file_ext = uploaded_file.name.split('.')[-1].lower()
        context_text = ""
        
        # ए) जर PDF फाईल असेल
        if file_ext == "pdf":
            st.success(f"Successfully Loaded PDF: {uploaded_file.name}")
            reader = PdfReader(uploaded_file)
            for page in reader.pages:
                context_text += page.extract_text() or ""
            
        # बी) जर CSV किंवा Excel डेटासेट असेल
        elif file_ext in ["csv", "xlsx", "xls"]:
            st.success(f"Successfully Loaded Dataset: {uploaded_file.name}")
            if file_ext == "csv":
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            # स्क्रीनवर डेटाची पहिली १० ओळी दाखवणे (Preview)
            st.write("📊 **Dataset Preview (डेटाची पहिली १० ओळी):**")
            st.dataframe(df.head(10))
            
            # डेटाला एआयसाठी टेक्स्ट स्वरूपात बदलणे
            context_text = f"This is a dataset/spreadsheet. Columns are: {list(df.columns)}. \nData Rows:\n" + df.to_string(index=False, max_rows=40)

        # ३. टेक्स्ट खूप मोठे असल्यास त्याचे तुकडे करणे (Chunking)
        splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=300)
        chunks = splitter.split_text(context_text)

        # ४. चॅट इनपुट बॉक्स (खाली टाईप करण्यासाठी)
        user_input = st.chat_input("तुमच्या फाईल/डेटासेटबद्दल इथे काहीही विचारा (Ask in English or Marathi)...")

        if user_input:
            # स्क्रीनवर युझरचा प्रश्न दाखवणे
            with st.chat_message("user"):
                st.write(user_input)

            with st.spinner("AI विचार करत आहे आणि उत्तर शोधत आहे..."):
                # संबंधित डेटा एकत्र करणे
                relevant_context = "\n".join(chunks[:2]) 
                
                # LLM मॉडेल कॉल करणे
                llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)
                
                # एआयला 'स्मार्ट' बनवण्यासाठी मास्टर प्रॉम्ट (Master Prompt)
                master_prompt = f"""
                You are an advanced, multi-talented AI Assistant, Technical Interviewer, and Data Analyst.
                Your task is to analyze the given context (which can be a general PDF, a candidate's Resume, or a CSV/Excel dataset) and answer the user's question perfectly.
                
                CRITICAL INSTRUCTIONS:
                1. IF THE FILE IS A RESUME: Act as an expert IT Recruiter and ATS Specialist. If asked about ATS, score, or review, give a detailed percentage score out of 100, suggest missing tech skills/keywords, and guide them as a fresher.
                2. IF THE FILE IS A DATASET (CSV/Excel): Act as a professional Data Analyst. Explain trends, column meanings, or summarize statistics based on the rows provided.
                3. LANGUAGE PROTOCOL: Reply in the exact same language or style the user used. If the user asks in Marathi, give a beautiful, structured Marathi response. If in English, reply professionally in English.

                Context Data:
                {relevant_context}

                User Question: {user_input}
                Answer:
                """
                
                response = llm.invoke(master_prompt)
                
                # स्क्रीनवर एआयचे उत्तर दाखवणे
                with st.chat_message("assistant"):
                    st.write(response.content)

else:
    st.error("कृपया वैध Gemini API Key सेट करा.")