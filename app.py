import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
# provides python bindings for GGML models


# function to get response from llama2 model

def getLlamaResponse(input_text, no_words, blog_style):
    # callin the llama2 model
    llm = CTransformers(model = r"E:\_Projects\genai\blog_gen\models\llama-2-7b-chat.Q8_0.gguf",
                        model_type = "llama",
                        config = {'max_new_tokens':256,
                                  'temperature':0.01})
    
    # prompt template
    template = """
    Write a blog for {blog_style} job profile for the topic {input_text} within {no_words} words.
    """
    prompt = PromptTemplate(input_variables = ["blog_style", "input_text", 'no_words'],template=template) 
    
    # generate response from llama model
    response = llm(prompt.format(blog_style = blog_style, input_text =input_text, no_words = no_words))
    print(response)
    return response
    
    
st.set_page_config(
    page_title = "Generate Blogs",
    page_icon = '❤️',
    layout="centered",
    initial_sidebar_state='collapsed'
)

st.header("generate blogs")

input_text = st.text_input("enter the blog topic")


# more columns for additional 2 fields
col1, col2 = st.columns([5,5])

with col1:
    no_words = st.text_input("no of words")
with col2:
    blog_style = st.selectbox("writing the blog for", 
                              ('Researchers', 'Data Scientist', 'Common People', 'Enthusiasts', 'Subject Experts'), index = True)
submit = st.button("Generate")


# final response
if submit:
    st.write(getLlamaResponse(input_text, no_words, blog_style))
    
    
