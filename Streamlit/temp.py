import streamlit as st
from transformers import T5Tokenizer
from transformers import T5ForConditionalGeneration


@st.cache_resource
def load_model():
    return T5ForConditionalGeneration.from_pretrained("kronos25/Temporal_Chatbot")
model = load_model()


@st.cache_resource
def load_tokenizer():
    return T5Tokenizer.from_pretrained("kronos25/Temporal_Chatbot")
tokenizer = load_tokenizer()

st.title("Temporal Chatbot")
st.subheader("Created by Pranav Rao & Rachit Jain")
st.text("This chatbot has the capability to understand temporal common sense")

if 'messages' not in st.session_state:
    st.session_state.messages = []
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What's up?"):
    st.markdown(prompt)
    st.session_state.messages.append({"role":"user", "content":prompt})

    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=100)

    # Decode the output tokens to text
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # response =

    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})

def reset_conversation():
  st.session_state.messages = []
st.button('Reset Chat', on_click=reset_conversation)
