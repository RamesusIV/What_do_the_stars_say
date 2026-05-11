import streamlit as st
from src.predict import predict_story

st.set_page_config(
    page_title="Constellation Mythology QA",
    page_icon="✨",
    layout="centered"
)

st.title("✨ Constellation Mythology Question Answering")
st.markdown(
    "Ask any question about the mythological stories behind constellations."
)

# Example questions
with st.expander("Example Questions"):
    st.write("- Who was Orion?")
    st.write("- What is the story behind Cassiopeia?")
    st.write("- Why was Andromeda placed in the sky?")
    st.write("- Tell me about the myth of Perseus.")

question = st.text_input(
    "Enter your question:",
    placeholder="Who was Orion?"
)

if st.button("Get Answer"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Searching mythology..."):
            try:
                answer = predict_story(question)
                st.success("Answer")
                st.write(answer)
            except Exception as e:
                st.error(f"Error: {e}")

st.markdown("---")
st.caption("Built with Streamlit, Sentence Transformers, FAISS, and DVC")