import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="NVIDIA AI Chat", page_icon="🤖")
st.title("🤖 NVIDIA AI Chat")

# Read API key from Streamlit secrets (safe)
try:
    api_key = st.secrets["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"]
except Exception:
    api_key = st.text_input("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:", type="password")

if not api_key:
    st.info("Please provide your NVIDIA API key to continue.")
    st.stop()

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=api_key
)

# Sidebar: model picker
MODELS = {
    "Nemotron 3.5 Lightning 30B (fast)": "nvidia/nemotron-3.5-lightning-30b-a3b",
    "Llama 3.1 8B Instruct": "meta/llama-3.1-8b-instruct",
    "Llama 3.1 70B Instruct": "meta/llama-3.1-70b-instruct",
    "Mistral 7B Instruct": "mistralai/mistral-7b-instruct-v0.3",
    "Qwen 2.5 72B Instruct": "qwen/qwen2.5-72b-instruct",
}

st.sidebar.header("Settings")
model_label = st.sidebar.selectbox("Choose a model", list(MODELS.keys()))
model_id = MODELS[model_label]

if st.sidebar.button("Clear conversation"):
    st.session_state.messages = []
    st.rerun()

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
prompt = st.chat_input("Ask something...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            extra = {}
            if "nemotron" in model_id.lower():
                extra = {
                    "chat_template_kwargs": {"enable_thinking": True},
                    "reasoning_budget": 16384
                }

            completion = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ],
                temperature=1,
                top_p=0.95,
                max_tokens=16384,
                extra_body=extra if extra else None,
                stream=True
            )

            placeholder = st.empty()
            full = ""
            for chunk in completion:
                if not chunk.choices:
                    continue
                content = chunk.choices[0].delta.content
                if content:
                    full += content
                    placeholder.markdown(full + "▌")
            placeholder.markdown(full)

            st.session_state.messages.append(
                {"role": "assistant", "content": full}
            )
        except Exception as e:
            st.error(f"Error: {e}")
