import streamlit as st
from streamlit.components.v1 import html as st_html
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import base64
import os

st.set_page_config(page_title="Multilingual Translation", page_icon="🌐", layout="centered")

# Load translation background image for white area (base64 so it works as CSS background)
_BG_IMG_B64 = None
_bg_path = os.path.join(os.path.dirname(__file__), "assets", "translation-bg.png")
if os.path.isfile(_bg_path):
    with open(_bg_path, "rb") as f:
        _BG_IMG_B64 = base64.b64encode(f.read()).decode()

# NLLB language codes (source/target)
LANG_CODES_NLLB = {
    "Hindi": "hin_Deva",
    "Kannada": "kan_Knda",
    "Tamil": "tam_Taml",
    "Telugu": "tel_Telu",
    "Marathi": "mar_Deva",
}
ENGLISH_CODE = "eng_Latn"

# Sample texts for quick try (English for En→Lang; add more for Lang→English if needed)
SAMPLE_TEXTS = {
    "— Select a sample —": "",
    "Greeting": "Hello, how are you? I hope you are doing well today.",
    "News": "The government announced a new initiative to improve public transport in the city.",
    "Culture": "The festival brings communities together with music, food, and colourful decorations.",
    "Short": "Thank you. Have a nice day.",
}

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
    if torch.cuda.is_available():
        model = model.to("cuda")
    return tokenizer, model

tokenizer, model = load_model()

# Optional: translation background image for white area
_bg_css = ""
if _BG_IMG_B64:
    _bg_css = f"""
    .stApp::before {{
        background-image: url(data:image/png;base64,{_BG_IMG_B64}),
            radial-gradient(circle at 20% 80%, rgba(0, 102, 204, 0.05) 0%, transparent 50%),
            radial-gradient(circle at 80% 20%, rgba(102, 255, 255, 0.06) 0%, transparent 50%);
        background-size: cover, auto, auto;
        background-position: center center, 0 0, 0 0;
        background-repeat: no-repeat, repeat, repeat;
        opacity: 0.88;
    }}
    """

st.markdown(
    f"""
    <style>
    /* Base typography & font enhancement */
    .stApp, .stApp * {{
        font-family: "Segoe UI", "SF Pro Display", system-ui, -apple-system, BlinkMacSystemFont, sans-serif !important;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
        text-rendering: optimizeLegibility;
    }}
    .main .block-container {{
        font-feature-settings: "kern" 1, "liga" 1;
    }}
    /* Hide Streamlit 3-dots menu, toolbar, and footer */
    #MainMenu, [data-testid="stToolbar"], footer {{
        visibility: hidden !important;
        display: none !important;
    }}
    /* Header bar: blue to cyan gradient */
    [data-testid="stDecoration"] {{
        background-image: linear-gradient(90deg, rgb(0, 102, 204), rgb(102, 255, 255));
    }}
    /* Page: full viewport, background covers entire page */
    .stApp {{
        min-height: 100vh;
        min-height: 100dvh;
        position: relative;
        background-color: #fafbff;
    }}
    @keyframes bgMove {{
        0%, 100% {{ background-position: center center, 0 0, 0 0; }}
        25% {{ background-position: 48% 52%, 0 0, 0 0; }}
        50% {{ background-position: 52% 48%, 0 0, 0 0; }}
        75% {{ background-position: 48% 48%, 0 0, 0 0; }}
    }}
    @keyframes gradientShift {{
        0%, 100% {{ opacity: 1; transform: scale(1); }}
        50% {{ opacity: 0.97; transform: scale(1.02); }}
    }}
    .stApp::before {{
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        height: 100dvh;
        min-height: 100%;
        z-index: 0;
        pointer-events: none;
        background-color: #fafbff;
        background-image: radial-gradient(circle at 20% 80%, rgba(0, 102, 204, 0.05) 0%, transparent 50%),
            radial-gradient(circle at 80% 20%, rgba(102, 255, 255, 0.06) 0%, transparent 50%);
        background-size: cover, auto, auto;
        background-position: center, 0 0, 0 0;
        background-repeat: no-repeat, repeat, repeat;
        animation: bgMove 25s ease-in-out infinite, gradientShift 15s ease-in-out infinite;
    }}
    {_bg_css}
    .main {{
        position: relative;
        z-index: 1;
    }}
    /* Main box: black container - enhanced shadow & border */
    .main .block-container {{
        padding: 2.5rem 2rem 3rem;
        max-width: 920px;
        background: linear-gradient(180deg, #1e1e1e 0%, #1a1a1a 100%);
        border-radius: 20px;
        box-shadow: 0 25px 50px -12px rgba(0,0,0,0.35), 0 0 0 1px rgba(255,255,255,0.06) inset;
        position: relative;
        transition: box-shadow 0.3s ease;
    }}
    .main .block-container:hover {{
        box-shadow: 0 30px 60px -15px rgba(0,0,0,0.4), 0 0 0 1px rgba(255,255,255,0.08) inset;
    }}
    /* Main heading: enhanced typography */
    .main-title {{
        font-size: 7.5rem;
        font-weight: 900;
        margin-bottom: 0.5rem;
        text-align: center;
        letter-spacing: -0.04em;
        line-height: 1.08;
        background: linear-gradient(135deg, #004499 0%, #0066CC 20%, #0088aa 45%, #006633 70%, #5522aa 100%);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        filter: contrast(1.1) brightness(0.98);
        text-rendering: optimizeLegibility;
    }}
    /* Form labels: enhanced font */
    .main .stRadio label, .main .stSelectbox label {{
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 1.5rem !important;
        letter-spacing: 0.03em;
        text-shadow: 0 1px 2px rgba(0,0,0,0.2);
    }}
    .main [data-testid="stVerticalBlock"] label, .main .stMarkdown label {{
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 1.5rem !important;
        letter-spacing: 0.03em;
        text-shadow: 0 1px 2px rgba(0,0,0,0.2);
    }}
    .main .stTextArea label {{
        color: #ffffff !important;
        font-weight: 800 !important;
        font-size: 1.85rem !important;
        letter-spacing: 0.03em;
        text-shadow: 0 1px 3px rgba(0,0,0,0.25);
    }}
    .main .stSubheader, .main h3 {{
        color: #ffffff !important;
        font-weight: 800 !important;
        font-size: 1.5rem !important;
        letter-spacing: 0.02em;
        text-shadow: 0 1px 2px rgba(0,0,0,0.2);
    }}
    .main [data-testid="stVerticalBlock"] p {{
        color: #e8e8e8 !important;
        font-size: 1.1rem !important;
        line-height: 1.5;
    }}
    /* Text areas: enhanced with transition */
    .stTextArea textarea {{
        font-size: 1.15rem;
        min-height: 320px;
        border-radius: 12px;
        background-color: #2d2d2d !important;
        color: #f0f0f0 !important;
        border: 1px solid #444;
        caret-color: #ffffff;
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }}
    .stTextArea textarea:focus {{
        border-color: #0066CC !important;
        box-shadow: 0 0 0 2px rgba(0, 102, 204, 0.25) !important;
        outline: none !important;
    }}
    .stTextArea textarea:disabled, .stTextArea textarea[disabled] {{
        background-color: #2d2d2d !important;
        color: #f5f5f5 !important;
        -webkit-text-fill-color: #f5f5f5 !important;
        opacity: 1 !important;
    }}
    .stTextArea textarea::placeholder {{
        color: #888;
    }}
    /* Buttons: smooth transitions */
    .stButton > button {{
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        padding: 0.65rem 1.6rem !important;
        border-radius: 12px !important;
        background: linear-gradient(180deg, #0066CC 0%, #0052a3 100%) !important;
        color: white !important;
        border: none !important;
        box-shadow: 0 4px 14px rgba(0, 102, 204, 0.35) !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease !important;
    }}
    .stButton > button:hover {{
        background: linear-gradient(180deg, #0077ee 0%, #0066CC 100%) !important;
        color: white !important;
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(0, 102, 204, 0.45) !important;
    }}
    .stDownloadButton > button {{
        background: #2a2a2a !important;
        color: #e8e8e8 !important;
        border: 1px solid #555 !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        transition: background 0.2s ease, transform 0.2s ease !important;
    }}
    .stDownloadButton > button:hover {{
        background: #3a3a3a !important;
        transform: translateY(-1px);
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<h1 class="main-title">Multilingual Translation</h1>', unsafe_allow_html=True)
direction = st.radio(
    "Direction",
    ["English → Language", "Language → English"],
    horizontal=True,
    label_visibility="visible",
)

if direction == "English → Language":
    target_lang = st.selectbox("Target language", list(LANG_CODES_NLLB.keys()))
    input_label = "Enter English text"
else:
    target_lang = st.selectbox("Source language", list(LANG_CODES_NLLB.keys()))
    input_label = f"Enter text in {target_lang}"

# Sample selector only for English → Language
if "input_text" not in st.session_state:
    st.session_state.input_text = ""
if "last_sample_choice" not in st.session_state:
    st.session_state.last_sample_choice = ""

if direction == "English → Language":
    sample_choice = st.selectbox("Load a sample", list(SAMPLE_TEXTS.keys()))
    if SAMPLE_TEXTS[sample_choice] and sample_choice != st.session_state.last_sample_choice:
        st.session_state.input_text = SAMPLE_TEXTS[sample_choice]
        st.session_state.last_sample_choice = sample_choice
    else:
        st.session_state.last_sample_choice = sample_choice
else:
    st.session_state.last_sample_choice = ""

input_text = st.text_area(input_label, height=320, placeholder="Type or paste text here...", key="input_text")

translate_clicked = st.button("Translate")

if "translation_output" not in st.session_state:
    st.session_state.translation_output = ""

if translate_clicked and input_text.strip():
    with st.spinner("Translating..."):
        try:
            if direction == "English → Language":
                tokenizer.src_lang = ENGLISH_CODE
                lang_code = LANG_CODES_NLLB[target_lang]
            else:
                tokenizer.src_lang = LANG_CODES_NLLB[target_lang]
                lang_code = ENGLISH_CODE
            # NLLB: get forced_bos_token_id for target language
            if hasattr(tokenizer, "lang_code_to_id") and lang_code in getattr(tokenizer, "lang_code_to_id", {}):
                forced_bos = tokenizer.lang_code_to_id[lang_code]
            else:
                # fallback: tokenizer may use language code as token
                tid = tokenizer.convert_tokens_to_ids(lang_code)
                forced_bos = tid[0] if isinstance(tid, list) and tid else (tid if isinstance(tid, int) else tokenizer.eos_token_id)

            inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
            if next(model.parameters()).is_cuda:
                inputs = {k: v.to("cuda") for k, v in inputs.items()}

            out_ids = model.generate(**inputs, forced_bos_token_id=forced_bos, max_length=256)
            st.session_state.translation_output = tokenizer.batch_decode(out_ids, skip_special_tokens=True)[0]
        except Exception as e:
            st.session_state.translation_output = ""
            st.error(str(e))
elif translate_clicked and not input_text.strip():
    st.warning("Please enter some text to translate.")

st.subheader("Output")
out = st.session_state.translation_output
# No key= so the box always shows current value from session_state
st.text_area("Translation output", value=out, height=320, disabled=True, label_visibility="collapsed")

col_copy, col_download, _ = st.columns([1, 1, 4])
with col_copy:
    # Copy button (uses JS to copy to clipboard)
    out_escaped = (out or "").replace("\\", "\\\\").replace("'", "\\'").replace("\n", "\\n").replace("\r", "\\r").replace("</script>", "<\\/script>")
    copy_html = f'''
    <button id="copyBtn" style="width:100%; padding: 0.5rem 1rem; font-size: 1rem; border-radius: 8px; border: 1px solid #0066CC; background: #0066CC; color: white; cursor: pointer;">
        Copy
    </button>
    <span id="copyMsg" style="margin-left:8px; color: #0066CC; font-size: 0.9rem;"></span>
    <script>
    (function() {{
        var btn = document.getElementById("copyBtn");
        var msg = document.getElementById("copyMsg");
        var text = "{out_escaped}";
        if (btn) {{
            btn.onclick = function() {{
                navigator.clipboard.writeText(text).then(function() {{
                    msg.textContent = "Copied!";
                    setTimeout(function() {{ msg.textContent = ""; }}, 2000);
                }});
            }};
        }}
    }})();
    </script>
    '''
    st_html(copy_html, height=50)
with col_download:
    st.download_button(
        "Download",
        data=out or "",
        file_name="translation.txt",
        mime="text/plain",
        use_container_width=True,
        disabled=not out,
    )
