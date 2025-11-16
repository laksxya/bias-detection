import streamlit as st
import os
import re
from together import run_pipeline  # Import the pipeline function

# --- Page Configuration ---
st.set_page_config(
    page_title="Automated News Reporter",
    layout="wide"
)

# --- Helper Function ---


def format_report(report_text: str) -> str:
    """
    Converts the raw report text into Markdown with clickable links.
    """
    # Regex to find a URL inside parentheses
    url_pattern = re.compile(r'\((https?://[^\s)]+)\)')

    def make_link(match):
        url = match.group(1)
        return f"([link]({url}))"

    # Replace URLs in parentheses with markdown links
    formatted_text = url_pattern.sub(make_link, report_text)

    # Ensure line breaks are rendered correctly in Markdown
    return formatted_text.replace('\n', '  \n')


# --- Main App UI ---
st.title("📝 Automated News Reporter")
st.write("Enter a topic to generate a neutral, factual news report from multiple sources.")

# --- Input Fields ---
with st.sidebar:
    st.header("Configuration")
    topic = st.text_input("Report Topic", "Bihar Elections 2025")

# --- Report Generation ---
if st.sidebar.button("Generate Report"):
    if not topic:
        st.error("Please enter a topic.")
    else:
        try:
            # Show a spinner and run the pipeline
            with st.spinner(f"Generating report for '{topic}'... This may take several minutes."):
                run_pipeline(topic=topic)

            # Read the final report
            report_path = os.path.join(os.path.dirname(__file__), "report.txt")
            if os.path.exists(report_path):
                with open(report_path, "r", encoding="utf-8") as f:
                    report_content = f.read()
                # Store the formatted report in session state to persist it
                st.session_state['report'] = format_report(report_content)
                st.success("Report generated successfully!")
            else:
                st.error(
                    "Pipeline finished, but the final report file was not found.")
                st.session_state['report'] = None

        except Exception as e:
            st.error(f"An error occurred during the pipeline: {e}")
            st.session_state['report'] = None

# --- Display Report ---
if 'report' in st.session_state and st.session_state['report']:
    st.markdown("---")
    st.header("Generated Report")
    # Use unsafe_allow_html to render the clickable links
    st.markdown(st.session_state['report'], unsafe_allow_html=True)
