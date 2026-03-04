import streamlit as st
import requests

st.set_page_config(
    page_title="SHL Assessment Recommender",
    page_icon="",
    layout="wide"
)

API_URL = "http://localhost:8000/recommend"

st.title(" SHL Assessment Recommendation Engine")

st.write(
"""
Enter a job description or hiring requirement and the system will recommend
relevant SHL assessments.
"""
)

query = st.text_area(
    "Enter Job Description / Hiring Requirement",
    height=200,
    placeholder="Example: Looking for a Python developer with SQL skills. Need an assessment within 60 minutes."
)

num_results = st.slider(
    "Number of recommendations",
    min_value=5,
    max_value=20,
    value=10
)

if st.button("Recommend Assessments"):

    if not query.strip():
        st.warning("Please enter a query.")
    else:

        with st.spinner("Finding best assessments..."):

            response = requests.post(
                API_URL,
                json={
                    "query": query,
                    "top_k": num_results
                }
            )

            if response.status_code != 200:
                st.error("API error")
            else:

                data = response.json()
                results = data.get("recommended_assessments", [])

                if not results:
                    st.error("No recommendations returned from API.")
                else:

                    st.success(f"Found {len(results)} assessments")

                    for i, r in enumerate(results, 1):

                        st.markdown(f"### {i}. {r.get('name','Assessment')}")

                        if r.get("description"):
                            st.write(r["description"])

                        if r.get("duration"):
                            st.write(f"⏱ Duration: {r['duration']} minutes")

                        if r.get("url"):
                            st.markdown(f"[Open Assessment]({r['url']})")

                        st.divider()

                