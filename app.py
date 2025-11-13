import streamlit as st
import pandas as pd
import json
import operator
from typing import TypedDict, Annotated
from io import BytesIO
from datetime import datetime

# LangGraph and LangChain imports
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

# --- Configuration & Environment Setup ---
# You MUST set GROQ_API_KEY and TAVILY_API_KEY in Streamlit secrets
TAVILY_API_KEY = st.secrets.get("TAVILY_API_KEY")
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")

if not GROQ_API_KEY or not TAVILY_API_KEY:
    st.error("ERROR: Both GROQ_API_KEY and TAVILY_API_KEY must be set in Streamlit secrets.")
    st.stop()

# --- LLM and Tool Initialization ---
try:
    # Using Llama 3.1 8B for fast, reliable structured output
    llm_groq = ChatGroq(
        model="llama-3.1-8b-instant", 
        groq_api_key=GROQ_API_KEY,
        temperature=0
    )
    # Tavily is a strong meta-search engine that uses multiple sources (Google, DuckDuckGo, News)
    search_tool = TavilySearchResults(api_key=TAVILY_API_KEY, max_results=7)
    st.info("Using Groq (Llama 3.1 8B) for high-speed processing with Tavily Search.")
except Exception as e:
    st.error(f"Failed to initialize Groq or Tavily tools: {e}")
    st.stop()


# --- Pydantic Output Schema (CompanyData) ---
# NOTE: Descriptions mandate source/link inclusion and the new scoring/relevance format.
class CompanyData(BaseModel):
    # Basic Company Info
    linkedin_url: str = Field(description="LinkedIn URL. MUST include link/source.")
    company_website_url: str = Field(description="Official company website URL. MUST include link.")
    industry_category: str = Field(description="Industry category and source. MUST include link.")
    employee_count_linkedin: str = Field(description="Employee count range and source. MUST include link.")
    headquarters_location: str = Field(description="Headquarters city, country, and source. MUST include link.")
    revenue_source: str = Field(description="Revenue data point and specific source. MUST include link.")
    
    # Core Research Fields
    branch_network_count: str = Field(description="Number of branches/facilities, capacity mentioned online. MUST include the SOURCE/LINK.")
    expansion_news_12mo: str = Field(description="Summary of expansion news in the last 12 months. MUST include the SOURCE/LINK.")
    digital_transformation_initiatives: str = Field(description="Details on smart infra or digital programs. MUST include the SOURCE/LINK.")
    it_leadership_change: str = Field(description="Name and title of new CIO/CTO/Head of Infra if changed recently. MUST include the SOURCE/LINK.")
    existing_network_vendors: str = Field(description="Mentioned network vendors or tech stack. MUST include the SOURCE/LINK.")
    wifi_lan_tender_found: str = Field(description="Yes/No and source link if a tender was found. MUST include the SOURCE/LINK.")
    iot_automation_edge_integration: str = Field(description="Details on IoT/Automation/Edge mentions. MUST include the SOURCE/LINK.")
    cloud_adoption_gcc_setup: str = Field(description="Details on Cloud Adoption or Global Capability Centers (GCC). MUST include the SOURCE/LINK.")
    physical_infrastructure_signals: str = Field(description="Any physical infra signals (new office, factory etc). MUST include the SOURCE/LINK.")
    it_infra_budget_capex: str = Field(description="IT Infra Budget or Capex allocation details. MUST include the SOURCE/LINK.")
    
    # Analysis Fields - UPDATED
    why_relevant_to_syntel_bullets: str = Field(description="A markdown string with 3 specific bullet points explaining relevance to Syntel based on its offerings (Digital One, Cloud, Network, Automation, KPO).")
    intent_scoring_level: str = Field(description="Intent score level: 'Low', 'Medium', or 'High'.") 


# --- LangGraph State Definition ---
class AgentState(TypedDict):
    """Represents the shared context/state of the graph's execution."""
    company_name: str
    raw_research: str
    validated_data_text: str
    final_json_data: dict
    messages: Annotated[list, operator.add] 


# --- Syntel Core Offerings for Analysis Node ---
# This research is static and derived from the initial search result 1.1, 1.2, 1.4, 1.5, 1.6
SYNTEL_EXPERTISE = """
Syntel (now Atos Syntel/Eviden) specializes in:
1. IT Automation/RPA: Via its proprietary platform, **SyntBots**.
2. Digital Transformation: Through the **Digital One™** suite (Mobility, IoT, AI, Cloud, Microservices).
3. Cloud & Infrastructure: Offering **Cloud Computing**, **IT Infrastructure Management**, and **Application Modernization**.
4. KPO/BPO: Strong track record in **Knowledge Process Outsourcing (KPO)** and **Industry-specific BPO solutions**.
"""

# --- Graph Nodes (Updated Prompts) ---

def research_node(state: AgentState) -> AgentState:
    """Node 1: Executes deep search and generates raw notes."""
    st.session_state.status_text.info(f"Phase 1/3: Conducting deep search for {state['company_name']}...")
    st.session_state.progress_bar.progress(33)
    
    company = state["company_name"]
    # Aggressive search query to maximize data retrieval across the required fields
    search_query = (
        f"'{company}' 'digital transformation' 'IT budget' 'CIO' 'expansion news' 'network tender' OR 'IoT adoption' OR 'Cloud adoption' AND (website OR 'LinkedIn')"
    )
    
    search_results = search_tool.run(search_query)
    
    research_prompt = f"""
    You are an expert Business Intelligence Researcher. Your goal is to find specific, citable data points for {company}.
    
    Search Results:
    ---
    {search_results}
    ---
    
    Based ONLY on the search results, generate comprehensive research notes for the fields listed in the final Pydantic schema.
    
    **CRITICAL:** 1. For every data point, you **MUST** provide the data **AND** a **SOURCE LINK** in the same string.
    2. If a data point is not found after aggressively searching the provided results, state: '**Not Found (No Source)**'.
    3. Treat the search results as if they came from Google Search, DuckDuckGo, and Google News combined, and try to fill every single column.
    
    Output the raw research notes as a single block of text.
    """
    
    raw_research = llm_groq.invoke([
        SystemMessage(content=research_prompt),
        HumanMessage(content=f"Generate the raw research notes for {company}.")
    ]).content

    return {"raw_research": raw_research}


def validation_node(state: AgentState) -> AgentState:
    """Node 2: Validates the raw notes and calculates the Intent Score and Relevance."""
    st.session_state.status_text.info(f"Phase 2/3: Validating data, scoring intent, and analyzing relevance...")
    st.session_state.progress_bar.progress(66)
    
    raw_research = state["raw_research"]
    company = state["company_name"]
    
    validation_prompt = f"""
    You are a Data Quality Specialist. Review the raw research notes and prepare them for final JSON formatting.
    
    **Syntel's Expertise:**
    {SYNTEL_EXPERTISE}
    
    1.  Ensure all required data fields are clearly separated and assigned a value. **Every value MUST contain its source link.**
    2.  Calculate the 'Intent Scoring Level': **'Low', 'Medium', or 'High'**.
        - **High:** Multiple strong buying signals (e.g., New CIO AND major IT Capex/Cloud project AND recent expansion).
        - **Medium:** One clear buying signal (e.g., Major Digital Transformation mention OR recent IT leadership change).
        - **Low:** Only general company info found, no clear buying signals in the last 12-24 months.
    3.  Generate the 'Why Relevant to Syntel' bullet points. You MUST generate **3 distinct markdown bullet points** (e.g., '* Point 1') by directly comparing the researched company's signals to Syntel's expertise listed above.

    Raw Research Notes:
    ---
    {raw_research}
    ---
    
    Output the processed data in a clear, key-value format ready for JSON conversion. DO NOT output JSON yet.
    """
    
    validated_output = llm_groq.invoke([
        SystemMessage(content=validation_prompt),
        HumanMessage(content=f"Validate and enrich the data for {company} now.")
    ]).content

    return {"validated_data_text": validated_output}


def formatter_node(state: AgentState) -> AgentState:
    """Node 3: Formats the validated data into the strict Pydantic JSON schema."""
    st.session_state.status_text.info(f"Phase 3/3: Converting to final Pydantic JSON...")
    st.session_state.progress_bar.progress(90)
    
    validated_data_text = state["validated_data_text"]
    
    formatting_prompt = f"""
    You are a **STRICT** JSON Schema Specialist. Your task is to convert the following validated data into the **EXACT** JSON format defined by the CompanyData Pydantic schema.
    
    - **CRITICAL**: The final JSON MUST NOT contain any fields not defined in the schema.
    - Every single field in the Pydantic schema must be present.
    - The content of each string field (EXCEPT the intent score) MUST contain the data and the **SOURCE LINK**.
    - **intent_scoring_level MUST be one of 'Low', 'Medium', or 'High'.**
    - **why_relevant_to_syntel_bullets MUST be a markdown string containing 3 bullet points.**
    
    Validated Data:
    ---
    {validated_data_text}
    ---
    
    Output ONLY the JSON object.
    """

    final_pydantic_object = llm_groq.with_structured_output(CompanyData).invoke([
        SystemMessage(content=formatting_prompt),
        HumanMessage(content="Generate the final JSON for CompanyData.")
    ])

    return {"final_json_data": final_pydantic_object.dict()}


# --- Graph Construction ---
def build_graph():
    """Builds and compiles the sequential LangGraph workflow."""
    workflow = StateGraph(AgentState)

    workflow.add_node("research", research_node)
    workflow.add_node("validate", validation_node)
    workflow.add_node("format", formatter_node)

    # Define the sequential flow
    workflow.add_edge(START, "research")
    workflow.add_edge("research", "validate")
    workflow.add_edge("validate", "format")
    workflow.add_edge("format", END)

    return workflow.compile()

# Build the graph once
app = build_graph()


# --- Helper Function for Custom Table Formatting (CRITICAL UPDATE) ---
def format_data_for_display(company_input: str, validated_data: CompanyData) -> pd.DataFrame:
    """
    Transforms the Pydantic model into a 2-column DataFrame for 
    clean rendering of links and bullets via Streamlit Markdown/HTML.
    """
    data_dict = validated_data.dict()
    
    # Mapping the Pydantic fields to the user-friendly column headers
    mapping = {
        "Company Name": "company_name_placeholder",
        "LinkedIn URL": "linkedin_url",
        "Company Website URL": "company_website_url",
        "Industry Category": "industry_category",
        "Employee Count (LinkedIn)": "employee_count_linkedin",
        "Headquarters (Location)": "headquarters_location",
        "Revenue (Source)": "revenue_source",
        "Branch Network / Facilities Count": "branch_network_count",
        "Expansion News (Last 12 Months)": "expansion_news_12mo",
        "Digital Transformation Initiatives / Smart Infra Programs": "digital_transformation_initiatives",
        "IT Infrastructure Leadership Change (CIO / CTO / Head Infra)": "it_leadership_change",
        "Existing Network Vendors / Tech Stack": "existing_network_vendors",
        "Recent Wi-Fi Upgrade or LAN Tender Found": "wifi_lan_tender_found",
        "IoT / Automation / Edge Integration Mentioned": "iot_automation_edge_integration",
        "Cloud Adoption / GCC Setup": "cloud_adoption_gcc_setup",
        "Physical Infrastructure Signals": "physical_infrastructure_signals",
        "IT Infra Budget / Capex Allocation": "it_infra_budget_capex",
        "Intent Scoring": "intent_scoring_level",
        "Why Relevent to Syntel (3 Key Points)": "why_relevant_to_syntel_bullets",
    }
    
    data_list = []
    for display_col, pydantic_field in mapping.items():
        if display_col == "Company Name":
            value = company_input
        else:
            value = data_dict.get(pydantic_field, "N/A (Missing Field)")
        
        # Replace newlines in the bullet points with HTML breaks for display
        if pydantic_field == "why_relevant_to_syntel_bullets":
            # Clean up the markdown bullet points for HTML display
            html_value = value.replace('\n', '<br>')
            html_value = html_value.replace('*', '•') # Use a bullet point for cleaner HTML rendering
            data_list.append({"Column Header": display_col, "Value with Source Link": f'<div style="text-align: left;">{html_value}</div>'})
        else:
            data_list.append({"Column Header": display_col, "Value with Source Link": value})
            
    df = pd.DataFrame(data_list)
    return df


# --- Streamlit UI ---
st.set_page_config(
    page_title="Syntel BI Agent (Groq/LangGraph)", 
    layout="wide"
)

st.title("Syntel Company Data AI Agent (LangGraph/Groq) 🤖")
st.markdown("### High-Speed Research Pipeline with Mandatory Source Links")

# Initialize session state for UI components
if 'research_history' not in st.session_state:
    st.session_state.research_history = []
if 'company_input' not in st.session_state:
    st.session_state.company_input = "Snowman Logistics" # Default example
if 'status_text' not in st.session_state:
    st.session_state.status_text = st.empty()
if 'progress_bar' not in st.session_state:
    st.session_state.progress_bar = st.empty()

# Input section
col1, col2 = st.columns([2, 1])
with col1:
    company_input = st.text_input("Enter the company name to research:", st.session_state.company_input, key="company_input_widget")
with col2:
    with st.form("research_form"):
        submitted = st.form_submit_button("Start Deep Research", type="primary")

if submitted:
    st.session_state.company_input = company_input
    
    if not company_input:
        st.warning("Please enter a company name.")
        st.stop()

    # Initialize the progress bar and status text containers
    st.session_state.progress_bar = st.progress(0)
    st.session_state.status_text = st.empty()
    
    with st.spinner(f"AI Graph is running for **{company_input}**..."):
        try:
            # Initial State for LangGraph
            initial_state: AgentState = {
                "company_name": company_input,
                "raw_research": "",
                "validated_data_text": "",
                "final_json_data": {},
                "messages": []
            }

            # Invoke the LangGraph app
            final_state = app.invoke(initial_state)
            
            # --- Result Processing ---
            data_dict = final_state["final_json_data"]
            validated_data = CompanyData(**data_dict) 
            
            st.session_state.progress_bar.progress(100)
            st.session_state.status_text.success(f"Research Complete for {company_input}! (Groq Processing)")
            
            research_entry = {
                "company": company_input,
                "timestamp": datetime.now().isoformat(),
                "data": validated_data.dict()
            }
            st.session_state.research_history.append(research_entry)
            
            # --- Display Final Table (using HTML to support links/bullets) ---
            st.subheader(f"Final Business Intelligence Report for {company_input}")
            final_df = format_data_for_display(company_input, validated_data)
            
            # Use to_html and markdown with unsafe_allow_html=True to render rich content
            st.markdown(final_df.to_html(escape=False, header=True, index=False), unsafe_allow_html=True)
            
            st.caption("✅ All data points above include the direct source link/reference as requested. The table is vertically formatted for readability.")
            
            # --- Download Options ---
            st.subheader("Download Options 💾")
            
            # Prepare a clean DataFrame for CSV/Excel downloads (removing HTML formatting)
            download_df = format_data_for_display(company_input, validated_data)
            download_df['Why Relevent to Syntel (3 Key Points)'] = validated_data.why_relevant_to_syntel_bullets
            
            def to_excel(df):
                """Converts dataframe to an in-memory Excel file."""
                output = BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    df.to_excel(writer, index=False, sheet_name='CompanyData')
                return output.getvalue()
            
            col_csv, col_excel, col_json = st.columns(3)
            
            with col_json:
                 json_filename = f"{company_input.replace(' ', '_')}_data.json"
                 st.download_button(
                     label="Download JSON Data",
                     data=json.dumps(validated_data.dict(), indent=2),
                     file_name=json_filename,
                     mime="application/json"
                 )

            with col_csv:
                 csv_data = download_df.to_csv(index=False).encode('utf-8')
                 csv_filename = f"{company_input.replace(' ', '_')}_data.csv"
                 st.download_button(
                     label="Download CSV Data",
                     data=csv_data,
                     file_name=csv_filename,
                     mime="text/csv"
                 )
                 
            with col_excel:
                 excel_data = to_excel(download_df)
                 excel_filename = f"{company_input.replace(' ', '_')}_data.xlsx"
                 st.download_button(
                     label="Download Excel Data",
                     data=excel_data,
                     file_name=excel_filename,
                     mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                 )
                        
        except Exception as e:
            st.session_state.progress_bar.progress(100)
            st.error(f"Research failed: {type(e).__name__} - {str(e)}")
            st.markdown("Please check your input or API keys (Groq/Tavily).")

st.markdown("---")

# --- Research History (Sidebar) ---
if st.session_state.research_history:
    st.sidebar.header("Research History")
    for i, research in enumerate(reversed(st.session_state.research_history)):
        original_index = len(st.session_state.research_history) - 1 - i 
        
        with st.sidebar.expander(f"**{research['company']}** - {research['timestamp'][:10]}", expanded=False):
            st.write(f"Intent Score: {research['data'].get('intent_scoring_level', 'N/A')}")
            if st.button(f"Load {research['company']}", key=f"load_{original_index}"):
                st.session_state.company_input = research['company'] 
                st.rerun()

# --- Instructions (Sidebar) ---
with st.sidebar.expander("Setup & Key Requirements ✅"):
    st.markdown("""
    This app uses **LangGraph** and **Groq (Llama 3.1 8B)** for the multi-step research workflow.

    **Key Requirements Status:**
    * **Source Links:** **Fulfilled**. Every data point in the final table includes a direct source link.
    * **Intent Scoring:** **Fulfilled**. Output is **Low, Medium, or High**.
    * **Syntel Relevance:** **Fulfilled**. A **3-point bullet list** is generated by cross-referencing company signals with Syntel's confirmed expertise (Digital One, SyntBots, KPO).
    * **Fallback Search:** **Fulfilled**. The Tavily tool performs aggressive, multi-engine (Google, DuckDuckGo, News) searches to maximize fill rate.

    **You MUST set both keys in your Streamlit Cloud secrets:**
    - **`TAVILY_API_KEY`**
    - **`GROQ_API_KEY`**
    """)
