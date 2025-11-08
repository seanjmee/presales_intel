"""
Pre-Call Intelligence Agent - OutSystems Edition
Gradio-based interface for generating presales call prep briefs
Optimized for low-code and agentic AI use cases
"""

import gradio as gr
import anthropic
import os
from datetime import datetime
import markdown
from dotenv import load_dotenv
import time
from weasyprint import HTML
import tempfile

load_dotenv(override=True)
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY environment variable not set")
client = anthropic.Anthropic(api_key=api_key)
# Using Claude 3.5 Haiku - 10x cheaper, faster, and higher rate limits than Sonnet
model = "claude-3-5-haiku-20241022"


class OutSystemsIntelAgent:
    """
    AI agent for OutSystems presales - researches prospects with focus on
    low-code platforms, digital transformation, and agentic AI opportunities
    """
    
    def __init__(self):
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        self.client = anthropic.Anthropic(api_key=api_key)
        # Using Claude 3.5 Haiku - 10x cheaper, faster, and higher rate limits than Sonnet
        self.model = "claude-3-5-haiku-20241022"

    def _call_api_with_retry(self, create_fn, max_retries=2, initial_delay=5):
        """
        Call API with exponential backoff retry for rate limits

        Args:
            create_fn: Function that makes the API call
            max_retries: Maximum number of retry attempts (reduced to avoid compounding rate limit issues)
            initial_delay: Initial delay in seconds
        """
        for attempt in range(max_retries):
            try:
                return create_fn()
            except anthropic.RateLimitError:
                if attempt == max_retries - 1:
                    raise Exception(f"Rate limit exceeded after {max_retries} attempts. Please wait 1-2 minutes and try again.")

                # Exponential backoff: 5s, 10s, etc.
                delay = initial_delay * (2 ** attempt)
                print(f"Rate limit hit, waiting {delay} seconds before retry {attempt + 1}/{max_retries}...")
                time.sleep(delay)
            except Exception:
                raise
    
    def generate_brief(self, company_name, company_url, industry, contacts, 
                       call_type, context, specific_questions, progress=gr.Progress()):
        """
        Generate intelligence brief with progress updates
        """
        if not company_name:
            return "❌ Error: Company name is required"
        
        start_time = datetime.now()
        
        try:
            # Step 1: Company overview
            progress(0.15, desc="🔍 Researching company overview...")
            company_intel = self._research_company_overview(company_name, company_url, industry)
            progress(0.2, desc="⏱️ Managing rate limits (8s)...")
            time.sleep(8)  # Delay to stay under rate limits (Haiku is faster with better limits)

            # Step 2: Executive research (if provided)
            executive_intel = None
            if contacts and contacts.strip():
                progress(0.35, desc="🔍 Researching executives...")
                executive_intel = self._research_executives(contacts, company_name)
                progress(0.4, desc="⏱️ Managing rate limits (8s)...")
                time.sleep(8)  # Delay to stay under rate limits

            # Step 3: Tech stack + low-code signals
            progress(0.55, desc="🔍 Analyzing tech stack and low-code readiness...")
            tech_intel = self._research_tech_stack(company_name, company_url)
            progress(0.6, desc="⏱️ Managing rate limits (8s)...")
            time.sleep(8)  # Delay to stay under rate limits

            # Step 4: Strategic context
            progress(0.75, desc="🔍 Gathering strategic context...")
            strategic_intel = self._research_strategic_context(company_name)
            progress(0.8, desc="⏱️ Managing rate limits (8s)...")
            time.sleep(8)  # Delay to stay under rate limits

            # Step 5: Synthesize
            progress(0.9, desc="📝 Synthesizing OutSystems-focused brief...")
            final_brief = self._synthesize_outsystems_brief(
                company_name, company_intel, executive_intel,
                tech_intel, strategic_intel, call_type, context, specific_questions
            )
            
            end_time = datetime.now()
            generation_time = (end_time - start_time).total_seconds()
            
            progress(1.0, desc="✅ Brief complete!")
            
            # Add generation metadata
            metadata = f"\n\n---\n\n**Generated:** {datetime.now().strftime('%B %d, %Y at %I:%M %p')}\n"
            metadata += f"**Generation Time:** {int(generation_time)} seconds\n"
            metadata += f"**Model:** Claude 3.5 Haiku (Optimized for speed & cost)\n"
            
            return final_brief + metadata

        except anthropic.RateLimitError as e:
            return f"""❌ **Rate Limit Exceeded**

You've hit the API rate limit of 30,000 tokens per minute.

**What this means:** Your organization is sending too many tokens to the API too quickly.

**Solutions:**
1. **Wait 1-2 minutes** and try again
2. **Reduce scope:** Research fewer contacts or skip optional sections
3. **Spacing:** If generating multiple briefs, wait 2-3 minutes between each

**Technical details:** {str(e)}

This is temporary - your quota resets every minute."""

        except Exception as e:
            error_msg = str(e)
            if "rate_limit" in error_msg.lower() or "429" in error_msg:
                return f"""❌ **Rate Limit Exceeded**

You've hit the API rate limit. Please wait 1-2 minutes and try again.

**Tips to avoid this:**
- Generate one brief at a time
- Wait 2-3 minutes between briefs
- Reduce the number of contacts to research

**Error details:** {error_msg}"""
            else:
                return f"❌ **Error generating brief:**\n\n{error_msg}\n\nPlease check your API key and try again."
    
    def _research_company_overview(self, company_name, company_url, industry):
        """Research company basics with OutSystems lens"""
        
        url_context = f"\nWebsite: {company_url}" if company_url else ""
        industry_context = f"\nIndustry: {industry}" if industry else ""
        
        prompt = f"""You are researching a prospect for an OutSystems presales call. OutSystems is a leading low-code platform for enterprise application development.

Company: {company_name}{url_context}{industry_context}

Search the web and provide:
1. Company size (employees, revenue if available)
2. Headquarters location and key markets
3. Top 2-3 most relevant news items from past 90 days (with dates)
4. Brief company description (what they do, who they serve)
5. **Digital transformation signals**: Any mentions of digital transformation initiatives, legacy system modernization, application development challenges

Focus on information relevant to a low-code platform sales conversation.

**CRITICAL - Source Citations:**
- Include ALL source URLs as clickable markdown links: [Source Title](https://url.com)
- Add inline citations where relevant: [Source](URL)
- Cite sources for every fact mentioned

Be factual - if you can't find something, say "Not found in public sources."
Never infer or make up information."""

        response = self._call_api_with_retry(lambda: self.client.messages.create(
            model=self.model,
            max_tokens=2000,  # Reduced to lower token consumption
            temperature=0.3,
            tools=[{"type": "web_search_20250305", "name": "web_search"}],
            messages=[{"role": "user", "content": prompt}]
        ))

        return self._extract_text(response)

    def _research_executives(self, contacts_list, company_name):
        """Research executives with focus on digital/tech priorities"""
        
        contacts = []
        for line in contacts_list.strip().split('\n'):
            if ',' in line:
                parts = line.split(',', 1)
                contacts.append({
                    'name': parts[0].strip(),
                    'title': parts[1].strip() if len(parts) > 1 else 'Unknown'
                })
        
        if not contacts:
            return "No valid contacts provided."
        
        contact_intel = []
        for contact in contacts[:3]:  # Limit to 3
            prompt = f"""Research this executive for an OutSystems low-code platform sales call:

Name: {contact['name']}
Title: {contact['title']}
Company: {company_name}

Find:
1. Recent LinkedIn posts, articles, or interviews (past 30-60 days)
2. Their priorities related to: digital transformation, application development, IT modernization, developer productivity, automation
3. Any mentions of: low-code, no-code, legacy systems, technical debt, digital initiatives
4. Background that would help in a low-code platform conversation

Be factual. If you can't find recent activity, say so.

**Source Citations:** Include all URLs as markdown links [Title](URL)."""

            response = self._call_api_with_retry(lambda: self.client.messages.create(
                model=self.model,
                max_tokens=1500,  # Reduced to lower token consumption
                temperature=0.3,
                tools=[{"type": "web_search_20250305", "name": "web_search"}],
                messages=[{"role": "user", "content": prompt}]
            ))

            contact_intel.append({
                'contact': contact,
                'intelligence': self._extract_text(response)
            })

            # Longer delay between contact lookups to manage rate limits
            if len(contact_intel) < len(contacts[:3]):
                time.sleep(5)
        
        formatted = ""
        for item in contact_intel:
            formatted += f"\n### {item['contact']['name']} - {item['contact']['title']}\n"
            formatted += f"{item['intelligence']}\n"
        
        return formatted
    
    def _research_tech_stack(self, company_name, company_url):
        """Research tech stack with focus on low-code readiness indicators"""
        
        url_context = f"\nWebsite: {company_url}" if company_url else ""
        
        prompt = f"""Research the technology landscape for this prospect (OutSystems low-code platform sales context):

Company: {company_name}{url_context}

Find evidence through job postings, tech blog, news, and partnerships:

**CRITICAL - Low-Code/Automation Signals:**
- Any mentions of: low-code, no-code, citizen development, application platforms
- Automation initiatives, RPA, workflow automation
- Developer productivity challenges or initiatives

**Current Tech Stack:**
- Cloud providers (AWS/Azure/GCP)
- Databases and data platforms
- Enterprise apps (Salesforce, SAP, ServiceNow, etc.)
- Development platforms and tools
- Integration/API infrastructure

**Pain Point Signals:**
- Legacy system modernization needs
- Technical debt mentions
- Application backlog issues
- Shadow IT challenges
- Developer hiring difficulties

**Agentic AI / AI Automation:**
- Any AI initiatives mentioned
- Intelligent automation projects
- AI agent implementations
- Machine learning use cases

Be specific. Focus on signals relevant to low-code platform opportunity.

**Source Citations:** Include all URLs as markdown links [Title](URL) for every claim."""

        response = self._call_api_with_retry(lambda: self.client.messages.create(
            model=self.model,
            max_tokens=2000,  # Reduced to lower token consumption
            temperature=0.3,
            tools=[{"type": "web_search_20250305", "name": "web_search"}],
            messages=[{"role": "user", "content": prompt}]
        ))

        return self._extract_text(response)

    def _research_strategic_context(self, company_name):
        """Research strategic context relevant to OutSystems opportunities"""
        
        prompt = f"""Research strategic context for: {company_name}

Focus on finding (within past 12 months):

**Business Changes:**
- Funding rounds, M&A activity, IPO plans
- New C-suite appointments (especially CTO, CIO, CDO, VP Engineering)
- Strategic initiatives announced

**Digital Transformation Indicators:**
- Digital transformation programs announced
- IT modernization initiatives
- Cloud migration plans
- Innovation lab or digital center launches

**Market Pressures:**
- Competition requiring faster time-to-market
- Regulatory/compliance changes requiring new applications
- Customer experience improvement initiatives
- Operational efficiency drives

**Growth Signals:**
- Geographic expansion
- New product launches
- Customer base growth
- Market entry

Prioritize information showing need for rapid application development, digital transformation, or IT modernization.
Be factual.

**Source Citations:** Include all URLs as markdown links [Title](URL) with dates."""

        response = self._call_api_with_retry(lambda: self.client.messages.create(
            model=self.model,
            max_tokens=2000,  # Reduced to lower token consumption
            temperature=0.3,
            tools=[{"type": "web_search_20250305", "name": "web_search"}],
            messages=[{"role": "user", "content": prompt}]
        ))

        return self._extract_text(response)

    def _synthesize_outsystems_brief(self, company_name, company_intel, executive_intel, 
                                      tech_intel, strategic_intel, call_type, context, questions):
        """Synthesize into OutSystems-focused brief"""
        
        questions_section = ""
        if questions and questions.strip():
            questions_section = f"\n\nSPECIFIC QUESTIONS TO ADDRESS:\n{questions}"
        
        prompt = f"""You are synthesizing presales research for an OutSystems call. OutSystems is a leading low-code application platform that enables rapid enterprise application development.

RESEARCH GATHERED:

Company Overview:
{company_intel}

Executive Intelligence:
{executive_intel if executive_intel else "No specific contacts researched."}

Tech Stack & Low-Code Signals:
{tech_intel}

Strategic Context:
{strategic_intel}

OPPORTUNITY CONTEXT:
Call Type: {call_type}
Context: {context if context else "No additional context provided"}{questions_section}

Create an OutSystems-focused intelligence brief following this format:

# PRE-CALL INTELLIGENCE BRIEF - OUTSYSTEMS OPPORTUNITY
**Company:** {company_name}  
**Date:** {datetime.now().strftime('%B %d, %Y')}  
**Call Type:** {call_type}  
**Focus:** Low-Code Platform Opportunity

---

## 📋 EXECUTIVE SUMMARY
[2-3 sentences: What the SE needs to know. Focus on: their digital transformation stage, key pain points OutSystems solves, urgency/timing signals]

---

## 🏢 COMPANY SNAPSHOT
- **Industry:** [Industry/sector]
- **Size:** [Employees/revenue]
- **Headquarters:** [Location]
- **Digital Maturity:** [Based on research: early/mid/advanced digital transformation journey]
- **Recent News:** [1-2 most relevant items with dates]

---

## 👥 KEY CONTACTS
[For each contact:]
**[Name] - [Title]**
- Recent activity: [Posts/articles/priorities]
- Low-code relevance: [How their role relates to application development, digital transformation]
- Conversation angle: [How to connect their priorities to OutSystems value]

---

## 💻 TECHNOLOGY LANDSCAPE

### Current State
[Known tech stack, development platforms, cloud infrastructure]

### Low-Code Readiness Signals
[Any mentions of: low-code interest, citizen development, app backlog, developer shortage, shadow IT]

### Integration Requirements
[Key systems OutSystems would need to integrate with: Salesforce, SAP, legacy systems, etc.]

### Agentic AI Opportunities
[Any AI initiatives, automation needs, or intelligent workflow requirements where agentic AI + low-code could apply]

---

## 🎯 OUTSYSTEMS OPPORTUNITY ANALYSIS

### Primary Use Cases (Rank by Fit)
Based on research, identify top 2-3 use cases:
1. **[Use Case]**: [Why this fits - tie to specific research findings]
2. **[Use Case]**: [Evidence from research]
3. **[Use Case]**: [Rationale]

### Pain Points OutSystems Solves
1. **[Pain Point]**: [How OutSystems addresses this - be specific]
2. **[Pain Point]**: [OutSystems solution]
3. **[Pain Point]**: [Value proposition]

### Competitive Landscape
[Any mentions of: PowerApps, Mendix, Salesforce Platform, ServiceNow, or other low-code tools. If none found, note "No competing platforms found in research."]

---

## 💡 CONVERSATION STARTERS (OutSystems-Focused)

1. **[Specific question tied to their business/tech landscape]**  
   *Leads to: [How this opens discussion about low-code value]*

2. **[Question about digital transformation initiative or pain point]**  
   *Leads to: [How OutSystems addresses this]*

3. **[Question about specific technology or business challenge found in research]**  
   *Leads to: [Relevant OutSystems capability]*

---

## 🎭 LIKELY OBJECTIONS & RESPONSES

**Objection:** "We already have developers/existing tools"  
**Context from research:** [Any relevant info]  
**Response angle:** [How to position OutSystems as complementary]

**Objection:** "Low-code isn't enterprise-grade"  
**Response angle:** [Reference their specific enterprise needs found in research]

**Objection:** [Any other objection suggested by their tech stack/context]  
**Response angle:** [How to address]

---

## 🗺️ RECOMMENDED APPROACH

[2-3 paragraphs:
- How to open the conversation (based on research)
- Key themes to emphasize (speed, governance, enterprise-scale, AI integration)
- What to show/demo (specific to their use cases)
- How to position against their current state]

---

## 🚀 OUTSYSTEMS DIFFERENTIATION POINTS

Based on their context, emphasize:
- [ ] **Speed:** [Why this matters for them]
- [ ] **Governance:** [Enterprise control needs]
- [ ] **AI-Powered Development:** [If they have AI initiatives]
- [ ] **Full-Stack:** [If they need mobile + web + workflows]
- [ ] **Integration:** [Their complex integration needs]
- [ ] **Scale:** [Enterprise-grade requirements]

---

## 🚩 RED FLAGS / WATCH OUTS
[Anything concerning: competing vendor mentions, budget signals, timing issues, recent leadership chaos, skepticism about low-code]

If none: "No significant red flags identified."

---

## 📊 QUALIFICATION CRITERIA

**Budget Authority:** [Any signals about budget, spending authority, fiscal year]  
**Timeline:** [Urgency signals, project timelines mentioned]  
**Pain:** [Severity of challenges found - high/medium/low]  
**Fit:** [How well do their needs match OutSystems sweet spot]

---

## 📚 SOURCES
**CRITICAL:** List ALL source URLs as clickable markdown links in this format:
- [Article/Page Title](https://full-url.com) - Brief description
- [Company Press Release](https://url.com) - What it covers

Include every URL referenced in the research above.

---

## ⚠️ DISCLAIMER
**This brief was AI-generated for OutSystems presales. Verify all facts before use.**
- Check sources above
- Confirm technical details
- Trust your judgment over AI suggestions
- This is a starting point, not gospel

---

**CRITICAL RULES:**
1. Be SPECIFIC with OutSystems use cases - tie to their actual business
2. Only include information found in research - no generic low-code pitches
3. Every claim should reference source
4. If research was thin, say so and suggest questions to ask
5. Focus on THEIR problems and how OutSystems solves them
6. Make conversation starters highly actionable
7. Connect agentic AI opportunities to their business where relevant"""

        response = self._call_api_with_retry(lambda: self.client.messages.create(
            model=self.model,
            max_tokens=4000,  # Reduced to lower token consumption
            temperature=0.5,
            messages=[{"role": "user", "content": prompt}]
        ))

        return self._extract_text(response)
    
    def _extract_text(self, response):
        """Extract text from Claude API response"""
        result = ""
        for block in response.content:
            if block.type == "text":
                result += block.text
        return result


def export_to_pdf(markdown_content, company_name):
    """
    Export markdown content to PDF with professional styling

    Args:
        markdown_content: The markdown text to convert
        company_name: Name of company for filename

    Returns:
        Path to the generated PDF file
    """
    # Convert markdown to HTML
    html_content = markdown.markdown(markdown_content, extensions=['extra', 'nl2br'])

    # Add CSS styling for professional appearance
    styled_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            @page {{
                size: A4;
                margin: 2cm;
            }}
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 800px;
                margin: 0 auto;
            }}
            h1 {{
                color: #1a1a1a;
                border-bottom: 3px solid #4a90e2;
                padding-bottom: 10px;
                font-size: 24px;
            }}
            h2 {{
                color: #2c3e50;
                border-bottom: 2px solid #e0e0e0;
                padding-bottom: 8px;
                margin-top: 30px;
                font-size: 20px;
            }}
            h3 {{
                color: #34495e;
                font-size: 16px;
                margin-top: 20px;
            }}
            a {{
                color: #4a90e2;
                text-decoration: none;
            }}
            a:hover {{
                text-decoration: underline;
            }}
            ul, ol {{
                padding-left: 25px;
            }}
            li {{
                margin-bottom: 8px;
            }}
            code {{
                background-color: #f5f5f5;
                padding: 2px 6px;
                border-radius: 3px;
                font-family: 'Courier New', monospace;
            }}
            hr {{
                border: none;
                border-top: 1px solid #e0e0e0;
                margin: 20px 0;
            }}
            strong {{
                color: #2c3e50;
            }}
            blockquote {{
                border-left: 4px solid #4a90e2;
                padding-left: 15px;
                margin-left: 0;
                color: #666;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin: 15px 0;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }}
            th {{
                background-color: #f5f5f5;
            }}
        </style>
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """

    # Generate filename
    safe_company_name = "".join(c for c in company_name if c.isalnum() or c in (' ', '-', '_')).strip()
    safe_company_name = safe_company_name.replace(' ', '_')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"OutSystems_Brief_{safe_company_name}_{timestamp}.pdf"

    # Create temporary directory if needed and save PDF
    output_dir = tempfile.gettempdir()
    pdf_path = os.path.join(output_dir, filename)

    # Convert HTML to PDF
    HTML(string=styled_html).write_pdf(pdf_path)

    return pdf_path


def create_demo():
    """Create Gradio interface"""
    
    agent = OutSystemsIntelAgent()
    
    # Custom CSS for better styling
    custom_css = """
    .gradio-container {
        max-width: 1200px !important;
    }
    .warning-box {
        background-color: #fef3c7;
        border-left: 4px solid #f59e0b;
        padding: 16px;
        margin: 20px 0;
        border-radius: 8px;
    }
    """
    
    with gr.Blocks(css=custom_css, title="OutSystems Pre-Call Intel Agent") as demo:
        
        gr.Markdown("""
        # 🎯 OutSystems Pre-Call Intelligence Agent
        
        AI-powered prospect research optimized for **low-code platform** and **agentic AI** opportunities.
        
        Generate comprehensive intelligence briefs in 2-5 minutes.
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📝 Input Information")
                
                company_name = gr.Textbox(
                    label="Company Name *",
                    placeholder="e.g., Acme Financial Services",
                    info="Required"
                )
                
                company_url = gr.Textbox(
                    label="Company Website",
                    placeholder="https://www.company.com",
                    info="Optional but helpful"
                )
                
                industry = gr.Textbox(
                    label="Industry/Vertical",
                    placeholder="e.g., Financial Services, Healthcare, Retail"
                )
                
                contacts = gr.Textbox(
                    label="Key Contacts (Optional)",
                    placeholder="One per line: Name, Title\ne.g., John Smith, CTO\nJane Doe, VP Digital",
                    lines=3,
                    info="Limit to 3 for faster results"
                )
                
                call_type = gr.Dropdown(
                    label="Call Type",
                    choices=[
                        "Discovery Call",
                        "Demo",
                        "Technical Deep Dive",
                        "Executive Meeting",
                        "Closing Call"
                    ],
                    value="Discovery Call"
                )
                
                context = gr.Textbox(
                    label="Opportunity Context",
                    placeholder="What do you know? Previous conversations? Known pain points?",
                    lines=3
                )
                
                specific_questions = gr.Textbox(
                    label="Specific Questions (Optional)",
                    placeholder="Any specific things you want researched?",
                    lines=2
                )
                
                generate_btn = gr.Button("🚀 Generate Brief", variant="primary", size="lg")
                
                gr.Markdown("""
                <div class="warning-box">
                    <strong>⚠️ Important Notes</strong><br>
                    • This agent may make mistakes - always verify facts before using with customers<br>
                    • Generation takes <strong>90-120 seconds</strong> (optimized with Claude Haiku)<br>
                    • Wait 1-2 minutes between generating multiple briefs<br>
                    • Limit contacts to 1-2 for fastest results
                </div>
                """)
        
            with gr.Column(scale=1):
                gr.Markdown("### 📄 Intelligence Brief")

                output = gr.Markdown(
                    label="Generated Brief",
                    value="*Brief will appear here after generation...*"
                )

                with gr.Row():
                    export_pdf_btn = gr.Button("📄 Export to PDF", variant="secondary", size="sm", visible=False)

                pdf_output = gr.File(label="📥 Your PDF is ready! Click to download:", visible=False)
        
        # Examples
        gr.Markdown("### 💡 Example Inputs")
        gr.Examples(
            examples=[
                [
                    "Stripe",
                    "https://stripe.com",
                    "Fintech / Payments",
                    "Patrick Collison, CEO",
                    "Discovery Call",
                    "Inbound lead interested in internal tools development. Large engineering team.",
                    "Any low-code or automation initiatives?"
                ],
                [
                    "Mayo Clinic",
                    "https://www.mayoclinic.org",
                    "Healthcare",
                    "Cris Ross, CIO",
                    "Executive Meeting",
                    "Exploring digital transformation. Focus on patient experience applications.",
                    "What are their current application development challenges?"
                ]
            ],
            inputs=[company_name, company_url, industry, contacts, call_type, context, specific_questions]
        )
        
        # State to store the latest brief content
        brief_content = gr.State(value="")

        # Generate brief event handler
        def generate_and_store(comp_name, comp_url, ind, cont, call_t, ctx, questions):
            brief = agent.generate_brief(comp_name, comp_url, ind, cont, call_t, ctx, questions)
            # Show export button if brief was generated successfully
            show_export = not brief.startswith("❌")
            return brief, brief, comp_name, gr.update(visible=show_export), gr.update(visible=False)

        generate_btn.click(
            fn=generate_and_store,
            inputs=[company_name, company_url, industry, contacts, call_type, context, specific_questions],
            outputs=[output, brief_content, company_name, export_pdf_btn, pdf_output]
        )

        # PDF export event handler
        def export_brief_to_pdf(brief, comp_name):
            if not brief or brief == "*Brief will appear here after generation...*":
                return gr.update(visible=False, value=None)

            try:
                pdf_path = export_to_pdf(brief, comp_name if comp_name else "Company")
                return gr.update(visible=True, value=pdf_path)
            except Exception as e:
                print(f"Error exporting PDF: {e}")
                return gr.update(visible=False, value=None)

        export_pdf_btn.click(
            fn=export_brief_to_pdf,
            inputs=[brief_content, company_name],
            outputs=[pdf_output]
        )
        
        gr.Markdown("""
        ---
        
        ### 📋 Tips for Best Results

        ✅ **DO:**
        - Provide company website URL
        - Add opportunity context
        - List key contacts with titles
        - Verify all facts before using
        - **Export to PDF:** After generation, click the "Export to PDF" button, then click the download link that appears

        ❌ **DON'T:**
        - Trust it blindly
        - Skip fact-checking
        - Use for confidential companies
        
        ### 🎯 OutSystems Focus Areas
        
        This agent looks for signals related to:
        - Low-code platform readiness
        - Application development bottlenecks
        - Digital transformation initiatives
        - Legacy system modernization
        - Developer productivity challenges
        - Agentic AI opportunities
        - Workflow automation needs
        
        ---

        **Version:** 0.1 (MVP for Pilot)
        **Model:** Claude 3.5 Haiku (Fast & Cost-Optimized)
        **Focus:** OutSystems Low-Code + Agentic AI
        """)
    
    return demo


if __name__ == "__main__":
    # Check for API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("=" * 70)
        print("⚠️  ERROR: ANTHROPIC_API_KEY environment variable not set!")
        print("=" * 70)
        print("\nSet it with:")
        print("  export ANTHROPIC_API_KEY='your-key-here'  # Mac/Linux")
        print("  set ANTHROPIC_API_KEY=your-key-here       # Windows")
        print("\nOr create a .env file with:")
        print("  ANTHROPIC_API_KEY=your-key-here")
        print("=" * 70)
        exit(1)
    
    print("=" * 70)
    print("🚀 OutSystems Pre-Call Intelligence Agent")
    print("=" * 70)
    print("\nStarting Gradio interface...")
    print("Once launched, share the URL with your pilot team.\n")
    
    # Create and launch
    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",  # Accessible from network
        server_port=7860,
        share=True,  # Set to True to get public URL
        show_error=True
    )

