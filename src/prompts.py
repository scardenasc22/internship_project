from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import ToolMessage, SystemMessage, HumanMessage

# generation of the criteria without any feedback
criteria_template = [
    (
        "system",
        "You are an expert recruiter. Your task is to analyze a job description and extract the key skills and areas "
        "a candidate should be evaluated on. For each category, provide a concise list of strings and a list "
        "of the weights for each category.\n"
        "Limit each list to a maximum of 5 elements and avoid making tool specific criteria. "
        "The list of weights must add exactly 100 and the order must be the same as the specified categories. "
        "Ensure that there is no repetition of criteria across different categories."
    ),
    (
        "human",
        "Please provide a JSON object with the following keys. "
        "The first 4 values of the keys should be a concise list of 3 strings. " # from 5 to 3 components per criteria
        "The 'weights' key should be a list of weights of the previous 4 keys "
        "relevant to the job description provided.\n\n"
        "- **domains**: Key subject areas or industries relevant to the role.\n"
        "- **technical_skills**: Specific technologies, programming languages, and tools required.\n"
        "- **soft_skills**: Interpersonal abilities and qualities necessary for success at work.\n"
        "- **culture**: Traits that align with the company's culture and values.\n"
        "- **weights**: [<integer_percentage_domains>, <integer_percentage_technical_skills>, <integer_percentage_soft_skills>, <integer_percentage_culture>]\n\n"
        "**Job Description:**\n{job_description}\n\n"
        "**Company Details:**\n{company_info}\n\n"
    )
]
criteria_prompt = ChatPromptTemplate.from_messages(messages = criteria_template)

# generation of the criteria with feedback
criteria_with_feedback_template = [
    (
        "system",
        "You are an expert recruiter. Your task is to analyze a job description and extract the key skills and areas "
        "a candidate should be evaluated on. For each category, provide a concise list of strings and a list "
        "of the weights for each category.\n"
        "Limit each list to a maximum of 5 elements and avoid making tool specific criteria. "
        "The list of weights must add exactly 100 and the order must be the same as the specified categories. "
        "Ensure that there is no repetition of criteria across different categories."
    ),
    (
        "human",
        "Please provide a JSON object with the following keys. "
        "The first 4 values of the keys should be a concise list of 3 strings. " # from 5 to 3 components per criteria
        "The 'weights' key should be a list of weights of the previous 4 keys "
        "relevant to the job description provided.\n\n"
        "- **domains**: Key subject areas or industries relevant to the role.\n"
        "- **technical_skills**: Specific technologies, programming languages, and tools required.\n"
        "- **soft_skills**: Interpersonal abilities and qualities necessary for success at work.\n"
        "- **culture**: Traits that align with the company's culture and values.\n"
        "- **weights**: [<integer_percentage_domains>, <integer_percentage_technical_skills>, <integer_percentage_soft_skills>, <integer_percentage_culture>]\n\n"
        "**Job Description:**\n{job_description}\n\n"
        "**Company Details:**\n{company_info}\n\n"
        "Consider the following feedback when generating the critiera:\n"
        "**Feedback:**\n\n{feedback}"
    )
]
criteria_with_feedback_prompt = ChatPromptTemplate.from_messages(messages = criteria_with_feedback_template)

# criteria refinement template
refinement_template = [
    (
        "human",
        (
            "Please give me some feedback following job evaluation criteria considering the job description."
            "Consider also that very tool specific criteria are usually recommended.\n"
            "**job description:**\n\n{job_description}"
            "**evaluation criteria:**\n\n{criteria}"
        )
    )
]
refinement_prompt = ChatPromptTemplate.from_messages(messages = refinement_template)

# company details extraction
company_details_request_template = [
    (
        "human",
        (
            "Please extract the company name, location, sector, and a brief description from the following job description"
            "**job description:**\n\n{job_description}"
        )
    )
]
company_details_request_prompt = ChatPromptTemplate.from_messages(messages = company_details_request_template)

# prompt to score a candidate
scoring_template = [
    (
        "system",
        (
            "You are an expert recruiter. Your task is to analyze a candidate's resume and a job description to "
            "provide a score from 0 to 100 on each component of the evaluation criteria. "
            "Your output must be a single list of scores. The order of the scores must precisely match "
            "the order of the criteria provided."
        ),
    ),
    (
        "human",
        (
            "Provide a list of scores for the following candidate. Each score should be an integer from 0-100."
            "\n\n**Job Description:**\n{job_description}"
            "\n\n**Candidate Resume:**\n{candidate_resume}"
            "\n\n**Evaluation Criteria:**\n{all_criteria}"
        ),
    ),
]
scoring_prompt = ChatPromptTemplate.from_messages(messages=scoring_template)

# interview questions prompt
interview_questions_template = [
    (
        "system",
        (
            "You are an expert recruiter. Analyze the candidate's experience and the job description to "
            "produce interview questions and what to look for in responses. "
            "First, identify the role level (e.g., junior/entry-level, mid-level, senior, lead/principal) from the job description. "
            "Then tailor the experience and technical questions to match the expected seniority and complexity level of the role. "
            "Return content that can be cleanly mapped into the enforced schema: each item has exactly two fields: "
            "'question' and 'look_for'. Do not include any other fields or commentary."
        ),
    ),
    (
        "human",
        (
            "Generate exactly 4 experience questions, 4 situational questions, and 4 technical questions. "
            "Each item must include what to look for.\n\n"
            "**IMPORTANT: Role Level Tailoring**\n"
            "- First, identify the role level from the job description (junior, mid-level, senior, lead, etc.).\n"
            "- **Experience Questions**: Tailor complexity based on role level:\n"
            "  * Junior roles: Focus on learning, basic project involvement, and foundational experiences.\n"
            "  * Mid-level roles: Emphasize independent work, problem-solving, and project ownership.\n"
            "  * Senior roles: Target strategic thinking, leadership, mentoring, and complex project management.\n"
            "- **Technical Questions**: Adjust depth and scope based on role level:\n"
            "  * Junior roles: Cover fundamental concepts, basic tool usage, and learning approaches.\n"
            "  * Mid-level roles: Include intermediate concepts, best practices, and troubleshooting scenarios.\n"
            "  * Senior roles: Focus on architecture, design patterns, scalability, system design, and trade-offs.\n"
            "- **Situational Questions**: Should be appropriate for the role level but can range across all levels.\n\n"
            "Constraints:\n"
            "- Questions must be open-ended, specific, and grounded in the Job Description and Candidate Experience.\n"
            "- Avoid yes/no questions and avoid leading the candidate to a specific answer.\n"
            "- Do not invent details not present in the inputs; no placeholders or boilerplate.\n"
            "- No duplication within or across categories; each question should target a distinct skill/topic.\n"
            "- Keep 'look_for' concise (1-2 sentences), focusing on observable indicators, relevant metrics/examples, and common red flags.\n"
            "- Exclude illegal, discriminatory, or irrelevant topics.\n"
            "- Ensure questions match the seniority expectations: junior roles should not be asked senior-level architectural questions, "
            "and senior roles should not be asked basic tutorial-level questions.\n\n"
            "Category intents:\n"
            "- Experience Questions: Clarify or expand on the candidate's prior work, decisions, and outcomes. "
            "Depth and complexity should match the role level identified in the job description.\n"
            "- Situational Questions: Elicit behavior, problem-solving, communication, and stakeholder management.\n"
            "- Technical Questions: Assess tools, architectures, practices, and trade-offs relevant to the role. "
            "Technical depth should be calibrated to the role's seniority level.\n\n"
            "Return only content that fits the schema fields ('question', 'look_for') for each item.\n\n"
            "**Job Description:**\n{job_description}\n\n"
            "**Candidate Experience:**\n{candidate_experience}"
        ),
    ),
]
interview_questions_prompt = ChatPromptTemplate.from_messages(messages = interview_questions_template)

# experience extraction prompt
exp_extraction_template = [
    (
        "system",
        (
            "You are an expert recruiter. Your task is to extract all professional experience from a candidate's resume "
            "into a structured JSON format. Provide all job titles, company names, years of experience, and "
            "a concise list of responsibilities for each role."
        ),
    ),
    (
        "human",
        (
            "Please extract the professional experience from the following resume:\n\n{resume}"
        ),
    )
]
exp_extraction_prompt = ChatPromptTemplate.from_messages(messages = exp_extraction_template)

# scores rationale prompt
rationale_template = [
    (
        "system",
        "You are a talent acquisition analyst. Your task is to provide a brief rationale for the scores given to each candidate "
        "within a specific group of evaluation criteria. The rationale should be a concise summary of why "
        "each candidate received their specific score, drawing direct evidence from their resume to justify their ranking "
        "relative to the others in the same group."
    ),
    (
        "human",
        "Provide a comparative rationale for each candidate in the following group for the '{criteria_elements}' evaluation components."
        "\n\nThe scores of the candidate '{candidate_1_id}' are: {candidate_1_scores}\n"
        "The candidate resume is the following:\n{candidate_1_resume}\n\n"
        "\n\nThe scores of the candidate '{candidate_2_id}' are: {candidate_2_scores}\n"
        "The candidate resume is the following:\n{candidate_2_resume}\n\n"
    )
]
rationale_prompt = ChatPromptTemplate.from_messages(messages = rationale_template)

# tournament selection prompt
selection_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        (
            "You are an expert recruiter conducting a tournament selection round. "
            "Analyze the provided resumes and select the {number_of_candidates_to_select} best candidates who are most qualified based **ONLY** on the required skills listed below. "
            "For each winner, you MUST provide a concise justification (1-2 sentences) that highlights the most relevant skills/experience "
            "from their resume compared to the other candidates in the group. **Include the candidate's name in your justification when available.** "
            "Your output must strictly follow the required JSON schema."
        ),
    ),
    (
        "human",
        (
            "Based on the **REQUIRED EVALUATION CRITERIA** provided, select the best {number_of_candidates_to_select} winners "
            "from the candidates with the following IDs: {all_candidate_ids}."
            "\n\n**REQUIRED EVALUATION CRITERIA:**\n{combined_criteria_list}"
            "\n\n**Candidate Resumes:**\n{group_resumes}"
        ),
    ),
])

# strengths and weaknesses prompt
strengths_and_weaknesses_prompt = ChatPromptTemplate.from_messages(messages = [
    (
        "system",
        "You are an expert recruiter that helps the recruitment department in the evaluation "
        "of candidates for a given role. Your task is to provide insights regarding what are the "
        "candidate's strengths and weaknesses for the role in question, as well as a comprehensive overview "
        "of the candidate's background and experience."
    ),
    (
        "human",
        "Based on the following job description and the candidate resume, please provide:\n"
        "1. An overview: A comprehensive summary of the candidate's background, professional experience, "
        "key accomplishments, and overall profile (2-4 sentences).\n"
        "2. Exactly 3 strengths that the candidate has for the role in question.\n"
        "3. Exactly 3 weaknesses that the candidate has for the role in question.\n\n"
        "**Job Description:**\n{job_description}\n\n"
        "**Candidate Name:** {candidate_name}\n"
        "**Candidate Resume:**\n{candidate_resume}"
    )
])