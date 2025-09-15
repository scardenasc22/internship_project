from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import ToolMessage, SystemMessage, HumanMessage

# generation of the criteria without any feedback
criteria_template = [
    (
        "system",
        (
            "You are an expert recruiter. Your task is to analyze a job description and extract the key skills and areas"
            "a candidate should be evaluated on. For each category, provide a concise list of strings. "
            "Limit each list to a maximum of 5 elements and avoid making tool specific criteria."
        ),
    ),
    (
        "human",
        (
            "Please provide a JSON object with the following keys. Each key's value should be a concise list of strings "
            "relevant to the job description provided.\n\n"
            "- **domains**: Key subject areas or industries relevant to the role.\n"
            "- **technical_skills**: Specific technologies, programming languages, and tools required.\n"
            "- **soft_skills**: Interpersonal abilities and qualities necessary for success at work.\n"
            "- **culture**: Traits that align with the company's culture and values.\n\n"
            "**Job Description:**\n{job_description}\n\n"
            "**Company Details:**\n{company_info}\n\n"
        ),
    ),
]
criteria_prompt = ChatPromptTemplate.from_messages(messages = criteria_template)

# generation of the criteria with feedback
criteria_with_feedback_template = [
    (
        "system",
        (
            "You are an expert recruiter. Your task is to analyze a job description and extract the key skills and areas "
            "a candidate should be evaluated on. For each category, provide a concise list of strings. "
            "Limit each list to a maximum of 5 elements and avoid making tool specific criteria."
        ),
    ),
    (
        "human",
        (
            "Please provide a JSON object with the following keys. Each key's value should be a concise list of strings "
            "relevant to the job description provided.\n\n"
            "- **domains**: Key subject areas or industries relevant to the role.\n"
            "- **technical_skills**: Specific technologies, programming languages, and tools required.\n"
            "- **soft_skills**: Interpersonal abilities and qualities necessary for success at work.\n"
            "- **values_and_culture**: Traits that align with the company's culture and values.\n\n"
            "**Job Description:**\n{job_description}\n\n"
            "**Company Details:**\n{company_info}\n\n"
            "Consider the following feedback when generating the critiera:\n"
            "**Feedback:**\n\n{feedback}"
        ),
    ),
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