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
        "The first 4 values of the keys should be a concise list of 5 strings. "
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
        "The first 4 values of the keys should be a concise list of 5 strings. "
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
            "You are an expert recruiter. Your task is to analyze a candidate's resume and a job description to "
            "generate a set of interview questions. Your output must be a list of questions and the name of the candidate."
        ),
    ),
    (
        "human",
        (
            "Please provide a JSON object with the following keys.\n"
            "- **Name**: <Name of the candidate>"
            "- **Experience questions**: <List of 4 questions aimed to clarify or expand upon the candidate's professional experience, skills, and achievements. These questions should help assess the candidate's expertise and past job performance.>\n"
            "- **Situational questions**: <List of 4 questions aimed to understand the candidate's behavior, problem-solving abilities, and soft skills in relation to the target role. These questions should explore how the candidate handles specific situations and challenges.>\n"
            "\n\n**Job Description:**\n{job_description}"
            "\n\n**Candidate Resume:**\n{candidate_resume}"
        )
    )
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