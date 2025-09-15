from schemas import (
    Refinement, 
    WorkflowState, 
    EvaluationCriteria,
    CompanyDetails,
    EvaluationScores,
    WeightedEvaluationCriteria
)
from functions import (
    print_execution_status,
    text_extraction,
    scores_to_dataframe
)
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from prompts import (
    criteria_prompt, 
    criteria_with_feedback_prompt, 
    refinement_prompt, 
    company_details_request_prompt,
    scoring_prompt
)
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from typing import Literal
import os

# loading environment variables
load_dotenv()

# one of the multiple LLMs that can be leveraged according to the task
llm = ChatOpenAI(
    model = "gpt-4o-mini-2024-07-18"
)
# tools to search information on the web
web_search = TavilySearch(
    max_results = 3
)

### nodes
# 1. company details extraction
@print_execution_status
def get_company_info(state : WorkflowState) -> WorkflowState:
    """the agent extracts the company details"""
    # create the chain of response
    llm_extractor = llm.with_structured_output(schema = CompanyDetails)
    info_extraction_chain = company_details_request_prompt | llm_extractor
    # extract info from the job description
    initial_details = info_extraction_chain.invoke(
        {
            "job_description" : state.job_description
        }
    )
    # update the state of considering the extracted details
    state.company_info = initial_details
    # prepare the web search
    search_query = f"{initial_details.name}, {initial_details.sector}, profile"
    web_results = web_search.invoke({
        "query" : search_query
    })
    # extract the content of the websearch
    details = [f"Source {i + 1}: {info['content']}" for i, info in enumerate(web_results['results'])]
    # adding the internet search to the company description
    state.company_info.description += (
        f"Information from the internet:"
        "\n".join(details)
    )
    return state
    
# 2. criteria generation node
@print_execution_status
def criteria_generation(state : WorkflowState) -> WorkflowState:
    """the agent generates the criteria to evaluate each candidate"""
    llm_constrained = llm.with_structured_output(schema = EvaluationCriteria)
    # check if there is any feedback from previous interactions
    if not state.criteria_feedback:
        criteria_chain = criteria_prompt | llm_constrained
        msg = criteria_chain.invoke({
            "job_description" : state.job_description,
            "company_info" : (
                f"Company Name: {state.company_info.name}\n"
                f"Location: {state.company_info.location}\n"
                f"Sector: {state.company_info.sector}\n"
                f"Description: {state.company_info.description}\n"
            )
        })
    else:
        criteria_chain = criteria_with_feedback_prompt | llm_constrained
        msg = criteria_chain.invoke({
            "job_description" : state.job_description,
            "feedback" : state.criteria_feedback,
            "company_info" : (
                f"Company Name: {state.company_info.name}\n"
                f"Location: {state.company_info.location}\n"
                f"Sector: {state.company_info.sector}\n"
                f"Description: {state.company_info.description}\n"
            )
        })
    state.job_criteria = msg
    return state

# 3. feedback from another LLM
@print_execution_status
def llm_criteria_evaluation(state : WorkflowState) -> WorkflowState:
    """the agent evaluates the criteria for the role"""
    llm_constrained = llm.with_structured_output(schema = Refinement)
    refinement_chain = refinement_prompt | llm_constrained
    # concatenating the whole criteria into a single string
    criteria_string = (
        f"Domains: {state.job_criteria.domains}\n"
        f"Technical skills: {state.job_criteria.technical_skills}\n"
        f"Soft skills: {state.job_criteria.soft_skills}\n"
        f"Culture: {state.job_criteria.culture}\n"
    )
    # getting the feedback from the llm
    msg = refinement_chain.invoke({
        "job_description" : state.job_description,
        "criteria" : criteria_string
    })
    # updating the state of the graph
    state.criteria_feedback = msg.feedback
    state.criteria_status = msg.grade
    state.count += 1
    return state

# 4. human feedback node
@print_execution_status
def human_criteria_assessment(state : WorkflowState) -> WorkflowState:
    """Prompts the user to provide feedback on the listed criteria"""
    criteria_string = (
        f"Domains: {', '.join(state.job_criteria.domains)}\n"
        f"Technical skills: {', '.join(state.job_criteria.technical_skills)}\n"
        f"Soft skills: {', '.join(state.job_criteria.soft_skills)}\n"
        f"Culture: {', '.join(state.job_criteria.culture)}"
    )
    print(f"The job evaluation criteria is the following:")
    print("*"*30)
    print(f"{criteria_string}")
    print("*"*30)
    human_feedback = input("Do you approve this criteria? (yes/no). If not, provide feedback: ")
    if human_feedback.lower().strip() == "yes":
        print(f"Approved criteria!")
        # flush the feedback and change workflow state
        state.criteria_feedback = ""
        state.criteria_status = "good"
        state.count = 0 # we might need that later
    else:
        print(f"Looping back for refinement")
        state.criteria_feedback = human_feedback
        state.criteria_status = "needs improvement"
    return state

# 5. Collecting candidates info
@print_execution_status
def get_candidates_info(state : WorkflowState) -> WorkflowState:
    """Populate the information from each of the candidates in a dictionary"""
    try:
        if len(os.listdir(state.candidates_folder)) == 0:
            raise ValueError(f"the folder path: '{state.candidates_folder}' is empty")
        resumes_list = os.listdir(state.candidates_folder)
        state.candidates_dict = {
            file.split("-")[0] : text_extraction(os.path.join(state.candidates_folder, file)) for file in resumes_list
        }
    except Exception as e:
        raise Exception(f"Error while populating candidates info: {e}")
    return state

# 6. ranking candidates
@print_execution_status
def score_candidates(state : WorkflowState) -> WorkflowState:
    """Evaluate the candidates on the provided criteria"""
    llm_for_scores = llm.with_structured_output(schema = EvaluationScores)
    scoring_chain = scoring_prompt | llm_for_scores
    
    # Concatenating the criteria into a single list
    criteria = state.job_criteria
    all_criteria = (
        criteria.domains +
        criteria.technical_skills + 
        criteria.soft_skills + 
        criteria.culture
    )

    # Initialize a temporary dictionary to hold all scores
    all_candidates_scores = {}
    
    # Scoring the candidates
    for id, resume in state.candidates_dict.items():
        try:
            # Ask the llm to evaluate the candidate based on their resume
            scores_object = scoring_chain.invoke({
                "job_description" : state.job_description,
                "candidate_resume" : resume,
                "all_criteria" : ", ".join(all_criteria)
            })
            # Get the scores from the response
            scores_list = scores_object.scores
        except Exception as e:
            print(f"Error generating the scores for candidate {id}: {e}")
            scores_list = ["N/A"] * len(all_criteria)
            
        # Generate the dictionary of "criteria" : score pairs
        candidate_scores_dict = {
            criterion: score for criterion, score in zip(all_criteria, scores_list)
        }
        
        # Update the temporary dictionary with the current candidate's scores
        all_candidates_scores[id] = candidate_scores_dict
    
    # Finally, update the state with the complete dictionary of all candidates' scores
    state.candidates_scores = all_candidates_scores

    return state

# 7. exporting candidates scores to csv, adding the batches
@print_execution_status
def export_scores(state : WorkflowState) -> WorkflowState:
    """Exports candidate results with their corresponding batches"""
    scores_df = scores_to_dataframe(
        candidates_scores = state.candidates_scores,
        n_groups = state.batches
    )
    scores_df.to_csv(
        path_or_buf = os.path.join(state.scores_folder, "scores.csv"),
        index = False
    )
    return state

### routing nodes
# a. evaluation routing
@print_execution_status
def route_evalution(state : WorkflowState) -> Literal["accepted", "rejected"]:
    """Routes the evaluation based on the feedback and the counter"""
    if state.criteria_status == "good" or state.count > 3:
        return "accepted"
    else:
        return "rejected"